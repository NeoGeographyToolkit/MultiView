/* Copyright (c) 2021, United States Government, as represented by the
 * Administrator of the National Aeronautics and Space Administration.
 *
 * All rights reserved.
 *
 * The "ISAAC - Integrated System for Autonomous and Adaptive Caretaking
 * platform" software is licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

// TODO(oalexan1): Modularize this code!

// The algorithm:

// We assume our camera rig has n camera types. Each can be image or
// depth + image. One camera must be the reference camera. It is used
// to bracket other cameras in time. The camera rig acquires many
// sets of pictures.

// We assume we know the precise time every camera image is acquired.
// Every non-ref camera will be bracketed by two ref cameras very
// close in time. Hence, given the two bracketing ref camera poses,
// the ref cam pose will be interpolated at the time a non-ref camera
// is measured. This allows one to model the transform between
// the ref camera and every other camera on the rig.

// The variables to be optimized will be the pose of each ref camera,
// and the transforms from the ref camera to every other camera type
// (the extrinsics), with these transforms independent of time as the
// rig is rigid. Also optimized are the intrinsics of each camera, and
// the transform from each depth camera's cloud coordinates to its
// image coordinates (it is a transform very close to the identity but
// not quite, and a scale factor may be present).

// One component of the cost function to minimize measures the
// reprojection error in each camera, from each triangulated point in
// world coordinates. A second one measures the error between a
// triangulated point and corresponding depth measurement at that
// pixel, when applicable, with appropriate transforms applied to
// bring the depth measurement to world coordinates. This second
// error's strength is controlled by depth_tri_weight.

// Optionally, one can constrain that the triangulated points
// intersect close to a preexisting mesh, representing the surface
// being scanned with the rig given a previous estimation of all
// the camera poses. That mesh is computed using the geometry mapper.
// One also can control how close the depth camera clouds are to this
// mesh. The flags for this are mesh_tri_weight and depth_tri_weight,
// and can be set to 0 if desired not to use them.

// These mesh constraints bring in additional information,
// particularly for the cameras lacking depth, and may help get the
// focal lengths correctly.

// If no prior mesh exists or is not reliable enough, and one desires
// to not change the scale of the camera configuration too much
// (after registration), consider using the --tri_weight parameter,
// but setting it to just a small value to ensure it does not prevent
// the algorithm from converging.

// If different camera sensors are on different CPUs, and a time
// offset exists among their clocks, this program can model that,
// and also float those offsets, if desired.

// The initial ref camera poses are computed using Theia SfM. The
// obtained "sparse map" of poses must be registered to world
// coordinates, to get the world scale correctly.  The sparse map can
// be fixed or further refined in this tool.

// The initial extrinsics are assumed known, and are refined by this
// tool. Likely SfM can be used to get an initial value of the
// extrinsics, but for that may need to use the Theia tool which can
// do SfM with cameras acquired with different sensors.

// Every camera object (struct cameraImage) can look up its type,
// timestamp, timestamps and indices of bracketing cameras, image topic,
// depth topic (if present), ref_to_cam_timestamp_offset, and
// ref_to_cam_transform (extrinsics). A camera object also stores its
// image and depth cloud.

// For every instance of a reference camera its
// ref_to_cam_timestamp_offset is 0 and kept fixed,
// ref_to_cam_transform (extrinsics) is the identity and kept fixed,
// and the indices pointing to the left and right ref bracketing
// cameras are identical.

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <ceres/cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/dynamic_numeric_diff_cost_function.h>
#include <ceres/numeric_diff_cost_function.h>
#include <ceres/autodiff_cost_function.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>

#include <rig_calibrator/basic_algs.h>
#include <rig_calibrator/dense_map_utils.h>
#include <rig_calibrator/system_utils.h>
#include <rig_calibrator/transform_utils.h>
#include <rig_calibrator/interest_point.h>
#include <rig_calibrator/texture_processing.h>
#include <rig_calibrator/camera_image.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <Eigen/Geometry>
#include <Eigen/Core>

#include <oneapi/tbb/task_arena.h>
#include <boost/filesystem.hpp>

#include <string>
#include <map>
#include <iostream>
#include <fstream>

namespace fs = boost::filesystem;

DEFINE_double(robust_threshold, 3.0,
              "Residual pixel errors and 3D point residuals (the latter multiplied "
              "by corresponding weight) much larger than this will be "
              "exponentially attenuated to affect less the cost function.\n");

DEFINE_int32(num_iterations, 20, "How many solver iterations to perform in calibration.");

DEFINE_double(bracket_len, 0.6,
              "Lookup non-reference cam images only between consecutive ref cam images "
              "whose distance in time is no more than this (in seconds), after adjusting "
              "for the timestamp offset between these cameras. It is assumed the rig "
              "moves slowly and uniformly during this time. A large value here will "
              "make the calibrator compute a poor solution but a small value may prevent "
              "enough images being bracketed.");

DEFINE_string(intrinsics_to_float, "", "Specify which intrinsics to float for each sensor. "
              "Example: 'cam1:focal_length,optical_center,distortion cam2:focal_length'.");

DEFINE_string(rig_transforms_to_float, "",
              "Specify the names of sensors whose transforms to float, relative to the ref sensor. Use quotes around this string if it has spaces. Also can use comma as separator. Example: 'cam1 cam2'.");

// TODO(oalexan1): With the rig constraint on, only ref cam poses can float on their own,
// as the others are tied to it.
DEFINE_string(camera_poses_to_float, "",
              "Specify the cameras of which sensor types can have their poses floated. "
              "Allowing the cameras for all sensors types to float can "
              "invalidate the registration and scale (while making the overall "
              "configuration more internally consistent). Example: 'cam1 cam3'.");

DEFINE_string(depth_to_image_transforms_to_float, "",
              "Specify for which sensors to float the depth-to-image transform "
              "(if depth data exists). Example: 'cam1 cam3'.");

DEFINE_bool(float_scale, false,
            "If to optimize the scale of the clouds, part of depth-to-image transform. "
            "If kept fixed, the configuration of cameras should adjust to respect the given "
            "scale. This parameter should not be used with --affine_depth_to_image when the "
            "transform is affine, rather than rigid and a scale.");

DEFINE_bool(float_timestamp_offsets, false,
            "If to optimize the timestamp offsets among the cameras. This is experimental.");

DEFINE_double(timestamp_offsets_max_change, 1.0,
              "If floating the timestamp offsets, do not let them change by more than this "
              "(measured in seconds). Existing image bracketing acts as an additional "
              "constraint.");

DEFINE_double(depth_tri_weight, 0.0,
              "The weight to give to the constraint that depth measurements agree with "
              "triangulated points. Use a bigger number as depth errors are usually on the "
              "order of 0.01 meters while reprojection errors are on the order of 1 pixel.");

DEFINE_string(mesh, "",
              "Use this mesh to help constrain the calibration (in .ply format).");

DEFINE_double(mesh_tri_weight, 0.0,
              "A larger value will give more weight to the constraint that triangulated points "
              "stay close to a preexisting mesh. Not suggested by default.");

DEFINE_double(depth_mesh_weight, 0.0,
              "A larger value will give more weight to the constraint that the depth clouds "
              "stay close to the mesh. Not suggested by default.");

DEFINE_double(tri_weight, 0.0,
              "The weight to give to the constraint that optimized triangulated "
              "points stay close to original triangulated points. A positive value will "
              "help ensure the cameras do not move too far, but a large value may prevent "
              "convergence.");

DEFINE_double(tri_robust_threshold, 0.0,
              "The robust threshold to use with the triangulation weight. Must be positive.");

DEFINE_bool(affine_depth_to_image, false, "Assume that the depth-to-image transform "
            "for each depth + image camera is an arbitrary affine transform rather than a "
            "rotation times a scale.");

DEFINE_int32(calibrator_num_passes, 2, "How many passes of optimization to do. Outliers will be "
             "removed after every pass. Each pass will start with the previously optimized "
             "solution as an initial guess. Mesh intersections (if applicable) and ray "
             "triangulation will be recomputed before each pass.");

DEFINE_double(initial_max_reprojection_error, 300.0, "If filtering outliers, remove interest "
              "points for which the reprojection error, in pixels, is larger than this. This "
              "filtering happens when matches are created, before cameras are optimized, and "
              "a big value should be used if the initial cameras are not trusted.");

DEFINE_double(max_reprojection_error, 25.0, "If filtering outliers, remove interest points for "
              "which the reprojection error, in pixels, is larger than this. This filtering "
              "happens after each optimization pass finishes, unless disabled. It is better to not "
              "filter too aggressively unless confident of the solution.");

DEFINE_double(refiner_min_angle, 0.5, "If filtering outliers, remove triangulated points "
              "for which all rays converging to it make an angle (in degrees) less than this. "
              "Note that some cameras in the rig may be very close to each other relative to "
              "the triangulated points, so care is needed here.");

DEFINE_string(out_texture_dir, "", "If non-empty and if an input mesh was provided, "
              "project the camera images using the optimized poses onto the mesh "
              "and write the obtained .obj files in the given directory.");

DEFINE_double(min_ray_dist, 0.0, "The minimum search distance from a starting point along a ray "
              "when intersecting the ray with a mesh, in meters (if applicable).");

DEFINE_double(max_ray_dist, 100.0, "The maximum search distance from a starting point along a ray "
              "when intersecting the ray with a mesh, in meters (if applicable).");

DEFINE_bool(registration, false,
            "If true, and registration control points for the sparse map exist and are specified "
            "by --hugin_file and --xyz_file, register all camera poses and the rig transforms "
            "before starting the optimization. For now, the depth-to-image transforms do not "
            "change as result of this, which may be a problem. To apply the registration only, "
            "use zero iterations.");

DEFINE_string(hugin_file, "", "The path to the hugin .pto file used for registration.");

DEFINE_string(xyz_file, "", "The path to the xyz file used for registration.");

DEFINE_double(parameter_tolerance, 1e-12, "Stop when the optimization variables change by "
              "less than this.");

DEFINE_int32(num_opt_threads, 16, "How many threads to use in the optimization.");

DEFINE_int32(num_match_threads, 8, "How many threads to use in feature detection/matching. "
             "A large number can use a lot of memory.");

DEFINE_bool(no_rig, false,
            "Do not assumes the cameras are on a rig. Hence the pose of any "
            "camera of any sensor type may vary on its own and not being tied to other sensor "
            "types. See also --camera_poses_to_float.");

DEFINE_string(out_dir, "",
              "Save in this directory the camera intrinsics and extrinsics. "
              "See also --save-images_and_depth_clouds, --save-matches, --verbose, and --in_dir.");

DEFINE_string(rig_config, "",
              "Read the rig configuration from this file.");

DEFINE_string(nvm, "",
              "Read images and camera poses from this nvm file, as exported by Theia.");

DEFINE_bool(no_nvm_matches, false,
            "Do not read interest point matches from the nvm file. So read only camera poses. "
            "This implies --num_overlaps is positive, to be able to find new matches.");

DEFINE_string(camera_poses, "",
              "Read the images and world-to-camera poses from this list. "
              "The same format is used when this tool saves the updated "
              "poses in the output directory. It is preferred to read the camera "
              "poses with the ``--nvm`` option, as then interest point matches will "
              "be read as well.");  

DEFINE_int32(num_overlaps, 0, "Match an image with this many images (of all camera types) "
             "following it in increasing order of timestamp value. Set to a positive value "
             "only if desired to find more interest point matches than read from the input "
             "nvm file. Not suggested by default. For advanced controls, "
             "run: rig_calibrator --help | grep -i sift.");

DEFINE_bool(use_initial_rig_transforms, false,
            "Use the transforms among the sensors of the rig specified via --rig_config. "
            "Otherwise derive it from the poses of individual cameras.");

DEFINE_bool(save_nvm, false,
            "Save the optimized camera poses and inlier interest point matches as "
            "<out dir>/cameras.nvm. Interest point matches are offset relative to the optical "
            "center, to be consistent with Theia. This file can be passed in to a new invocation "
            "of this tool via --nvm.");

DEFINE_bool(save_matches, false,
            "Save the interest point matches (all matches and inlier matches, after filtering). "
            "Stereo Pipeline's viewer can be used for visualizing these.");

DEFINE_bool(export_to_voxblox, false,
            "Save the depth clouds and optimized transforms needed to create a mesh with voxblox "
            "(if depth clouds exist).");

DEFINE_bool(save_transformed_depth_clouds, false,
            "Save the depth clouds with the camera transform applied to them to make "
            "them be in world coordinates.");

DEFINE_bool(verbose, false,
            "Print a lot of verbose information about how matching goes.");

namespace dense_map {

// TODO(oalexan1): Move to transform_utils.cc.
  
Eigen::Affine3d calc_interp_world_to_ref(const double* beg_world_to_ref_t,
                                         const double* end_world_to_ref_t,
                                         double beg_ref_stamp,
                                         double end_ref_stamp,
                                         double ref_to_cam_offset,
                                         double cam_stamp) {
    Eigen::Affine3d beg_world_to_ref_aff;
    array_to_rigid_transform(beg_world_to_ref_aff, beg_world_to_ref_t);

    Eigen::Affine3d end_world_to_ref_aff;
    array_to_rigid_transform(end_world_to_ref_aff, end_world_to_ref_t);

    // Handle the degenerate case
    if (end_ref_stamp == beg_ref_stamp) 
      return beg_world_to_ref_aff;
    
    // Covert from cam time to ref time and normalize. It is very
    // important that below we subtract the big numbers from each
    // other first, which are the timestamps, then subtract whatever
    // else is necessary. Otherwise we get problems with numerical
    // precision with CERES.
    double alpha = ((cam_stamp - beg_ref_stamp) - ref_to_cam_offset)
        / (end_ref_stamp - beg_ref_stamp);
    
    if (alpha < 0.0 || alpha > 1.0) LOG(FATAL) << "Out of bounds in interpolation.\n";

    // Interpolate at desired time
    Eigen::Affine3d interp_world_to_ref_aff = dense_map::linearInterp(alpha, beg_world_to_ref_aff,
                                                                      end_world_to_ref_aff);

    return interp_world_to_ref_aff;
}
  
// Calculate interpolated world to camera transform. Use the
// convention that if beg_ref_stamp == end_ref_stamp, then this is the
// reference camera, and then only beg_world_to_ref_t is used, while
// end_world_to_ref_t is undefined. For the reference camera it is
// also expected that ref_to_cam_aff is the identity. This saves some
// code duplication later as the ref cam need not be treated
// separately.
Eigen::Affine3d calc_world_to_cam_trans(const double* beg_world_to_ref_t,
                                        const double* end_world_to_ref_t,
                                        const double* ref_to_cam_trans,
                                        double beg_ref_stamp,
                                        double end_ref_stamp,
                                        double ref_to_cam_offset,
                                        double cam_stamp) {

  Eigen::Affine3d interp_world_to_cam_aff;
  if (beg_ref_stamp == end_ref_stamp) {
    Eigen::Affine3d beg_world_to_ref_aff;
    array_to_rigid_transform(beg_world_to_ref_aff, beg_world_to_ref_t);
    interp_world_to_cam_aff = beg_world_to_ref_aff;
  } else {

    Eigen::Affine3d ref_to_cam_aff;
    array_to_rigid_transform(ref_to_cam_aff, ref_to_cam_trans);

    Eigen::Affine3d interp_world_to_ref_aff =
      calc_interp_world_to_ref(beg_world_to_ref_t, end_world_to_ref_t,  
                               beg_ref_stamp,  
                               end_ref_stamp,  ref_to_cam_offset,  
                               cam_stamp);
    
    interp_world_to_cam_aff = ref_to_cam_aff * interp_world_to_ref_aff;
  }

  return interp_world_to_cam_aff;
}

// TODO(oalexan1): Move to a separate file named costFunctions.h

ceres::LossFunction* GetLossFunction(std::string cost_fun, double th) {
  // Convert to lower-case
  std::transform(cost_fun.begin(), cost_fun.end(), cost_fun.begin(), ::tolower);

  ceres::LossFunction* loss_function = NULL;
  if (cost_fun == "l2")
    loss_function = NULL;
  else if (cost_fun == "huber")
    loss_function = new ceres::HuberLoss(th);
  else if (cost_fun == "cauchy")
    loss_function = new ceres::CauchyLoss(th);
  else if (cost_fun == "l1")
    loss_function = new ceres::SoftLOneLoss(th);
  else
    LOG(FATAL) << "Unknown cost function: " + cost_fun;

  return loss_function;
}

// TODO(oalexan1): Move to a separate file named costFunctions.h
  
// An error function minimizing the error of projecting
// an xyz point into a camera that is bracketed by
// two reference cameras. The precise timestamp offset
// between them is also floated.
struct BracketedCamError {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BracketedCamError(Eigen::Vector2d const& meas_dist_pix,
                    double left_ref_stamp, double right_ref_stamp, double cam_stamp,
                    std::vector<int> const& block_sizes,
                    camera::CameraParameters const& cam_params):
    m_meas_dist_pix(meas_dist_pix),
    m_left_ref_stamp(left_ref_stamp),
    m_right_ref_stamp(right_ref_stamp),
    m_cam_stamp(cam_stamp),
    m_block_sizes(block_sizes),
    m_cam_params(cam_params),
    m_num_focal_lengths(1) {
    // Sanity check
    if (m_block_sizes.size() != 8 || m_block_sizes[0] != NUM_RIGID_PARAMS ||
        m_block_sizes[1] != NUM_RIGID_PARAMS || m_block_sizes[2] != NUM_RIGID_PARAMS ||
        m_block_sizes[3] != NUM_XYZ_PARAMS || m_block_sizes[4] != NUM_SCALAR_PARAMS ||
        m_block_sizes[5] != m_num_focal_lengths || m_block_sizes[6] != NUM_OPT_CTR_PARAMS ||
        m_block_sizes[7] != 1  // This will be overwritten shortly
    ) {
      LOG(FATAL) << "BracketedCamError: The block sizes were not set up properly.\n";
    }

    // Set the correct distortion size. This cannot be done in the interface for now.
    m_block_sizes[7] = m_cam_params.GetDistortion().size();
  }

  // Call to work with ceres::DynamicNumericDiffCostFunction.
  bool operator()(double const* const* parameters, double* residuals) const {
    Eigen::Affine3d world_to_cam_trans =
      calc_world_to_cam_trans(parameters[0],  // beg_world_to_ref_t
                              parameters[1],  // end_world_to_ref_t
                              parameters[2],  // ref_to_cam_trans
                              m_left_ref_stamp, m_right_ref_stamp,
                              parameters[4][0],  // ref_to_cam_offset
                              m_cam_stamp);

    // World point
    Eigen::Vector3d X(parameters[3][0], parameters[3][1], parameters[3][2]);

    // Make a deep copy which we will modify
    camera::CameraParameters cam_params = m_cam_params;
    Eigen::Vector2d focal_vector = Eigen::Vector2d(parameters[5][0], parameters[5][0]);
    Eigen::Vector2d optical_center(parameters[6][0], parameters[6][1]);
    Eigen::VectorXd distortion(m_block_sizes[7]);
    for (int i = 0; i < m_block_sizes[7]; i++) distortion[i] = parameters[7][i];
    cam_params.SetFocalLength(focal_vector);
    cam_params.SetOpticalOffset(optical_center);
    cam_params.SetDistortion(distortion);

    // Convert world point to given cam coordinates
    X = world_to_cam_trans * X;

    // Project into the image
    Eigen::Vector2d undist_pix = cam_params.GetFocalVector().cwiseProduct(X.hnormalized());
    Eigen::Vector2d curr_dist_pix;
    cam_params.Convert<camera::UNDISTORTED_C, camera::DISTORTED>(undist_pix, &curr_dist_pix);

    // Compute the residuals
    residuals[0] = curr_dist_pix[0] - m_meas_dist_pix[0];
    residuals[1] = curr_dist_pix[1] - m_meas_dist_pix[1];

    return true;
  }

  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction*
  Create(Eigen::Vector2d const& meas_dist_pix, double left_ref_stamp, double right_ref_stamp,
         double cam_stamp, std::vector<int> const& block_sizes,
         camera::CameraParameters const& cam_params) {
    ceres::DynamicNumericDiffCostFunction<BracketedCamError>* cost_function =
      new ceres::DynamicNumericDiffCostFunction<BracketedCamError>
      (new BracketedCamError(meas_dist_pix, left_ref_stamp, right_ref_stamp,
                             cam_stamp, block_sizes, cam_params));

    cost_function->SetNumResiduals(NUM_PIX_PARAMS);

    // The camera wrapper knows all of the block sizes to add, except
    // for distortion, which is last
    for (size_t i = 0; i + 1 < block_sizes.size(); i++)  // note the i + 1
      cost_function->AddParameterBlock(block_sizes[i]);

    // The distortion block size is added separately as it is variable
    cost_function->AddParameterBlock(cam_params.GetDistortion().size());

    return cost_function;
  }

 private:
  Eigen::Vector2d m_meas_dist_pix;             // Measured distorted current camera pixel
  double m_left_ref_stamp, m_right_ref_stamp;  // left and right ref cam timestamps
  double m_cam_stamp;                          // Current cam timestamp
  std::vector<int> m_block_sizes;
  camera::CameraParameters m_cam_params;
  int m_num_focal_lengths;
};  // End class BracketedCamError

// TODO(oalexan1): Move to future costFunctions.h
  
// An error function minimizing the product of a given weight and the
// error between a triangulated point and a measured depth point. The
// depth point needs to be transformed to world coordinates first. For
// that one has to do pose interpolation.
struct BracketedDepthError {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BracketedDepthError(double weight, Eigen::Vector3d const& meas_depth_xyz,
                      double left_ref_stamp, double right_ref_stamp, double cam_stamp,
                      std::vector<int> const& block_sizes):
    m_weight(weight),
    m_meas_depth_xyz(meas_depth_xyz),
    m_left_ref_stamp(left_ref_stamp),
    m_right_ref_stamp(right_ref_stamp),
    m_cam_stamp(cam_stamp),
    m_block_sizes(block_sizes) {
    // Sanity check
    if (m_block_sizes.size() != 7 ||
        m_block_sizes[0] != NUM_RIGID_PARAMS  ||
        m_block_sizes[1] != NUM_RIGID_PARAMS  ||
        m_block_sizes[2] != NUM_RIGID_PARAMS  ||
        (m_block_sizes[3] != NUM_RIGID_PARAMS  && m_block_sizes[3] != NUM_AFFINE_PARAMS) ||
        m_block_sizes[4] != NUM_SCALAR_PARAMS ||
        m_block_sizes[5] != NUM_XYZ_PARAMS    ||
        m_block_sizes[6] != NUM_SCALAR_PARAMS) {
      LOG(FATAL) << "BracketedDepthError: The block sizes were not set up properly.\n";
    }
  }

  // Call to work with ceres::DynamicNumericDiffCostFunction.
  bool operator()(double const* const* parameters, double* residuals) const {
    // Current world to camera transform
    Eigen::Affine3d world_to_cam_trans =
      calc_world_to_cam_trans(parameters[0],  // beg_world_to_ref_t
                              parameters[1],  // end_world_to_ref_t
                              parameters[2],  // ref_to_cam_trans
                              m_left_ref_stamp, m_right_ref_stamp,
                              parameters[6][0],  // ref_to_cam_offset
                              m_cam_stamp);

    // The current transform from the depth point cloud to the camera image
    Eigen::Affine3d depth_to_image;
    if (m_block_sizes[3] == NUM_AFFINE_PARAMS)
      array_to_affine_transform(depth_to_image, parameters[3]);
    else
      array_to_rigid_transform(depth_to_image, parameters[3]);

    // Apply the scale
    double depth_to_image_scale = parameters[4][0];
    depth_to_image.linear() *= depth_to_image_scale;

    // Convert from depth cloud coordinates to cam coordinates
    Eigen::Vector3d M = depth_to_image * m_meas_depth_xyz;

    // Convert to world coordinates
    M = world_to_cam_trans.inverse() * M;

    // Triangulated world point
    Eigen::Vector3d X(parameters[5][0], parameters[5][1], parameters[5][2]);

    // Compute the residuals
    for (size_t it = 0; it < NUM_XYZ_PARAMS; it++) {
      residuals[it] = m_weight * (X[it] - M[it]);
    }

    return true;
  }

  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(double weight, Eigen::Vector3d const& meas_depth_xyz,
                                     double left_ref_stamp, double right_ref_stamp,
                                     double cam_stamp, std::vector<int> const& block_sizes) {
    ceres::DynamicNumericDiffCostFunction<BracketedDepthError>* cost_function =
      new ceres::DynamicNumericDiffCostFunction<BracketedDepthError>
      (new BracketedDepthError(weight, meas_depth_xyz, left_ref_stamp, right_ref_stamp,
                             cam_stamp, block_sizes));

    // The residual size is always the same.
    cost_function->SetNumResiduals(NUM_XYZ_PARAMS);

    for (size_t i = 0; i < block_sizes.size(); i++)
      cost_function->AddParameterBlock(block_sizes[i]);

    return cost_function;
  }

 private:
  double m_weight;                             // How much weight to give to this constraint
  Eigen::Vector3d m_meas_depth_xyz;            // Measured depth measurement
  double m_left_ref_stamp, m_right_ref_stamp;  // left and right ref cam timestamps
  double m_cam_stamp;                          // Current cam timestamp
  std::vector<int> m_block_sizes;
};  // End class BracketedDepthError

// An error function minimizing the product of a given weight and the
// error between a mesh point and a transformed measured depth point. The
// depth point needs to be transformed to world coordinates first. For
// that one has to do pose interpolation.
struct BracketedDepthMeshError {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BracketedDepthMeshError(double weight,
                          Eigen::Vector3d const& meas_depth_xyz,
                          Eigen::Vector3d const& mesh_xyz,
                          double left_ref_stamp, double right_ref_stamp, double cam_stamp,
                          std::vector<int> const& block_sizes):
    m_weight(weight),
    m_meas_depth_xyz(meas_depth_xyz),
    m_mesh_xyz(mesh_xyz),
    m_left_ref_stamp(left_ref_stamp),
    m_right_ref_stamp(right_ref_stamp),
    m_cam_stamp(cam_stamp),
    m_block_sizes(block_sizes) {
    // Sanity check
    if (m_block_sizes.size() != 6 ||
        m_block_sizes[0] != NUM_RIGID_PARAMS  ||
        m_block_sizes[1] != NUM_RIGID_PARAMS  ||
        m_block_sizes[2] != NUM_RIGID_PARAMS  ||
        (m_block_sizes[3] != NUM_RIGID_PARAMS  && m_block_sizes[3] != NUM_AFFINE_PARAMS) ||
        m_block_sizes[4] != NUM_SCALAR_PARAMS ||
        m_block_sizes[5] != NUM_SCALAR_PARAMS) {
      LOG(FATAL) << "BracketedDepthMeshError: The block sizes were not set up properly.\n";
    }
  }

  // Call to work with ceres::DynamicNumericDiffCostFunction.
  bool operator()(double const* const* parameters, double* residuals) const {
    // Current world to camera transform
    Eigen::Affine3d world_to_cam_trans =
      calc_world_to_cam_trans(parameters[0],  // beg_world_to_ref_t
                              parameters[1],  // end_world_to_ref_t
                              parameters[2],  // ref_to_cam_trans
                              m_left_ref_stamp, m_right_ref_stamp,
                              parameters[5][0],  // ref_to_cam_offset
                              m_cam_stamp);

    // The current transform from the depth point cloud to the camera image
    Eigen::Affine3d depth_to_image;
    if (m_block_sizes[3] == NUM_AFFINE_PARAMS)
      array_to_affine_transform(depth_to_image, parameters[3]);
    else
      array_to_rigid_transform(depth_to_image, parameters[3]);

    // Apply the scale
    double depth_to_image_scale = parameters[4][0];
    depth_to_image.linear() *= depth_to_image_scale;

    // Convert from depth cloud coordinates to cam coordinates
    Eigen::Vector3d M = depth_to_image * m_meas_depth_xyz;

    // Convert to world coordinates
    M = world_to_cam_trans.inverse() * M;

    // Compute the residuals
    for (size_t it = 0; it < NUM_XYZ_PARAMS; it++) {
      residuals[it] = m_weight * (m_mesh_xyz[it] - M[it]);
    }

    return true;
  }

  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(double weight,
                                     Eigen::Vector3d const& meas_depth_xyz,
                                     Eigen::Vector3d const& mesh_xyz,
                                     double left_ref_stamp, double right_ref_stamp,
                                     double cam_stamp, std::vector<int> const& block_sizes) {
    ceres::DynamicNumericDiffCostFunction<BracketedDepthMeshError>* cost_function =
      new ceres::DynamicNumericDiffCostFunction<BracketedDepthMeshError>
      (new BracketedDepthMeshError(weight, meas_depth_xyz, mesh_xyz,
                                   left_ref_stamp, right_ref_stamp,
                                   cam_stamp, block_sizes));

    // The residual size is always the same.
    cost_function->SetNumResiduals(NUM_XYZ_PARAMS);

    for (size_t i = 0; i < block_sizes.size(); i++)
      cost_function->AddParameterBlock(block_sizes[i]);

    return cost_function;
  }

 private:
  double m_weight;                             // How much weight to give to this constraint
  Eigen::Vector3d m_meas_depth_xyz;            // Measured depth measurement
  Eigen::Vector3d m_mesh_xyz;                  // Point on preexisting mesh
  double m_left_ref_stamp, m_right_ref_stamp;  // left and right ref cam timestamps
  double m_cam_stamp;                          // Current cam timestamp
  std::vector<int> m_block_sizes;
};  // End class BracketedDepthMeshError

// An error function minimizing a weight times the distance from a
// variable xyz point to a fixed reference xyz point.
struct XYZError {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  XYZError(Eigen::Vector3d const& ref_xyz, std::vector<int> const& block_sizes, double weight)
      : m_ref_xyz(ref_xyz), m_block_sizes(block_sizes), m_weight(weight) {
    // Sanity check
    if (m_block_sizes.size() != 1 || m_block_sizes[0] != NUM_XYZ_PARAMS)
      LOG(FATAL) << "XYZError: The block sizes were not set up properly.\n";
  }

  // Call to work with ceres::DynamicNumericDiffCostFunction.
  // Takes array of arrays as parameters.
  // TODO(oalexan1): May want to use the analytical Ceres cost function
  bool operator()(double const* const* parameters, double* residuals) const {
    // Compute the residuals
    for (int it = 0; it < NUM_XYZ_PARAMS; it++)
      residuals[it] = m_weight * (parameters[0][it] - m_ref_xyz[it]);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(Eigen::Vector3d const& ref_xyz,
                                     std::vector<int> const& block_sizes,
                                     double weight) {
    ceres::DynamicNumericDiffCostFunction<XYZError>* cost_function =
      new ceres::DynamicNumericDiffCostFunction<XYZError>
      (new XYZError(ref_xyz, block_sizes, weight));

    // The residual size is always the same
    cost_function->SetNumResiduals(NUM_XYZ_PARAMS);

    // The camera wrapper knows all of the block sizes to add.
    for (size_t i = 0; i < block_sizes.size(); i++) {
      cost_function->AddParameterBlock(block_sizes[i]);
    }
    return cost_function;
  }

 private:
  Eigen::Vector3d m_ref_xyz;  // reference xyz
  std::vector<int> m_block_sizes;
  double m_weight;
};  // End class XYZError

void calc_residuals_stats(std::vector<double> const& residuals,
                          std::vector<std::string> const& residual_names,
                          std::string const& tag) {
  size_t num = residuals.size();

  if (num != residual_names.size())
    LOG(FATAL) << "There must be as many residuals as residual names.";

  std::map<std::string, std::vector<double>> stats;
  for (size_t it = 0; it < residuals.size(); it++)
    stats[residual_names[it]] = std::vector<double>();  // initialize

  for (size_t it = 0; it < residuals.size(); it++)
    stats[residual_names[it]].push_back(std::abs(residuals[it]));

  std::cout << "The 25, 50, 75, and 100th percentile residual stats " << tag << std::endl;
  for (auto it = stats.begin(); it != stats.end(); it++) {
    std::string const& name = it->first;
    std::vector<double> vals = stats[name];  // make a copy
    std::sort(vals.begin(), vals.end());

    int len = vals.size();

    int it1 = static_cast<int>(0.25 * len);
    int it2 = static_cast<int>(0.50 * len);
    int it3 = static_cast<int>(0.75 * len);
    int it4 = static_cast<int>(len - 1);

    if (len == 0)
      std::cout << name << ": " << "none";
    else
      std::cout << std::setprecision(5)
                << name << ": " << vals[it1] << ' ' << vals[it2] << ' '
                << vals[it3] << ' ' << vals[it4];
    std::cout << " (" << len << " residuals)" << std::endl;
  }
}

// Compute the transforms from the world to every camera, based on the rig transforms
void calc_world_to_cam_using_rig(// Inputs
                                  std::vector<dense_map::cameraImage> const& cams,
                                  std::vector<double> const& world_to_ref_vec,
                                  std::vector<double> const& ref_timestamps,
                                  std::vector<double> const& ref_to_cam_vec,
                                  std::vector<double> const& ref_to_cam_timestamp_offsets,
                                  // Output
                                  std::vector<Eigen::Affine3d>& world_to_cam) {
  if (ref_to_cam_vec.size() / dense_map::NUM_RIGID_PARAMS != ref_to_cam_timestamp_offsets.size())
    LOG(FATAL) << "Must have as many transforms to reference as timestamp offsets.\n";
  if (world_to_ref_vec.size() / dense_map::NUM_RIGID_PARAMS != ref_timestamps.size())
    LOG(FATAL) << "Must have as many reference timestamps as reference cameras.\n";

  world_to_cam.resize(cams.size());

  for (size_t it = 0; it < cams.size(); it++) {
    int beg_index = cams[it].beg_ref_index;
    int end_index = cams[it].end_ref_index;
    int cam_type = cams[it].camera_type;
    world_to_cam[it] = dense_map::calc_world_to_cam_trans
      (&world_to_ref_vec[dense_map::NUM_RIGID_PARAMS * beg_index],
       &world_to_ref_vec[dense_map::NUM_RIGID_PARAMS * end_index],
       &ref_to_cam_vec[dense_map::NUM_RIGID_PARAMS * cam_type],
       ref_timestamps[beg_index], ref_timestamps[end_index],
       ref_to_cam_timestamp_offsets[cam_type],
       cams[it].timestamp);
  }
  return;
}

// A version of the above with the data stored differently
void calc_world_to_cam_using_rig(// Inputs
                                 std::vector<dense_map::cameraImage> const& cams,
                                 std::vector<Eigen::Affine3d> const& world_to_ref,
                                 std::vector<double> const& ref_timestamps,
                                 std::vector<Eigen::Affine3d> const& ref_to_cam,
                                 std::vector<double> const& ref_to_cam_timestamp_offsets,
                                 // Output
                                 std::vector<Eigen::Affine3d>& world_to_cam) {
  
  int num_cam_types = ref_to_cam.size();
  std::vector<double> ref_to_cam_vec(num_cam_types * dense_map::NUM_RIGID_PARAMS);
  for (int cam_type = 0; cam_type < num_cam_types; cam_type++)
    dense_map::rigid_transform_to_array
      (ref_to_cam[cam_type], &ref_to_cam_vec[dense_map::NUM_RIGID_PARAMS * cam_type]);

  int num_ref_cams = world_to_ref.size();
  if (world_to_ref.size() != ref_timestamps.size())
    LOG(FATAL) << "Must have as many ref cam timestamps as ref cameras.\n";
  std::vector<double> world_to_ref_vec(num_ref_cams * dense_map::NUM_RIGID_PARAMS);
  for (int cid = 0; cid < num_ref_cams; cid++)
    dense_map::rigid_transform_to_array(world_to_ref[cid],
                                        &world_to_ref_vec[dense_map::NUM_RIGID_PARAMS * cid]);

  calc_world_to_cam_using_rig(// Inputs
                              cams, world_to_ref_vec, ref_timestamps, ref_to_cam_vec,  
                              ref_to_cam_timestamp_offsets,  
                              // Output
                              world_to_cam);
}
  
// Calculate world_to_cam transforms from their representation in a
// vector, rather than using reference cameras, extrinsics and
// timestamp interpolation. Only for use with --no_rig, when
// each camera varies independently.
void calc_world_to_cam_no_rig(// Inputs
  std::vector<dense_map::cameraImage> const& cams, std::vector<double> const& world_to_cam_vec,
  // Output
  std::vector<Eigen::Affine3d>& world_to_cam) {
  if (world_to_cam_vec.size() != cams.size() * dense_map::NUM_RIGID_PARAMS)
    LOG(FATAL) << "Incorrect size for world_to_cam_vec.\n";

  for (size_t cid = 0; cid < cams.size(); cid++)
    dense_map::array_to_rigid_transform(world_to_cam[cid],  // output
                                        &world_to_cam_vec[dense_map::NUM_RIGID_PARAMS * cid]);
}

// Use one of the two implementations above. Care is needed as when there are no extrinsics,
// each camera is on its own, so the input is in world_to_cam_vec and not in world_to_ref_vec
void calc_world_to_cam_rig_or_not(// Inputs
  bool no_rig, std::vector<dense_map::cameraImage> const& cams,
  std::vector<double> const& world_to_ref_vec, std::vector<double> const& ref_timestamps,
  std::vector<double> const& ref_to_cam_vec, std::vector<double> const& world_to_cam_vec,
  std::vector<double> const& ref_to_cam_timestamp_offsets,
  // Output
  std::vector<Eigen::Affine3d>& world_to_cam) {
  if (!no_rig)
    calc_world_to_cam_using_rig(// Inputs
                                 cams, world_to_ref_vec, ref_timestamps, ref_to_cam_vec,
                                 ref_to_cam_timestamp_offsets,
                                 // Output
                                 world_to_cam);
  else
    calc_world_to_cam_no_rig(// Inputs
      cams, world_to_cam_vec,
      // Output
      world_to_cam);

  return;
}

void parameterValidation() {
    
  if (FLAGS_robust_threshold <= 0.0)
    LOG(FATAL) << "The robust threshold must be positive.\n";

  if (FLAGS_bracket_len <= 0.0) LOG(FATAL) << "Bracket length must be positive.";

  if (FLAGS_num_overlaps < 1 && (FLAGS_nvm == "" || FLAGS_no_nvm_matches))
    LOG(FATAL) << "No nvm file was specified or it is not desired to read its matches. "
      "Then must set a positive --num_overlaps to be able to find new interest point matches.";

  if (FLAGS_timestamp_offsets_max_change < 0)
    LOG(FATAL) << "The timestamp offsets must be non-negative.";

  if (FLAGS_refiner_min_angle <= 0.0)
    LOG(FATAL) << "The min triangulation angle must be positive.\n";

  if (FLAGS_depth_tri_weight < 0.0)
    LOG(FATAL) << "The depth weight must non-negative.\n";

  if (FLAGS_mesh_tri_weight < 0.0)
    LOG(FATAL) << "The mesh weight must non-negative.\n";

  if (FLAGS_depth_mesh_weight < 0.0)
    LOG(FATAL) << "The depth mesh weight must non-negative.\n";

  if (FLAGS_tri_weight < 0.0)
    LOG(FATAL) << "The triangulation weight must non-negative.\n";

  if (FLAGS_tri_weight > 0.0 && FLAGS_tri_robust_threshold <= 0.0)
    LOG(FATAL) << "The triangulation robust threshold must be positive.\n";

  if (FLAGS_registration && (FLAGS_xyz_file.empty() || FLAGS_hugin_file.empty()))
    LOG(FATAL) << "In order to register the map, the hugin and xyz file must be specified.";

  if (FLAGS_float_scale && FLAGS_affine_depth_to_image)
    LOG(FATAL) << "The options --float_scale and --affine_depth_to_image should not be used "
               << "together. If the latter is used, the scale is always floated.\n";

  if (FLAGS_no_rig && FLAGS_float_timestamp_offsets)
      LOG(FATAL) << "Cannot float timestamps with option --no_rig.\n";

  if (FLAGS_out_dir == "")
    LOG(FATAL) << "The output directory was not specified.\n";

  if (FLAGS_use_initial_rig_transforms && FLAGS_no_rig)
    LOG(FATAL) << "Cannot use initial rig transforms if not modeling the rig.\n";

  if (FLAGS_out_texture_dir != "" && FLAGS_mesh == "")
      LOG(FATAL) << "Cannot project camera images onto a mesh if a mesh was not provided.\n";

  if (FLAGS_rig_config == "")
    LOG(FATAL) << "Must specify the initial rig configuration via --rig_config.\n";


    if (FLAGS_camera_poses != "" && FLAGS_nvm != "")
      LOG(FATAL) << "Cannot specify both --nvm and --camera_poses.\n";

  if (FLAGS_camera_poses == "" && FLAGS_nvm == "")
    LOG(FATAL) << "Must specify the cameras via --nvm or --camera_poses.\n";

  return;
}

void set_up_block_sizes(int num_depth_params,
                        std::vector<int> & bracketed_cam_block_sizes,
                        std::vector<int> & bracketed_depth_block_sizes,
                        std::vector<int> & bracketed_depth_mesh_block_sizes,
                        std::vector<int> & xyz_block_sizes) {
  // Wipe the outputs
  bracketed_cam_block_sizes.clear();
  bracketed_depth_block_sizes.clear();
  bracketed_depth_mesh_block_sizes.clear();
  xyz_block_sizes.clear();

  int num_focal_lengths = 1;      // The x and y focal length are assumed to be the same
  int num_distortion_params = 1;  // will be overwritten later

  // Set up the variable blocks to optimize for BracketedCamError

  bracketed_cam_block_sizes.push_back(dense_map::NUM_RIGID_PARAMS);
  bracketed_cam_block_sizes.push_back(dense_map::NUM_RIGID_PARAMS);
  bracketed_cam_block_sizes.push_back(dense_map::NUM_RIGID_PARAMS);
  bracketed_cam_block_sizes.push_back(dense_map::NUM_XYZ_PARAMS);
  bracketed_cam_block_sizes.push_back(dense_map::NUM_SCALAR_PARAMS);
  bracketed_cam_block_sizes.push_back(num_focal_lengths);
  bracketed_cam_block_sizes.push_back(dense_map::NUM_OPT_CTR_PARAMS);
  bracketed_cam_block_sizes.push_back(num_distortion_params);

  // Set up variable blocks to optimize for BracketedDepthError
  bracketed_depth_block_sizes.push_back(dense_map::NUM_RIGID_PARAMS);
  bracketed_depth_block_sizes.push_back(dense_map::NUM_RIGID_PARAMS);
  bracketed_depth_block_sizes.push_back(dense_map::NUM_RIGID_PARAMS);
  bracketed_depth_block_sizes.push_back(num_depth_params);
  bracketed_depth_block_sizes.push_back(dense_map::NUM_SCALAR_PARAMS);
  bracketed_depth_block_sizes.push_back(dense_map::NUM_XYZ_PARAMS);
  bracketed_depth_block_sizes.push_back(dense_map::NUM_SCALAR_PARAMS);

  // Set up the variable blocks to optimize for BracketedDepthMeshError
  bracketed_depth_mesh_block_sizes.push_back(dense_map::NUM_RIGID_PARAMS);
  bracketed_depth_mesh_block_sizes.push_back(dense_map::NUM_RIGID_PARAMS);
  bracketed_depth_mesh_block_sizes.push_back(dense_map::NUM_RIGID_PARAMS);
  bracketed_depth_mesh_block_sizes.push_back(num_depth_params);
  bracketed_depth_mesh_block_sizes.push_back(dense_map::NUM_SCALAR_PARAMS);
  bracketed_depth_mesh_block_sizes.push_back(dense_map::NUM_SCALAR_PARAMS);

  // Set up the variable blocks to optimize for the mesh xyz
  xyz_block_sizes.push_back(dense_map::NUM_XYZ_PARAMS);
}

void flagOutlierByExclusionDist(// Inputs
                                int ref_cam_type,
                                std::vector<camera::CameraParameters> const& cam_params,
                                std::vector<dense_map::cameraImage> const& cams,
                                std::vector<std::map<int, int>> const& pid_to_cid_fid,
                                std::vector<std::vector<std::pair<float, float>>>
                                const& keypoint_vec,
                                // Outputs
  std::vector<std::map<int, std::map<int, int>>>& pid_cid_fid_inlier) {

  // Initialize the output
  pid_cid_fid_inlier.resize(pid_to_cid_fid.size());

  // Iterate though interest point matches
  for (size_t pid = 0; pid < pid_to_cid_fid.size(); pid++) {
    for (auto cid_fid = pid_to_cid_fid[pid].begin(); cid_fid != pid_to_cid_fid[pid].end();
         cid_fid++) {
      int cid = cid_fid->first;
      int fid = cid_fid->second;
      int cam_type = cams[cid].camera_type;

      // Initially there are inliers only
      pid_cid_fid_inlier[pid][cid][fid] = 1;

      // Flag as outliers pixels at the image boundary.
      Eigen::Vector2d dist_pix(keypoint_vec[cid][fid].first, keypoint_vec[cid][fid].second);
      Eigen::Vector2i dist_size = cam_params[cam_type].GetDistortedSize();
      Eigen::Vector2i dist_crop_size = cam_params[cam_type].GetDistortedCropSize();
      // Note that if dist_crop_size equals dist_size, which is image
      // size, no outliers are flagged
      if (std::abs(dist_pix[0] - dist_size[0] / 2.0) > dist_crop_size[0] / 2.0  ||
          std::abs(dist_pix[1] - dist_size[1] / 2.0) > dist_crop_size[1] / 2.0) 
        dense_map::setMapValue(pid_cid_fid_inlier, pid, cid, fid, 0);
    }
  }
  return;
}

// Flag outliers by triangulation angle and reprojection error.  It is
// assumed that the cameras in world_to_cam are up-to-date given the
// current state of optimization, and that the residuals (including
// the reprojection errors) have also been updated beforehand.
void flagOutliersByTriAngleAndReprojErr(  // Inputs
  double refiner_min_angle, double max_reprojection_error,
  std::vector<std::map<int, int>> const& pid_to_cid_fid,
  std::vector<std::vector<std::pair<float, float>>> const& keypoint_vec,
  std::vector<Eigen::Affine3d> const& world_to_cam, std::vector<Eigen::Vector3d> const& xyz_vec,
  std::vector<std::map<int, std::map<int, int>>> const& pid_cid_fid_to_residual_index,
  std::vector<double> const& residuals,
  // Outputs
  std::vector<std::map<int, std::map<int, int>>>& pid_cid_fid_inlier) {
  // Must deal with outliers by triangulation angle before
  // removing outliers by reprojection error, as the latter will
  // exclude some rays which form the given triangulated points.
  int num_outliers_by_angle = 0, num_total_features = 0;
  for (size_t pid = 0; pid < pid_to_cid_fid.size(); pid++) {
    // Find the largest angle among any two intersecting rays
    double max_rays_angle = 0.0;

    for (auto cid_fid1 = pid_to_cid_fid[pid].begin();
         cid_fid1 != pid_to_cid_fid[pid].end(); cid_fid1++) {
      int cid1 = cid_fid1->first;
      int fid1 = cid_fid1->second;

      // Deal with inliers only
      if (!dense_map::getMapValue(pid_cid_fid_inlier, pid, cid1, fid1)) continue;

      num_total_features++;

      Eigen::Vector3d cam_ctr1 = (world_to_cam[cid1].inverse()) * Eigen::Vector3d(0, 0, 0);
      Eigen::Vector3d ray1 = xyz_vec[pid] - cam_ctr1;
      ray1.normalize();

      for (auto cid_fid2 = pid_to_cid_fid[pid].begin();
           cid_fid2 != pid_to_cid_fid[pid].end(); cid_fid2++) {
        int cid2 = cid_fid2->first;
        int fid2 = cid_fid2->second;

        // Look at each cid and next cids
        if (cid2 <= cid1)
          continue;

        // Deal with inliers only
        if (!dense_map::getMapValue(pid_cid_fid_inlier, pid, cid2, fid2)) continue;

        Eigen::Vector3d cam_ctr2 = (world_to_cam[cid2].inverse()) * Eigen::Vector3d(0, 0, 0);
        Eigen::Vector3d ray2 = xyz_vec[pid] - cam_ctr2;
        ray2.normalize();

        double curr_angle = (180.0 / M_PI) * acos(ray1.dot(ray2));

        if (std::isnan(curr_angle) || std::isinf(curr_angle)) continue;

        max_rays_angle = std::max(max_rays_angle, curr_angle);
      }
    }

    if (max_rays_angle >= refiner_min_angle)
      continue;  // This is a good triangulated point, with large angle of convergence

    // Flag as outliers all the features for this cid
    for (auto cid_fid = pid_to_cid_fid[pid].begin();
         cid_fid != pid_to_cid_fid[pid].end(); cid_fid++) {
      int cid = cid_fid->first;
      int fid = cid_fid->second;

      // Deal with inliers only
      if (!dense_map::getMapValue(pid_cid_fid_inlier, pid, cid, fid)) continue;

      num_outliers_by_angle++;
      dense_map::setMapValue(pid_cid_fid_inlier, pid, cid, fid, 0);
    }
  }
  std::cout << std::setprecision(4) << "Removed " << num_outliers_by_angle
            << " outlier features with small angle of convergence, out of "
            << num_total_features << " ("
            << (100.0 * num_outliers_by_angle) / num_total_features << " %)\n";

  int num_outliers_reproj = 0;
  num_total_features = 0;  // reusing this variable
  for (size_t pid = 0; pid < pid_to_cid_fid.size(); pid++) {
    for (auto cid_fid = pid_to_cid_fid[pid].begin();
         cid_fid != pid_to_cid_fid[pid].end(); cid_fid++) {
      int cid = cid_fid->first;
      int fid = cid_fid->second;

      // Deal with inliers only
      if (!dense_map::getMapValue(pid_cid_fid_inlier, pid, cid, fid)) continue;

      num_total_features++;

      // Find the pixel residuals
      size_t residual_index = dense_map::getMapValue(pid_cid_fid_to_residual_index, pid, cid, fid);
      if (residuals.size() <= residual_index + 1) LOG(FATAL) << "Too few residuals.\n";

      double res_x = residuals[residual_index + 0];
      double res_y = residuals[residual_index + 1];
      // NaN values will never be inliers if the comparison is set as below
      bool is_good = (Eigen::Vector2d(res_x, res_y).norm() <= max_reprojection_error);
      if (!is_good) {
        num_outliers_reproj++;
        dense_map::setMapValue(pid_cid_fid_inlier, pid, cid, fid, 0);
      }
    }
  }

  std::cout << std::setprecision(4) << "Removed " << num_outliers_reproj
            << " outlier features using reprojection error, out of " << num_total_features
            << " (" << (100.0 * num_outliers_reproj) / num_total_features << " %)\n";

  return;
}

// Evaluate the residuals before and after optimization
void evalResiduals(  // Inputs
  std::string const& tag, std::vector<std::string> const& residual_names,
  std::vector<double> const& residual_scales,
  // Outputs
  ceres::Problem& problem, std::vector<double>& residuals) {
  double total_cost = 0.0;
  ceres::Problem::EvaluateOptions eval_options;
  eval_options.num_threads = 1;
  eval_options.apply_loss_function = false;  // want raw residuals
  problem.Evaluate(eval_options, &total_cost, &residuals, NULL, NULL);

  // Sanity checks, after the residuals are created
  if (residuals.size() != residual_names.size())
    LOG(FATAL) << "There must be as many residual names as residual values.";
  if (residuals.size() != residual_scales.size())
    LOG(FATAL) << "There must be as many residual values as residual scales.";

  // Compensate for the scale
  for (size_t it = 0; it < residuals.size(); it++)
    residuals[it] /= residual_scales[it];

  dense_map::calc_residuals_stats(residuals, residual_names, tag);
  return;
}

  
// TODO(oalexan1): Move to transforms
  
// Given the transforms from each camera to the world and their timestamps,
// find an initial guess for the relationship among the sensors on the rig.
// Note that strictly speaking the transforms in world_to_ref_vec are among
// those in world_to_cam, but we don't have a way of looking them up in that
// vector.
void calc_rig_using_word_to_cam(int ref_cam_type, int num_cam_types,
                         std::vector<dense_map::cameraImage> const& cams,
                         std::vector<Eigen::Affine3d>        const& world_to_ref,
                         std::vector<Eigen::Affine3d>        const& world_to_cam,
                         std::vector<double>                 const& ref_timestamps,
                         std::vector<double>                 const& ref_to_cam_timestamp_offsets,
                         // Output
                         std::vector<Eigen::Affine3d>             & ref_to_cam_trans) {

  // Sanity check
  if (cams.size() != world_to_cam.size()) 
    LOG(FATAL) << "There must be as many world to cam transforms as metadata sets for them.\n";

  int num_ref_cams = world_to_ref.size();
  if (world_to_ref.size() != ref_timestamps.size())
    LOG(FATAL) << "Must have as many ref cam timestamps as ref cameras.\n";
  std::vector<double> world_to_ref_vec(num_ref_cams * dense_map::NUM_RIGID_PARAMS);
  for (int cid = 0; cid < num_ref_cams; cid++)
    dense_map::rigid_transform_to_array(world_to_ref[cid],
                                        &world_to_ref_vec[dense_map::NUM_RIGID_PARAMS * cid]);
  
  // Wipe the output
  ref_to_cam_trans.resize(num_cam_types);

  // Calc all transforms
  std::map<int, std::vector<Eigen::MatrixXd>> transforms;
  for (size_t cam_it = 0; cam_it < cams.size(); cam_it++) {
    int beg_index = cams[cam_it].beg_ref_index;
    int end_index = cams[cam_it].end_ref_index;
    int cam_type = cams[cam_it].camera_type;
    
    if (cam_type == ref_cam_type) {
      // The identity transform, from the ref sensor to itself
      transforms[cam_type].push_back(Eigen::MatrixXd::Identity(4, 4));
    } else {
      // We have world_to_ref transform at times bracketing current time,
      // and world_to_cam at current time. Interp world_to_ref
      // at current time, then find ref_to_cam.
      Eigen::Affine3d interp_world_to_ref_aff
        = dense_map::calc_interp_world_to_ref
        (&world_to_ref_vec[dense_map::NUM_RIGID_PARAMS * beg_index],
         &world_to_ref_vec[dense_map::NUM_RIGID_PARAMS * end_index],
         ref_timestamps[beg_index], ref_timestamps[end_index],
         ref_to_cam_timestamp_offsets[cam_type],
         cams[cam_it].timestamp);
      
      Eigen::Affine3d ref_to_cam_aff
        = world_to_cam[cam_it] * (interp_world_to_ref_aff.inverse());
      transforms[cam_type].push_back(ref_to_cam_aff.matrix());
    }
  }
  
  // Find median, for robustness
  for (auto it = transforms.begin(); it != transforms.end(); it++) {
    int cam_type = it->first;
    auto & transforms = it->second;
    Eigen::MatrixXd median_trans = Eigen::MatrixXd::Zero(4, 4);
    for (int col = 0; col < 4; col++) {
      for (int row = 0; row < 4; row++) {
        std::vector<double> vals;
        for (size_t cam_it = 0; cam_it < transforms.size(); cam_it++)
          vals.push_back(transforms[cam_it](col, row));

        if (vals.empty()) 
          LOG(FATAL) << "No poses were found for rig sensor with id: " << cam_type << "\n";

        median_trans(col, row) = vals[vals.size()/2];
      }
    }
    ref_to_cam_trans[cam_type].matrix() = median_trans;
    ref_to_cam_trans[cam_type].linear() /= 
      pow(ref_to_cam_trans[cam_type].linear().determinant(), 1.0 / 3.0);
  }
  
  return;
}

} // end namespace dense_map

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  tbb::task_arena schedule(tbb::task_arena::automatic); // to force linking to tbb

  dense_map::parameterValidation();

  // We assume that the first cam is the ref cam (its id is 0)
  int ref_cam_type = 0;

  // Image names
  std::vector<std::string> cam_names;
  std::vector<Eigen::Affine3d> depth_to_image;
  std::vector<camera::CameraParameters> cam_params;
  std::vector<Eigen::Affine3d>          ref_to_cam_trans;
  std::vector<double>                   ref_to_cam_timestamp_offsets;

  bool use_initial_rig_transforms = FLAGS_use_initial_rig_transforms; // this may change
  dense_map::readRigConfig(FLAGS_rig_config, use_initial_rig_transforms, ref_cam_type, cam_names,
                           cam_params, ref_to_cam_trans, depth_to_image,
                           ref_to_cam_timestamp_offsets);

  int num_cam_types = cam_params.size();

  // Optionally load the mesh
  mve::TriangleMesh::Ptr mesh;
  std::shared_ptr<mve::MeshInfo> mesh_info;
  std::shared_ptr<tex::Graph> graph;
  std::shared_ptr<BVHTree> bvh_tree;
  if (FLAGS_mesh != "")
    dense_map::loadMeshBuildTree(FLAGS_mesh, mesh, mesh_info, graph, bvh_tree);

  // world_to_ref has the transforms from the ref cameras to the world,
  // while world_to_cam has the transforms from the world to all cameras,
  // including world_to_ref. Both of these are needed in certain circumstances,
  // and it is very important to always keep these in sync.
  std::vector<Eigen::Affine3d> world_to_ref, world_to_cam;

  // image_data is on purpose stored in vectors of vectors, with each
  // image_data[i] having data in increasing order of timestamps. This
  // way it is fast to find next timestamps after a given one.
  std::vector<double> ref_timestamps; // Timestamps for the ref cameras
  std::vector<std::vector<dense_map::ImageMessage>> image_data;
  std::vector<std::vector<dense_map::ImageMessage>> depth_data;
  std::vector<std::string> ref_image_files;
  dense_map::nvmData nvm;

  if (FLAGS_camera_poses != "")
    dense_map::readCameraPoses(FLAGS_camera_poses, ref_cam_type, cam_names, // in
                               nvm, ref_timestamps, world_to_ref, ref_image_files,
                               image_data, depth_data); // out
  else if (FLAGS_nvm != "") 
    dense_map::readNvm(FLAGS_nvm, ref_cam_type, cam_names, // in
                       nvm, ref_timestamps, world_to_ref, ref_image_files,
                       image_data, depth_data); // out
  
  // Keep here the images, timestamps, and bracketing information
  std::vector<dense_map::cameraImage> cams;
  //  The range of ref_to_cam_timestamp_offsets[cam_type] before
  //  getting out of the bracket.
  std::vector<double> min_timestamp_offset, max_timestamp_offset;
  // Select the images to use. If the rig is used, keep non-ref images
  // only within the bracket.
  dense_map::lookupImages(// Inputs
                          ref_cam_type, FLAGS_no_rig, FLAGS_bracket_len,
                          FLAGS_timestamp_offsets_max_change,
                          cam_names, cam_params,
                          ref_timestamps, image_data, depth_data,
                          ref_to_cam_timestamp_offsets,
                          // Outputs
                          cams, min_timestamp_offset, max_timestamp_offset);
  
  // If we have initial rig transforms, compute the transform from the
  // world to every camera based on the rig transforms and ref_to_cam
  // transforms. It assumes that world_to_ref and ref_to_cam
  // are up-to-date. Use the version of calc_world_to_cam_using_rig
  // without world_to_cam_vec, on input which was not computed yet.

  if (use_initial_rig_transforms && !FLAGS_no_rig) {
    // Using the rig transforms in ref_to_cam_vec and transforms from
    // world to each ref cam in world_to_ref, calculate world_to_cam,
    // the transforms from the world to each camera
    dense_map::calc_world_to_cam_using_rig(// Inputs
                                           cams, world_to_ref, ref_timestamps, ref_to_cam_trans,
                                           ref_to_cam_timestamp_offsets,
                                           // Output
                                           world_to_cam);
  } else {
    // Parse the transform from the world to each cam, which were known on input
    world_to_cam.resize(cams.size());
    std::vector<int> start_pos(num_cam_types, 0);  // to help advance in time
    for (size_t cam_it = 0; cam_it < cams.size(); cam_it++) {
      int cam_type = cams[cam_it].camera_type;
      for (size_t pos = start_pos[cam_type]; pos < image_data[cam_type].size(); pos++) {
        // Count on the fact that image_data[cam_type] is sorted chronologically
        if (cams[cam_it].timestamp == image_data[cam_type][pos].timestamp) {
          world_to_cam[cam_it] = image_data[cam_type][pos].world_to_cam;
          start_pos[cam_type] = pos;  // save for next time
        }
      }
    }

    // Using the transforms from the world to each camera, compute
    // the rig transforms
    dense_map::calc_rig_using_word_to_cam(ref_cam_type, num_cam_types,  
                                          cams, world_to_ref, world_to_cam,  
                                          ref_timestamps,  ref_to_cam_timestamp_offsets,  
                                          // Output
                                          ref_to_cam_trans);
  }

  // TODO(oalexan1): Not clear what to do about depth_to_image.
  if (FLAGS_registration  && (FLAGS_hugin_file != "" && FLAGS_xyz_file != "")) {
    Eigen::Affine3d registration_trans
      = dense_map::registrationTransform(FLAGS_hugin_file, FLAGS_xyz_file,  
                                         cam_params[ref_cam_type],  
                                         ref_image_files,  
                                         world_to_ref);
    // The above transformed world_to_ref. Also transform world_to_cam.
    // TODO(oalexan1): Apply transform passed in from outside
    dense_map::TransformCameras(registration_trans, world_to_cam);
    dense_map::TransformRig(registration_trans, ref_to_cam_trans); // this uses different logic
    // TODO(oalexan1): if we have triangulated points, those need transforming as well
    
    // TODO(oalexan1): In post-registration, apply transform to depth_to_image.
    // Not here though, as the current scale is from Theia's coords to world,
    // rather than within the world.
    //depth_to_image[cam_type].linear() *= map_scale;
    //depth_to_image[cam_type].translation() *= map_scale;
  }

  // Put the rig transforms in arrays, so we can optimize them
  std::vector<double> ref_to_cam_vec(num_cam_types * dense_map::NUM_RIGID_PARAMS);
  for (int cam_type = 0; cam_type < num_cam_types; cam_type++)
    dense_map::rigid_transform_to_array
      (ref_to_cam_trans[cam_type],
       &ref_to_cam_vec[dense_map::NUM_RIGID_PARAMS * cam_type]);

  // Put transforms of the reference cameras in a vector so we can optimize them.
  // TODO(oalexan1): Eliminate world_to_ref. Use only world_to_ref_vec.
  int num_ref_cams = world_to_ref.size();
  if (world_to_ref.size() != ref_timestamps.size())
    LOG(FATAL) << "Must have as many ref cam timestamps as ref cameras.\n";
  std::vector<double> world_to_ref_vec(num_ref_cams * dense_map::NUM_RIGID_PARAMS);
  for (int cid = 0; cid < num_ref_cams; cid++)
    dense_map::rigid_transform_to_array(world_to_ref[cid],
                                        &world_to_ref_vec[dense_map::NUM_RIGID_PARAMS * cid]);
  
  // Need the identity transform for when the cam is the ref cam, and
  // have to have a placeholder for the right bracketing cam which won't be used.
  Eigen::Affine3d identity = Eigen::Affine3d::Identity();
  std::vector<double> identity_vec(dense_map::NUM_RIGID_PARAMS);
  dense_map::rigid_transform_to_array(identity, &identity_vec[0]);

  // Which intrinsics from which cameras to float. Indexed by cam_type.
  std::vector<std::set<std::string>> intrinsics_to_float;
  dense_map::parse_intrinsics_to_float(FLAGS_intrinsics_to_float, cam_names,
                                       intrinsics_to_float);

  // TODO(oalexan1): When --no_rig is on, these must be empty!
  std::set<std::string> rig_transforms_to_float;
  dense_map::parse_rig_transforms_to_float(cam_names, ref_cam_type,
                                           FLAGS_rig_transforms_to_float, rig_transforms_to_float);
  
  std::set<std::string> camera_poses_to_float;
  dense_map::parse_camera_names(cam_names, 
                                FLAGS_camera_poses_to_float,
                                camera_poses_to_float);

  std::set<std::string> depth_to_image_transforms_to_float;
  dense_map::parse_camera_names(cam_names, 
                                FLAGS_depth_to_image_transforms_to_float,
                                depth_to_image_transforms_to_float);
  
  // Set up the variable blocks to optimize for BracketedDepthError
  int num_depth_params = dense_map::NUM_RIGID_PARAMS;
  if (FLAGS_affine_depth_to_image) num_depth_params = dense_map::NUM_AFFINE_PARAMS;

  // Separate the initial scale. This is convenient if
  // cam_depth_to_image is scale * rotation + translation and if
  // it is desired to keep the scale fixed. In either case, the scale
  // will be multiplied back when needed.
  std::vector<double> depth_to_image_scales;
  for (int cam_type = 0; cam_type < num_cam_types; cam_type++) {
    double depth_to_image_scale
      = pow(depth_to_image[cam_type].matrix().determinant(), 1.0 / 3.0);
    depth_to_image[cam_type].linear() /= depth_to_image_scale;
    depth_to_image_scales.push_back(depth_to_image_scale);
  }

  // Put depth_to_image arrays, so we can optimize them
  std::vector<double> depth_to_image_vec(num_cam_types * num_depth_params);
  for (int cam_type = 0; cam_type < num_cam_types; cam_type++) {
    if (FLAGS_affine_depth_to_image)
      dense_map::affine_transform_to_array
        (depth_to_image[cam_type],
         &depth_to_image_vec[num_depth_params * cam_type]);
    else
      dense_map::rigid_transform_to_array
        (depth_to_image[cam_type],
         &depth_to_image_vec[num_depth_params * cam_type]);
  }

  // Put the intrinsics in arrays
  std::vector<double> focal_lengths(num_cam_types);
  std::vector<Eigen::Vector2d> optical_centers(num_cam_types);
  std::vector<Eigen::VectorXd> distortions(num_cam_types);
  for (int it = 0; it < num_cam_types; it++) {
    focal_lengths[it] = cam_params[it].GetFocalLength();  // average the two focal lengths
    optical_centers[it] = cam_params[it].GetOpticalOffset();
    distortions[it] = cam_params[it].GetDistortion();
  }

  // TODO(oalexan1): Eliminate world_to_cam, use only world_to_cam_vec
  // If using no extrinsics, each camera will float separately, using
  // world_to_cam as initial guesses. Use world_to_cam_vec as storage
  // for the camera poses to optimize.
  std::vector<double> world_to_cam_vec;
  if (FLAGS_no_rig) {
    world_to_cam_vec.resize(cams.size() * dense_map::NUM_RIGID_PARAMS);
    for (size_t cid = 0; cid < cams.size(); cid++)
      dense_map::rigid_transform_to_array(world_to_cam[cid],
                                          &world_to_cam_vec[dense_map::NUM_RIGID_PARAMS * cid]);
  }

  // Detect and match features if the user chooses to, so if --num_overlaps > 0. Normally,
  // these are read from the nvm file only, as below.
  std::vector<std::vector<std::pair<float, float>>> keypoint_vec;
  std::vector<std::map<int, int>> pid_to_cid_fid;
  if (FLAGS_num_overlaps > 0)
    dense_map::detectMatchFeatures(// Inputs
                                   cams, cam_params, FLAGS_out_dir, FLAGS_save_matches,
                                   world_to_cam,
                                   FLAGS_num_overlaps, FLAGS_initial_max_reprojection_error,
                                   FLAGS_num_match_threads,
                                   FLAGS_verbose,
                                   // Outputs
                                   keypoint_vec, pid_to_cid_fid);

  // Append the interest point matches from the nvm file
  if (!FLAGS_no_nvm_matches)
    dense_map::appendMatchesFromNvm(// Inputs
                                    cam_params, cams, nvm,  
                                    // Outputs (these get appended to)
                                    pid_to_cid_fid, keypoint_vec);

  if (pid_to_cid_fid.empty())
    LOG(FATAL) << "No interest points were found.\n";
  
  // Set up the block sizes
  std::vector<int> bracketed_cam_block_sizes;
  std::vector<int> bracketed_depth_block_sizes;
  std::vector<int> bracketed_depth_mesh_block_sizes;
  std::vector<int> xyz_block_sizes;
  dense_map::set_up_block_sizes(num_depth_params,
                                bracketed_cam_block_sizes, bracketed_depth_block_sizes,
                                bracketed_depth_mesh_block_sizes, xyz_block_sizes);

  // For a given fid = pid_to_cid_fid[pid][cid], the value
  // pid_cid_fid_inlier[pid][cid][fid] will be non-zero only if this
  // pixel is an inlier. Originally all pixels are inliers. Once an
  // inlier becomes an outlier, it never becomes an inlier again.
  std::vector<std::map<int, std::map<int, int>>> pid_cid_fid_inlier;
  std::vector<Eigen::Vector3d> xyz_vec; // triangulated points go here
  
  // TODO(oalexan1): Must initialize all points as inliers outside this function,
  // as now this function resets those.
  dense_map::flagOutlierByExclusionDist(// Inputs
                                        ref_cam_type, cam_params, cams, pid_to_cid_fid,
                                        keypoint_vec,
                                        // Outputs
                                        pid_cid_fid_inlier);

  // Structures needed to intersect rays with the mesh
  std::vector<std::map<int, std::map<int, Eigen::Vector3d>>> pid_cid_fid_mesh_xyz;
  std::vector<Eigen::Vector3d> pid_mesh_xyz;
  Eigen::Vector3d bad_xyz(1.0e+100, 1.0e+100, 1.0e+100);  // use this to flag invalid xyz

  // TODO(oalexan1): All the logic for one pass should be its own function,
  // as the block below is too big.
  for (int pass = 0; pass < FLAGS_calibrator_num_passes; pass++) {
    std::cout << "\nOptimization pass "
              << pass + 1 << " / " << FLAGS_calibrator_num_passes << "\n";

    // The transforms from the world to all cameras must be updated
    // given the current state of optimization
    // TODO(oalexan1): The call below is likely not necessary since this function
    // is already called earlier, and also whenever a pass finishes, see below.
    dense_map::calc_world_to_cam_rig_or_not(  // Inputs
      FLAGS_no_rig, cams, world_to_ref_vec, ref_timestamps, ref_to_cam_vec, world_to_cam_vec,
      ref_to_cam_timestamp_offsets,
      // Output
      world_to_cam);

    dense_map::multiViewTriangulation(// Inputs
                                      cam_params, cams, world_to_cam, pid_to_cid_fid, keypoint_vec,
                                      // Outputs
                                      pid_cid_fid_inlier, xyz_vec);

    // This is a copy which won't change
    std::vector<Eigen::Vector3d> xyz_vec_orig;
    if (FLAGS_tri_weight > 0.0) {
      // Better copy manually to ensure no shallow copy
      xyz_vec_orig.resize(xyz_vec.size());
      for (size_t pt_it = 0; pt_it < xyz_vec.size(); pt_it++) {
        for (int coord_it = 0; coord_it < 3; coord_it++) {
          xyz_vec_orig[pt_it][coord_it] = xyz_vec[pt_it][coord_it];
        }
      } 
    }
    
    // Compute where each ray intersects the mesh
    if (FLAGS_mesh != "")
      dense_map::meshTriangulations(  // Inputs
                                    cam_params, cams, world_to_cam, pid_to_cid_fid,
        pid_cid_fid_inlier, keypoint_vec, bad_xyz, FLAGS_min_ray_dist, FLAGS_max_ray_dist, mesh,
        bvh_tree,
        // Outputs
        pid_cid_fid_mesh_xyz, pid_mesh_xyz);

    // For a given fid = pid_to_cid_fid[pid][cid],
    // the value pid_cid_fid_to_residual_index[pid][cid][fid] will be the index in the array
    // of residuals (look only at pixel residuals). This structure is populated only for
    // inliers, so its total number of elements changes at each pass.
    std::vector<std::map<int, std::map<int, int>>> pid_cid_fid_to_residual_index;
    pid_cid_fid_to_residual_index.resize(pid_to_cid_fid.size());

    // If distortion can be floated, and the RPC distortion model is
    // used, must forbid undistortion until its updated value is
    // computed later.
    // TODO(oalexan1): Need to have an actual flag for when we use RPC.
    for (int cam_type = 0; cam_type < num_cam_types; cam_type++) {
      if (intrinsics_to_float[cam_type].find("distortion")
          != intrinsics_to_float[cam_type].end() && distortions[cam_type].size() > 5)
        cam_params[cam_type].m_rpc.set_can_undistort(false);
    }
    
    // For when we don't have distortion but must get a pointer to distortion for the interface
    double distortion_placeholder = 0.0;
    
    // Form the problem
    ceres::Problem problem;
    std::vector<std::string> residual_names;
    std::vector<double> residual_scales;
    for (size_t pid = 0; pid < pid_to_cid_fid.size(); pid++) {
      for (auto cid_fid = pid_to_cid_fid[pid].begin();
           cid_fid != pid_to_cid_fid[pid].end(); cid_fid++) {
        int cid = cid_fid->first;
        int fid = cid_fid->second;

        // Deal with inliers only
        if (!dense_map::getMapValue(pid_cid_fid_inlier, pid, cid, fid))
          continue;

        int cam_type = cams[cid].camera_type;
        double beg_ref_timestamp = -1.0, end_ref_timestamp = -1.0, cam_timestamp = -1.0;

        // Pointers to bracketing cameras and ref to cam transform. Their precise
        // definition is spelled out below.
        double *beg_cam_ptr = NULL, *end_cam_ptr = NULL, *ref_to_cam_ptr = NULL;

        if (!FLAGS_no_rig) {
          // Model the rig, use timestamps
          int beg_ref_index = cams[cid].beg_ref_index;
          int end_ref_index = cams[cid].end_ref_index;

          // Left bracketing ref cam for a given cam. For a ref cam, this is itself.
          beg_cam_ptr = &world_to_ref_vec[dense_map::NUM_RIGID_PARAMS * beg_ref_index];

          // Right bracketing camera. When the cam is the ref type,
          // this is nominal and not used. Also when the current cam
          // is the last one and has exactly same timestamp as the ref cam
          if (cam_type == ref_cam_type || beg_ref_index == end_ref_index)
            end_cam_ptr = &identity_vec[0];
          else
            end_cam_ptr = &world_to_ref_vec[dense_map::NUM_RIGID_PARAMS * end_ref_index];

          // The beg and end timestamps will be the same only for the
          // ref cam
          beg_ref_timestamp = ref_timestamps[beg_ref_index];
          end_ref_timestamp = ref_timestamps[end_ref_index];
          cam_timestamp = cams[cid].timestamp;  // uses current camera's clock

        } else {
          // No rig. Then, beg_cam_ptr is just current camera,
          // not the ref bracket, end_cam_ptr is the identity and
          // fixed. The beg and end timestamps are declared to be
          // same, which will be used in calc_world_to_cam_trans() to
          // ignore the rig transform and end_cam_ptr.
          cam_timestamp     = cams[cid].timestamp;
          beg_ref_timestamp = cam_timestamp;
          end_ref_timestamp = cam_timestamp;

          // Note how we use world_to_cam_vec and not world_to_ref_vec
          beg_cam_ptr  = &world_to_cam_vec[dense_map::NUM_RIGID_PARAMS * cid];
          end_cam_ptr = &identity_vec[0];
        }

        // Transform from reference camera to given camera. Won't be used when
        // FLAGS_no_rig is true or when the cam is of ref type.
        ref_to_cam_ptr = &ref_to_cam_vec[dense_map::NUM_RIGID_PARAMS * cam_type];

        Eigen::Vector2d dist_ip(keypoint_vec[cid][fid].first, keypoint_vec[cid][fid].second);

        ceres::CostFunction* bracketed_cost_function =
          dense_map::BracketedCamError::Create(dist_ip, beg_ref_timestamp, end_ref_timestamp,
                                               cam_timestamp, bracketed_cam_block_sizes,
                                               cam_params[cam_type]);
        ceres::LossFunction* bracketed_loss_function
          = dense_map::GetLossFunction("cauchy", FLAGS_robust_threshold);

        // Remember the index of the residuals about to create
        pid_cid_fid_to_residual_index[pid][cid][fid] = residual_names.size();

        // Handle the case of no distortion
        double * distortion_ptr = NULL;
        if (distortions[cam_type].size() > 0) 
          distortion_ptr = &distortions[cam_type][0];
        else
          distortion_ptr = &distortion_placeholder;
        
        residual_names.push_back(cam_names[cam_type] + "_pix_x");
        residual_names.push_back(cam_names[cam_type] + "_pix_y");
        residual_scales.push_back(1.0);
        residual_scales.push_back(1.0);
        problem.AddResidualBlock
          (bracketed_cost_function, bracketed_loss_function,
           beg_cam_ptr, end_cam_ptr, ref_to_cam_ptr, &xyz_vec[pid][0],
           &ref_to_cam_timestamp_offsets[cam_type],
           &focal_lengths[cam_type], &optical_centers[cam_type][0], distortion_ptr);

        // See which intrinsics to float
        if (intrinsics_to_float[cam_type].find("focal_length") ==
            intrinsics_to_float[cam_type].end())
          problem.SetParameterBlockConstant(&focal_lengths[cam_type]);
        if (intrinsics_to_float[cam_type].find("optical_center") ==
            intrinsics_to_float[cam_type].end())
          problem.SetParameterBlockConstant(&optical_centers[cam_type][0]);
        if (intrinsics_to_float[cam_type].find("distortion")
            == intrinsics_to_float[cam_type].end() || distortions[cam_type].size() == 0)
          problem.SetParameterBlockConstant(distortion_ptr);

        // When the camera is the ref type, the right bracketing
        // camera is just a placeholder which is not used, hence
        // should not be optimized. Same for the ref_to_cam_vec and
        // ref_to_cam_timestamp_offsets, etc., as can be seen further
        // down.
        if (!FLAGS_no_rig) {
          // See if to float the ref cameras
          if (camera_poses_to_float.find(cam_names[ref_cam_type]) == camera_poses_to_float.end())
            problem.SetParameterBlockConstant(beg_cam_ptr);
        } else {
          // There is no rig. Then beg_cam_ptr refers to camera
          // for cams[cid], and not to its ref bracketing cam.
          // See if the user wants it floated.
          if (camera_poses_to_float.find(cam_names[cam_type]) == camera_poses_to_float.end()) {
            problem.SetParameterBlockConstant(beg_cam_ptr);
          }
        }

        // The end cam floats only if told to, and if it brackets
        // a given non-ref cam.
        if (camera_poses_to_float.find(cam_names[ref_cam_type]) == camera_poses_to_float.end() ||
            cam_type == ref_cam_type || FLAGS_no_rig) {
          problem.SetParameterBlockConstant(end_cam_ptr);
        }
        
        if (!FLAGS_float_timestamp_offsets || cam_type == ref_cam_type || FLAGS_no_rig) {
          // Either we don't float timestamp offsets at all, or the cam is the ref type,
          // or with no extrinsics, when it can't float anyway.
          problem.SetParameterBlockConstant(&ref_to_cam_timestamp_offsets[cam_type]);
        } else {
          problem.SetParameterLowerBound(&ref_to_cam_timestamp_offsets[cam_type], 0,
                                         min_timestamp_offset[cam_type]);
          problem.SetParameterUpperBound(&ref_to_cam_timestamp_offsets[cam_type], 0,
                                         max_timestamp_offset[cam_type]);
        }
        // ref_to_cam is kept fixed at the identity if the cam is the ref type or
        // no rig
        if (rig_transforms_to_float.find(cam_names[cam_type]) == rig_transforms_to_float.end() ||
            cam_type == ref_cam_type || FLAGS_no_rig) {
          problem.SetParameterBlockConstant(ref_to_cam_ptr);
        }

        Eigen::Vector3d depth_xyz(0, 0, 0);
        bool have_depth_tri_constraint
          = (FLAGS_depth_tri_weight > 0 &&
             dense_map::depthValue(cams[cid].depth_cloud, dist_ip, depth_xyz));

        if (have_depth_tri_constraint) {
          // Ensure that the depth points agree with triangulated points
          ceres::CostFunction* bracketed_depth_cost_function
            = dense_map::BracketedDepthError::Create(FLAGS_depth_tri_weight, depth_xyz,
                                                     beg_ref_timestamp, end_ref_timestamp,
                                                     cam_timestamp, bracketed_depth_block_sizes);

          ceres::LossFunction* bracketed_depth_loss_function
            = dense_map::GetLossFunction("cauchy", FLAGS_robust_threshold);

          residual_names.push_back("depth_tri_x_m");
          residual_names.push_back("depth_tri_y_m");
          residual_names.push_back("depth_tri_z_m");
          residual_scales.push_back(FLAGS_depth_tri_weight);
          residual_scales.push_back(FLAGS_depth_tri_weight);
          residual_scales.push_back(FLAGS_depth_tri_weight);
          problem.AddResidualBlock
            (bracketed_depth_cost_function, bracketed_depth_loss_function,
             beg_cam_ptr, end_cam_ptr, ref_to_cam_ptr,
             &depth_to_image_vec[num_depth_params * cam_type],
             &depth_to_image_scales[cam_type],
             &xyz_vec[pid][0],
             &ref_to_cam_timestamp_offsets[cam_type]);

          // Note that above we already considered fixing some params.
          // We won't repeat that code here.
          // If we model an affine depth to image, fix its scale here,
          // it will change anyway as part of depth_to_image_vec.
          if (!FLAGS_float_scale || FLAGS_affine_depth_to_image) {
            problem.SetParameterBlockConstant(&depth_to_image_scales[cam_type]);
          }

          if (depth_to_image_transforms_to_float.find(cam_names[cam_type])
              == depth_to_image_transforms_to_float.end())
            problem.SetParameterBlockConstant(&depth_to_image_vec[num_depth_params * cam_type]);
        }

        // Add the depth to mesh constraint
        bool have_depth_mesh_constraint = false;
        depth_xyz = Eigen::Vector3d(0, 0, 0);
        Eigen::Vector3d mesh_xyz(0, 0, 0);
        if (FLAGS_mesh != "") {
          mesh_xyz = dense_map::getMapValue(pid_cid_fid_mesh_xyz, pid, cid, fid);
          have_depth_mesh_constraint
            = (FLAGS_depth_mesh_weight > 0 && mesh_xyz != bad_xyz &&
               dense_map::depthValue(cams[cid].depth_cloud, dist_ip, depth_xyz));
        }

        if (have_depth_mesh_constraint) {
          // Try to make each mesh intersection agree with corresponding depth measurement,
          // if it exists
          ceres::CostFunction* bracketed_depth_mesh_cost_function
            = dense_map::BracketedDepthMeshError::Create
            (FLAGS_depth_mesh_weight, depth_xyz, mesh_xyz, beg_ref_timestamp,
             end_ref_timestamp, cam_timestamp, bracketed_depth_mesh_block_sizes);

          ceres::LossFunction* bracketed_depth_mesh_loss_function
            = dense_map::GetLossFunction("cauchy", FLAGS_robust_threshold);

          residual_names.push_back("depth_mesh_x_m");
          residual_names.push_back("depth_mesh_y_m");
          residual_names.push_back("depth_mesh_z_m");
          residual_scales.push_back(FLAGS_depth_mesh_weight);
          residual_scales.push_back(FLAGS_depth_mesh_weight);
          residual_scales.push_back(FLAGS_depth_mesh_weight);
          problem.AddResidualBlock
            (bracketed_depth_mesh_cost_function, bracketed_depth_mesh_loss_function,
             beg_cam_ptr, end_cam_ptr, ref_to_cam_ptr,
             &depth_to_image_vec[num_depth_params * cam_type],
             &depth_to_image_scales[cam_type],
             &ref_to_cam_timestamp_offsets[cam_type]);

          // Note that above we already fixed some of these variables.
          // Repeat the fixing of depth variables, however, as the previous block
          // may not take place.
          if (!FLAGS_float_scale || FLAGS_affine_depth_to_image)
            problem.SetParameterBlockConstant(&depth_to_image_scales[cam_type]);

          if (depth_to_image_transforms_to_float.find(cam_names[cam_type])
              == depth_to_image_transforms_to_float.end())
            problem.SetParameterBlockConstant(&depth_to_image_vec[num_depth_params * cam_type]);
        }
      }  // end iterating over all cid for given pid

      // The constraints below will be for each triangulated point. Skip such a point
      // if all rays converging to it come from outliers.
      bool isTriInlier = false;
      for (auto cid_fid = pid_to_cid_fid[pid].begin();
           cid_fid != pid_to_cid_fid[pid].end(); cid_fid++) {
        int cid = cid_fid->first;
        int fid = cid_fid->second;
        
        if (dense_map::getMapValue(pid_cid_fid_inlier, pid, cid, fid)) {
          isTriInlier = true;
          break; // found it to be an inlier, no need to do further checking
        }
      }

      // Add mesh-to-triangulated point constraint
      bool have_mesh_tri_constraint = false;
      Eigen::Vector3d avg_mesh_xyz(0, 0, 0);
      if (FLAGS_mesh != "" && isTriInlier) {
        avg_mesh_xyz = pid_mesh_xyz.at(pid);
        if (FLAGS_mesh_tri_weight > 0 && avg_mesh_xyz != bad_xyz)
          have_mesh_tri_constraint = true;
      }
      if (have_mesh_tri_constraint) {
        // Try to make the triangulated point agree with the mesh intersection

        ceres::CostFunction* mesh_cost_function =
          dense_map::XYZError::Create(avg_mesh_xyz, xyz_block_sizes, FLAGS_mesh_tri_weight);

        ceres::LossFunction* mesh_loss_function =
          dense_map::GetLossFunction("cauchy", FLAGS_robust_threshold);

        problem.AddResidualBlock(mesh_cost_function, mesh_loss_function,
                                 &xyz_vec[pid][0]);

        residual_names.push_back("mesh_tri_x_m");
        residual_names.push_back("mesh_tri_y_m");
        residual_names.push_back("mesh_tri_z_m");
        residual_scales.push_back(FLAGS_mesh_tri_weight);
        residual_scales.push_back(FLAGS_mesh_tri_weight);
        residual_scales.push_back(FLAGS_mesh_tri_weight);
      }

      // Add the constraint that the triangulated point does not go too far
      if (FLAGS_tri_weight > 0.0 && isTriInlier) {
        // Try to make the triangulated points (and hence cameras) not move too far
        ceres::CostFunction* tri_cost_function =
          dense_map::XYZError::Create(xyz_vec_orig[pid], xyz_block_sizes, FLAGS_tri_weight);
        ceres::LossFunction* tri_loss_function =
          dense_map::GetLossFunction("cauchy", FLAGS_tri_robust_threshold);
        problem.AddResidualBlock(tri_cost_function, tri_loss_function,
                                 &xyz_vec[pid][0]);

        residual_names.push_back("tri_x_m");
        residual_names.push_back("tri_y_m");
        residual_names.push_back("tri_z_m");
        residual_scales.push_back(FLAGS_tri_weight);
        residual_scales.push_back(FLAGS_tri_weight);
        residual_scales.push_back(FLAGS_tri_weight);
      }
      
    }  // end iterating over pid

    // Evaluate the residuals before optimization
    std::vector<double> residuals;
    dense_map::evalResiduals("before opt", residual_names, residual_scales, problem, residuals);

    // Solve the problem
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.num_threads = FLAGS_num_opt_threads;  // The result is more predictable with one thread
    options.max_num_iterations = FLAGS_num_iterations;
    options.minimizer_progress_to_stdout = true;
    options.gradient_tolerance = 1e-16;
    options.function_tolerance = 1e-16;
    options.parameter_tolerance = FLAGS_parameter_tolerance;
    ceres::Solve(options, &problem, &summary);

    // The optimization is done. Right away copy the optimized states
    // to where they belong to keep all data in sync.

    if (!FLAGS_no_rig) {
      // Copy back the reference transforms
      for (int cid = 0; cid < num_ref_cams; cid++)
        dense_map::array_to_rigid_transform
          (world_to_ref[cid], &world_to_ref_vec[dense_map::NUM_RIGID_PARAMS * cid]);
    } else {
      // Each camera floats individually. Update world_to_cam from optimized world_to_cam_vec.
      for (size_t cid = 0; cid < cams.size(); cid++) {
        dense_map::array_to_rigid_transform
          (world_to_cam[cid], &world_to_cam_vec[dense_map::NUM_RIGID_PARAMS * cid]);
      }
    }

    // Copy back the optimized intrinsics
    for (int cam_type = 0; cam_type < num_cam_types; cam_type++) {
      cam_params[cam_type].SetFocalLength(Eigen::Vector2d(focal_lengths[cam_type],
                                                          focal_lengths[cam_type]));
      cam_params[cam_type].SetOpticalOffset(optical_centers[cam_type]);
      cam_params[cam_type].SetDistortion(distortions[cam_type]);

      // This is needed for RPC, as that one has undistortion coeffs
      // which must be synced up with new distortion coeffs.
      if (intrinsics_to_float[cam_type].find("distortion")
          != intrinsics_to_float[cam_type].end() && distortions[cam_type].size() > 5)
        cam_params[cam_type].updateRpcUndistortion(FLAGS_num_opt_threads);
    }

    // Copy back the optimized extrinsics, whether it was optimized or fixed
    for (int cam_type = 0; cam_type < num_cam_types; cam_type++)
      dense_map::array_to_rigid_transform
        (ref_to_cam_trans[cam_type],
         &ref_to_cam_vec[dense_map::NUM_RIGID_PARAMS * cam_type]);

    // Copy back the depth to image transforms without scales
    for (int cam_type = 0; cam_type < num_cam_types; cam_type++) {
      if (FLAGS_affine_depth_to_image)
        dense_map::array_to_affine_transform
          (depth_to_image[cam_type],
           &depth_to_image_vec[num_depth_params * cam_type]);
      else
        dense_map::array_to_rigid_transform(
          depth_to_image[cam_type],
          &depth_to_image_vec[num_depth_params * cam_type]);
    }

    // Evaluate the residuals after optimization
    dense_map::evalResiduals("after opt", residual_names, residual_scales, problem, residuals);

    // Must have up-to-date world_to_cam and residuals to flag the outliers
    dense_map::calc_world_to_cam_rig_or_not(  // Inputs
      FLAGS_no_rig, cams, world_to_ref_vec, ref_timestamps, ref_to_cam_vec, world_to_cam_vec,
      ref_to_cam_timestamp_offsets,
      // Output
      world_to_cam);

    // Flag outliers after this pass
    dense_map::flagOutliersByTriAngleAndReprojErr(  // Inputs
        FLAGS_refiner_min_angle, FLAGS_max_reprojection_error, pid_to_cid_fid, keypoint_vec,
        world_to_cam, xyz_vec, pid_cid_fid_to_residual_index, residuals,
        // Outputs
        pid_cid_fid_inlier);
  }  // End optimization passes

  // Put back the scale in depth_to_image
  for (int cam_type = 0; cam_type < num_cam_types; cam_type++)
    depth_to_image[cam_type].linear() *= depth_to_image_scales[cam_type];

  if (FLAGS_save_matches)
    dense_map::saveInlinerMatchPairs(cams, FLAGS_num_overlaps, pid_to_cid_fid,
                                     keypoint_vec, pid_cid_fid_inlier, FLAGS_out_dir);


  // Update the transforms from the world to every camera
  dense_map::calc_world_to_cam_rig_or_not(  // Inputs
    FLAGS_no_rig, cams, world_to_ref_vec, ref_timestamps, ref_to_cam_vec, world_to_cam_vec,
    ref_to_cam_timestamp_offsets,
    // Output
    world_to_cam);

  // TODO(oalexan1): Why the call below works without dense_map:: prepended to it?
  // TODO(oalexan1): This call to calc_world_to_cam_rig_or_not is likely not
  // necessary since world_to_cam has been updated by now.
  if (FLAGS_out_texture_dir != "")
    dense_map::meshProjectCameras(cam_names, cam_params, cams, world_to_cam, mesh, bvh_tree,
                                  FLAGS_out_texture_dir);

  dense_map::saveCameraPoses(FLAGS_out_dir, cams, world_to_cam);
  
  bool model_rig = (!FLAGS_no_rig);
  dense_map::writeRigConfig(FLAGS_out_dir, model_rig, ref_cam_type, cam_names,
                            cam_params, ref_to_cam_trans, depth_to_image,
                            ref_to_cam_timestamp_offsets);

  if (FLAGS_save_nvm) {
    std::string nvm_file = FLAGS_out_dir + "/cameras.nvm";
    dense_map::writeNvm(nvm_file, cam_params, cams, world_to_cam, keypoint_vec,
                        pid_to_cid_fid, pid_cid_fid_inlier, xyz_vec);
  }
  
  if (FLAGS_export_to_voxblox)
    dense_map::exportToVoxblox(cam_names, cams, depth_to_image, world_to_cam, FLAGS_out_dir);

  if (FLAGS_save_transformed_depth_clouds)
    dense_map::saveTransformedDepthClouds(cam_names, cams, depth_to_image,
                                          world_to_cam, FLAGS_out_dir);

  return 0;
} // NOLINT // TODO(oalexan1): Remove this, after making the code more modular
