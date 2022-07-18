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
// to bracket other cameras in time.

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

DEFINE_int32(num_overlaps, 10, "How many images (of all camera types) close and forward in "
             "time to match to given image.");

DEFINE_double(max_image_to_depth_timestamp_diff, 0.2,
              "Use a depth cloud only if it is within this distance in time "
              "from the nearest image with the same camera. Measured in seconds.");

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
              "Specify the cameras of which sensor types can have their poses floated. Note that allowing the cameras for all sensors types to float can invalidate the registration and scale (while making the overall configuration more internally consistent). Hence, one may need to use an external mesh as a constraint, or otherwise subsequent registration may be needed. Example: 'cam1 cam3'.");

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
              "(measured in seconds). Existing image bracketing acts as an additional constraint.");

DEFINE_double(depth_tri_weight, 0.0,
              "The weight to give to the constraint that depth measurements agree with "
              "triangulated points. Use a bigger number as depth errors are usually on the "
              "order of 0.01 meters while reprojection errors are on the order of 1 pixel.");

DEFINE_string(mesh, "",
              "Use this geometry mapper mesh from a previous geometry mapper run to help constrain "
              "the calibration (e.g., use fused_mesh.ply).");

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

DEFINE_string(camera_poses, "",
              "Read the images and world-to-camera poses from this list. "
              "The same format is used as for when this tool saves the updated poses "
              "in the output directory.");

DEFINE_bool(use_initial_rig_transforms, false,
            "Use the transforms among the sensors of the rig specified via --rig_config. "
            "Otherwise derive it from the poses of individual cameras.");

DEFINE_bool(save_matches, false,
            "Save the interest point matches. Stereo Pipeline's viewer can be used for "
            "visualizing these.");

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

// Sort by timestamps adjusted to be relative to the ref camera clock
bool timestampLess(cameraImage i, cameraImage j) {
  return (i.ref_timestamp < j.ref_timestamp);
}

// Find the depth measurement. Use nearest neighbor interpolation
// to look into the depth cloud.
bool depthValue(// Inputs
                cv::Mat const& depth_cloud, Eigen::Vector2d const& dist_ip,
                // Output
                Eigen::Vector3d& depth_xyz) {
  depth_xyz = Eigen::Vector3d(0, 0, 0);  // initialize

  if (depth_cloud.cols == 0 && depth_cloud.rows == 0) return false;  // empty cloud

  int col = round(dist_ip[0]);
  int row = round(dist_ip[1]);

  if (col < 0 || row < 0 || col > depth_cloud.cols || row > depth_cloud.rows)
    LOG(FATAL) << "Out of range in depth cloud.";

  // After rounding one may hit the bound
  if (col == depth_cloud.cols || row == depth_cloud.rows)
    return false;

  cv::Vec3f cv_depth_xyz = depth_cloud.at<cv::Vec3f>(row, col);

  // Skip invalid measurements
  if (cv_depth_xyz == cv::Vec3f(0, 0, 0))
    return false;

  depth_xyz = Eigen::Vector3d(cv_depth_xyz[0], cv_depth_xyz[1], cv_depth_xyz[2]);

  return true;
}

// Project given images with optimized cameras onto mesh. It is
// assumed that the most up-to-date cameras were copied/interpolated
// form the optimizer structures into the world_to_cam vector.
void meshProjectCameras(std::vector<std::string> const& cam_names,
                        std::vector<camera::CameraParameters> const& cam_params,
                        std::vector<dense_map::cameraImage> const& cam_images,
                        std::vector<Eigen::Affine3d> const& world_to_cam,
                        mve::TriangleMesh::Ptr const& mesh,
                        std::shared_ptr<BVHTree> const& bvh_tree,
                        std::string const& out_dir) {
  if (cam_names.size() != cam_params.size())
    LOG(FATAL) << "There must be as many camera names as sets of camera parameters.\n";
  if (cam_images.size() != world_to_cam.size())
    LOG(FATAL) << "There must be as many camera images as camera poses.\n";
  if (out_dir.empty())
    LOG(FATAL) << "The output directory is empty.\n";
  
  char filename_buffer[1000];

  for (size_t cid = 0; cid < cam_images.size(); cid++) {
    double timestamp = cam_images[cid].timestamp;
    int cam_type = cam_images[cid].camera_type;

    // Must use the 10.7f format for the timestamp as everywhere else in the code
    snprintf(filename_buffer, sizeof(filename_buffer), "%s/%10.7f_%s",
             out_dir.c_str(), timestamp, cam_names[cam_type].c_str());
    std::string out_prefix = filename_buffer;  // convert to string

    std::cout << "Creating texture for: " << out_prefix << std::endl;
    meshProject(mesh, bvh_tree, cam_images[cid].image, world_to_cam[cid], cam_params[cam_type],
                out_prefix);
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

  if (FLAGS_num_overlaps < 1) LOG(FATAL) << "Number of overlaps must be positive.";

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

  if (FLAGS_camera_poses == "" && FLAGS_nvm == "")
    LOG(FATAL) << "Must specify the image list via --camera_poses or --nvm.\n";

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

// Look up each ref cam image by timestamp. In between any two ref cam timestamps,
// which are no further from each other than the bracket length, look up an image
// of each of the other camera types. If more than one choice, try to stay as close
// as possible to the midpoint of the two bracketing ref cam timestamps. This way
// there's more wiggle room later if one attempts to modify the timestamp offset.
// TODO(oalexan1): This function is too big and hard to understand.
// It needs to be broken up, with the bracketing being in its own function.  
void lookupImagesAndBrackets(  // Inputs
  int ref_cam_type, double bracket_len, double timestamp_offsets_max_change,
  double max_image_to_depth_timestamp_diff,
  std::vector<std::string> const& cam_names,
  std::vector<camera::CameraParameters> const& cam_params,
  std::vector<double> const& ref_timestamps,
  std::vector<std::vector<ImageMessage>> const& image_data,
  std::vector<std::vector<ImageMessage>> const& depth_data,
  std::vector<double> const& ref_to_cam_timestamp_offsets,
  // Outputs
  std::vector<dense_map::cameraImage>& cams, std::vector<double>& min_timestamp_offset,
  std::vector<double>& max_timestamp_offset) {
  std::cout << "Looking up the images and bracketing the timestamps." << std::endl;

  int num_ref_cams = ref_timestamps.size();
  int num_cam_types = cam_names.size();

  // Initialize the outputs
  cams.clear();
  min_timestamp_offset.resize(num_cam_types, -1.0e+100);
  max_timestamp_offset.resize(num_cam_types,  1.0e+100);

  // A lot of care is needed with positions. This remembers how we travel in time
  // for each camera type so we have fewer messages to search.
  // But if a mistake is done below it will mess up this bookkeeping.
  std::vector<int> image_start_positions(num_cam_types, 0);
  std::vector<int> cloud_start_positions(num_cam_types, 0);

  // Populate the data for each camera image
  for (int beg_ref_it = 0; beg_ref_it < num_ref_cams; beg_ref_it++) {
    if (ref_cam_type != 0)
      LOG(FATAL) << "It is assumed that the ref sensor id is 0.";

    bool save_grayscale = true; // for matching we will need grayscale

    // For when we have last ref timestamp and last other cam timestamp and they are equal
    int end_ref_it = beg_ref_it + 1;
    bool last_timestamp = (end_ref_it == num_ref_cams);
    if (last_timestamp) end_ref_it = beg_ref_it;

    for (int cam_type = ref_cam_type; cam_type < num_cam_types; cam_type++) {
      dense_map::cameraImage cam;
      bool success = false;

      // The ref cam does not need bracketing, but the others need to be bracketed
      // by ref cam, so there are two cases to consider.

      if (cam_type == ref_cam_type) {
        cam.camera_type   = cam_type;
        cam.timestamp     = ref_timestamps[beg_ref_it];
        cam.ref_timestamp = cam.timestamp;  // the time offset is 0 between ref and itself
        cam.beg_ref_index = beg_ref_it;
        cam.end_ref_index = beg_ref_it;  // same index for beg and end

        // Start looking up the image timestamp from this position. Some care
        // is needed here as we advance in time in image_start_positions[cam_type].
        double found_time = -1.0;
        // This has to succeed since this timestamp originally came from an existing image
        // Read from images/depth clouds stored in files
        bool have_lookup =  
          dense_map::lookupImage(cam.timestamp, image_data[cam_type],
                                 // Outputs
                                 cam.image, cam.image_name, 
                                 image_start_positions[cam_type],  // this will move forward
                                 found_time);
        
        if (!have_lookup)
          LOG(FATAL) << std::fixed << std::setprecision(17)
                     << "Cannot look up camera at time " << cam.timestamp << ".\n";

        // The exact time is expected
        if (found_time != cam.timestamp)
          LOG(FATAL) << std::fixed << std::setprecision(17)
                     << "Cannot look up camera at time " << cam.timestamp << ".\n";

        success = true;

      } else {
        // Need care here since sometimes ref_cam and current cam can have
        // exactly the same timestamp, so then bracketing should succeed.

        // Convert the bracketing timestamps to current cam's time
        double ref_to_cam_offset = ref_to_cam_timestamp_offsets[cam_type];
        double beg_timestamp     = ref_timestamps[beg_ref_it] + ref_to_cam_offset;
        double end_timestamp     = ref_timestamps[end_ref_it] + ref_to_cam_offset;
        if (end_timestamp == beg_timestamp && last_timestamp)  // necessary adjustment
          end_timestamp = std::nextafter(end_timestamp, end_timestamp + 1.0); 

        if (end_timestamp <= beg_timestamp)
          LOG(FATAL) << "Ref timestamps must be in strictly increasing order.\n";

        // Find the image timestamp closest to the midpoint of the brackets. This will give
        // more room to vary the timestamp later.
        double mid_timestamp = (beg_timestamp + end_timestamp)/2.0;

        // Search forward in time from image_start_positions[cam_type].
        // We will update that too later. One has to be very careful
        // with it so it does not go too far forward in time
        // so that at the next iteration we are passed what we
        // search for.
        int start_pos = image_start_positions[cam_type];  // care here
        double curr_timestamp = beg_timestamp;            // start here
        cv::Mat best_image;
        std::string best_image_name;
        double best_dist = 1.0e+100;
        double best_time = -1.0, found_time = -1.0;
        while (1) {
          if (found_time > end_timestamp) break;  // out of range

          cv::Mat image;
          std::string image_name;
            // Read from images/depth clouds stored in files
          bool have_lookup =
            dense_map::lookupImage(curr_timestamp, image_data[cam_type],
                                   // Outputs
                                   image, image_name,
                                   // care here, start_pos moves forward
                                   start_pos,
                                   // found_time will be updated now
                                   found_time);

          if (!have_lookup)
            break;  // Need not succeed, but then there's no need to go on as we are at the end

          // Check if the found time is in the bracket
          bool is_in_bracket = (beg_timestamp <= found_time && found_time < end_timestamp);
          double curr_dist = std::abs(found_time - mid_timestamp);

          if (curr_dist < best_dist && is_in_bracket) {
            best_dist = curr_dist;
            best_time = found_time;
            // Update the start position for the future only if this is a good
            // solution. Otherwise we may have moved too far.
            image_start_positions[cam_type] = start_pos;
            image.copyTo(best_image);
            best_image_name = image_name;
          }

          // Go forward in time. We count on the fact that
          // lookupImage() looks forward from given guess.
          // Careful here with the api of std::nextafter().
          curr_timestamp = std::nextafter(found_time, found_time + 1.0);
        }

        if (best_time < 0.0) continue;  // bracketing failed

        if (best_time > beg_timestamp && end_timestamp - beg_timestamp > bracket_len)
          continue;  // Must respect the bracket length, unless best time equals beg time
        
        // Note how we allow best_time == beg_timestamp if there's no other choice
        if (best_time < beg_timestamp || best_time >= end_timestamp)
          continue;  // no luck

        cam.camera_type   = cam_type;
        cam.timestamp     = best_time;
        cam.ref_timestamp = best_time - ref_to_cam_offset;
        cam.beg_ref_index = beg_ref_it;
        cam.end_ref_index = end_ref_it;
        cam.image         = best_image;
        cam.image_name    = best_image_name;

        success = true;
      }

      if (!success) continue;

      if (cam_type != ref_cam_type) {
        double ref_to_cam_offset = ref_to_cam_timestamp_offsets[cam_type];

        // cam.timestamp was chosen as centrally as possible so that
        // ref_timestamps[beg_ref_it] + ref_to_cam_offset <= cam.timestamp
        // and
        // cam.timestamp <= ref_timestamps[end_ref_it] + ref_to_cam_offset
        // Find the range of potential future values of ref_to_cam_offset so that
        // cam.timestamp still respects these bounds.
        min_timestamp_offset[cam_type]
          = std::max(min_timestamp_offset[cam_type], cam.timestamp - ref_timestamps[end_ref_it]);
        max_timestamp_offset[cam_type]
          = std::min(max_timestamp_offset[cam_type], cam.timestamp - ref_timestamps[beg_ref_it]);
      }

      // Look up the closest cloud in time (either before or after cam.timestamp)
      // This need not succeed.
      // Read from images/depth clouds stored in files
      cam.cloud_timestamp = -1.0;  // will change
      if (!depth_data.empty()) 
        dense_map::lookupImage(cam.timestamp,  // start looking from this time forward
                               depth_data[cam_type],
                               // Outputs
                               cam.depth_cloud, cam.depth_name, 
                               cloud_start_positions[cam_type],  // this will move forward
                               cam.cloud_timestamp);             // found time
      
      cams.push_back(cam);
    }  // end loop over camera types
  }    // end loop over ref images

  // See how many timestamps we have for each camera
  std::map<int, int> num_images;
  for (int cam_type_it = 0; cam_type_it < num_cam_types; cam_type_it++)
    num_images[cam_type_it] = 0;
  for (size_t cam_it = 0; cam_it < cams.size(); cam_it++)
    num_images[cams[cam_it].camera_type]++;

  bool is_good = true;
  for (int cam_type_it = 0; cam_type_it < num_cam_types; cam_type_it++) {
    std::cout << "Number of images for camera: " << cam_names[cam_type_it] << ": "
              << num_images[cam_type_it] << std::endl;

    if (num_images[cam_type_it] == 0) is_good = false;
  }

  if (!is_good)
    LOG(FATAL) << "Could not bracket all images. Cannot continue.\n";

  std::cout << "Timestamp offset allowed ranges based on current bracketing:\n";
  // Adjust for timestamp_offsets_max_change
  for (int cam_type = ref_cam_type; cam_type < num_cam_types; cam_type++) {
    if (cam_type == ref_cam_type) continue;  // bounds don't make sense here
    min_timestamp_offset[cam_type] = std::max(min_timestamp_offset[cam_type],
                                              ref_to_cam_timestamp_offsets[cam_type]
                                              - timestamp_offsets_max_change);
    max_timestamp_offset[cam_type] = std::min(max_timestamp_offset[cam_type],
                                              ref_to_cam_timestamp_offsets[cam_type]
                                              + timestamp_offsets_max_change);

    // Tighten a bit to ensure we don't exceed things when we add
    // and subtract timestamps later. Note that timestamps are
    // measured in seconds and fractions of a second since epoch and
    // can be quite large so precision loss can easily happen.
    min_timestamp_offset[cam_type] += 1.0e-5;
    max_timestamp_offset[cam_type] -= 1.0e-5;
    std::cout << std::setprecision(8) << cam_names[cam_type]
              << ": [" << min_timestamp_offset[cam_type]
              << ", " << max_timestamp_offset[cam_type] << "]\n";
  }

  // The images may need to be resized to be the same
  // size as in the calibration file. Sometimes the full-res images
  // can be so blurry that interest point matching fails, hence the
  // resizing.
  for (size_t it = 0; it < cams.size(); it++)
    dense_map::adjustImageSize(cam_params[cams[it].camera_type], cams[it].image);

  // Sort by the timestamp in reference camera time. This is essential
  // for matching each image to other images close in time. Note
  // that this does not affect the book-keeping of beg_ref_index
  // and end_ref_it in this vector because those indices point to
  // world_to_ref and ref_timestamp, which do not change.
  std::sort(cams.begin(), cams.end(), dense_map::timestampLess);
}

// TODO(oalexan1): Move to utils
void meshTriangulations(  // Inputs
  std::vector<camera::CameraParameters> const& cam_params,
  std::vector<dense_map::cameraImage> const& cams, std::vector<Eigen::Affine3d> const& world_to_cam,
  std::vector<std::map<int, int>> const& pid_to_cid_fid,
  std::vector<std::map<int, std::map<int, int>>> const& pid_cid_fid_inlier,
  std::vector<std::vector<std::pair<float, float>>> const& keypoint_vec,
  Eigen::Vector3d const& bad_xyz, double min_ray_dist, double max_ray_dist,
  mve::TriangleMesh::Ptr const& mesh, std::shared_ptr<BVHTree> const& bvh_tree,
  // Outputs
  std::vector<std::map<int, std::map<int, Eigen::Vector3d>>>& pid_cid_fid_mesh_xyz,
  std::vector<Eigen::Vector3d>& pid_mesh_xyz) {
  // Initialize the outputs
  pid_cid_fid_mesh_xyz.resize(pid_to_cid_fid.size());
  pid_mesh_xyz.resize(pid_to_cid_fid.size());

  for (size_t pid = 0; pid < pid_to_cid_fid.size(); pid++) {
    Eigen::Vector3d avg_mesh_xyz(0, 0, 0);
    int num_intersections = 0;

    for (auto cid_fid = pid_to_cid_fid[pid].begin(); cid_fid != pid_to_cid_fid[pid].end();
         cid_fid++) {
      int cid = cid_fid->first;
      int fid = cid_fid->second;

      // Initialize this
      pid_cid_fid_mesh_xyz[pid][cid][fid] = bad_xyz;

      // Deal with inliers only
      if (!dense_map::getMapValue(pid_cid_fid_inlier, pid, cid, fid))
        continue;

      // Intersect the ray with the mesh
      Eigen::Vector2d dist_ip(keypoint_vec[cid][fid].first, keypoint_vec[cid][fid].second);
      Eigen::Vector3d mesh_xyz(0.0, 0.0, 0.0);
      bool have_mesh_intersection
        = dense_map::ray_mesh_intersect(dist_ip, cam_params[cams[cid].camera_type],
                                        world_to_cam[cid], mesh, bvh_tree,
                                        min_ray_dist, max_ray_dist,
                                        // Output
                                        mesh_xyz);

      if (have_mesh_intersection) {
        pid_cid_fid_mesh_xyz[pid][cid][fid] = mesh_xyz;
        avg_mesh_xyz += mesh_xyz;
        num_intersections += 1;
      }
    }

    // Average the intersections of all rays with the mesh
    if (num_intersections >= 1)
      avg_mesh_xyz /= num_intersections;
    else
      avg_mesh_xyz = bad_xyz;

    pid_mesh_xyz[pid] = avg_mesh_xyz;
  }

  return;
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

// Given all the merged and filtered tracks in pid_cid_fid, for each
// image pair cid1 and cid2 with cid1 < cid2 < cid1 + num_overlaps + 1,
// save the matches of this pair which occur in the set of tracks.
void saveInlinerMatchPairs(// Inputs
                           std::vector<dense_map::cameraImage> const& cams,
                           int num_overlaps,
                           std::vector<std::map<int, int>> const& pid_to_cid_fid,
                           std::vector<std::vector<std::pair<float, float>>> const& keypoint_vec,
                           std::vector<std::map<int, std::map<int, int>>> const& pid_cid_fid_inlier,
                           std::string const& out_dir) {
  MATCH_MAP matches;

  for (size_t pid = 0; pid < pid_to_cid_fid.size(); pid++) {
    for (auto cid_fid1 = pid_to_cid_fid[pid].begin();
         cid_fid1 != pid_to_cid_fid[pid].end(); cid_fid1++) {
      int cid1 = cid_fid1->first;
      int fid1 = cid_fid1->second;

      for (auto cid_fid2 = pid_to_cid_fid[pid].begin();
           cid_fid2 != pid_to_cid_fid[pid].end(); cid_fid2++) {
        int cid2 = cid_fid2->first;
        int fid2 = cid_fid2->second;

        bool is_good = (cid1 < cid2 && cid2 < cid1 + num_overlaps + 1);
        if (!is_good)
          continue;

        // Consider inliers only
        if (!dense_map::getMapValue(pid_cid_fid_inlier, pid, cid1, fid1) ||
            !dense_map::getMapValue(pid_cid_fid_inlier, pid, cid2, fid2))
          continue;

        auto index_pair = std::make_pair(cid1, cid2);

        InterestPoint ip1(keypoint_vec[cid1][fid1].first, keypoint_vec[cid1][fid1].second);
        InterestPoint ip2(keypoint_vec[cid2][fid2].first, keypoint_vec[cid2][fid2].second);

        matches[index_pair].first.push_back(ip1);
        matches[index_pair].second.push_back(ip2);
      }
    }
  }  // End iterations over pid

  for (auto it = matches.begin(); it != matches.end(); it++) {
    auto & index_pair = it->first;
    dense_map::MATCH_PAIR const& match_pair = it->second;

    int left_index = index_pair.first;
    int right_index = index_pair.second;

    std::string match_dir = out_dir + "/matches";
    dense_map::createDir(match_dir);

    std::string suffix = "-inliers";
    std::string match_file = dense_map::matchFileName(match_dir,
                                                      cams[left_index].image_name,
                                                      cams[right_index].image_name,
                                                      suffix);

    std::cout << "Writing: " << cams[left_index].image_name << ' ' << cams[right_index].image_name
              << " " << match_file << std::endl;
    dense_map::writeMatchFile(match_file, match_pair.first, match_pair.second);
  }
}

// A function to copy image data from maps to vectors with the data stored
// chronologically in them, to speed up traversal.
void ImageDataToVectors
(// Inputs
 int ref_cam_type,
 std::map<int, std::map<double, dense_map::ImageMessage>> const& image_maps,
 std::map<int, std::map<double, dense_map::ImageMessage>> const& depth_maps,
 // Outputs
 std::vector<double>& ref_timestamps,
 std::vector<Eigen::Affine3d> & world_to_ref,
 std::vector<std::string> & ref_image_files,
 std::vector<std::vector<dense_map::ImageMessage>> & image_data,
 std::vector<std::vector<dense_map::ImageMessage>> & depth_data) {

  // Wipe the outputs
  ref_timestamps.clear();
  world_to_ref.clear();
  ref_image_files.clear();
  image_data.clear();
  depth_data.clear();
  
  // Find the range of sensor ids.
  int max_cam_type = 0;
  for (auto it = image_maps.begin(); it != image_maps.end(); it++)
    max_cam_type = std::max(max_cam_type, it->first);
  for (auto it = depth_maps.begin(); it != depth_maps.end(); it++)
    max_cam_type = std::max(max_cam_type, it->first);

  image_data.resize(max_cam_type + 1);
  depth_data.resize(max_cam_type + 1);
  for (size_t cam_type = 0; cam_type < image_data.size(); cam_type++) {

    auto image_map_it = image_maps.find(cam_type);
    if (image_map_it != image_maps.end()) {
      auto image_map = image_map_it->second; 

      for (auto it = image_map.begin(); it != image_map.end(); it++) {
        image_data[cam_type].push_back(it->second);
        
        // Collect the ref cam timestamps, world_to_ref, and image names,
        // in chronological order
        if (cam_type == ref_cam_type) {
          world_to_ref.push_back(it->second.world_to_cam);
          ref_timestamps.push_back(it->second.timestamp);
          ref_image_files.push_back(it->second.name);
        }
      }
    }
    
    auto depth_map_it = depth_maps.find(cam_type);
    if (depth_map_it != depth_maps.end()) {
      auto depth_map = depth_map_it->second; 
      for (auto it = depth_map.begin(); it != depth_map.end(); it++)
        depth_data[cam_type].push_back(it->second);
    }
  }
}

void readImageEntry(// Inputs
                    std::string const& image_file,
                    Eigen::Affine3d const& world_to_cam,
                    std::vector<std::string> const& cam_names,
                    // Outputs
                    std::map<int, std::map<double, dense_map::ImageMessage>> & image_maps,
                    std::map<int, std::map<double, dense_map::ImageMessage>> & depth_maps) {
  
  // The cam name is the subdir having the images
  std::string cam_name =
    fs::path(image_file).parent_path().filename().string();
    
  std::string basename = fs::path(image_file).filename().string();
  if (basename.empty() || basename[0] < '0' || basename[0] > '9')
    LOG(FATAL) << "Image name (without directory) must start with digits. Got: "
               << basename << "\n";
  double timestamp = atof(basename.c_str());

  // Infer cam type from cam name
  int cam_type = 0;
  bool success = false;
  for (size_t cam_it = 0; cam_it < cam_names.size(); cam_it++) {
    if (cam_names[cam_it] == cam_name) {
      cam_type = cam_it;
      success = true;
      break;
    }
  }
  if (!success) 
    LOG(FATAL) << "Could not extract cam_name from path/to/cam_name/image.jpg, "
               << "given image: " << image_file << "\n";
    
  // Aliases
  std::map<double, ImageMessage> & image_map = image_maps[cam_type];
  std::map<double, ImageMessage> & depth_map = depth_maps[cam_type];

  if (image_map.find(timestamp) != image_map.end())
    LOG(FATAL) << "Duplicate timestamp " << std::setprecision(17) << timestamp
               << " for sensor id " << cam_type << "\n";
  
  // Read the image as grayscale
  // TODO(oalexan1): How about color? But need grayscale for feature matching
  std::cout << "Reading: " << image_file << std::endl;
  image_map[timestamp].image        = cv::imread(image_file, cv::IMREAD_GRAYSCALE);
  image_map[timestamp].name         = image_file;
  image_map[timestamp].timestamp    = timestamp;
  image_map[timestamp].world_to_cam = world_to_cam;

  // Sanity check
  if (depth_map.find(timestamp) != depth_map.end())
    LOG(FATAL) << "Duplicate timestamp " << std::setprecision(17) << timestamp
               << " for sensor id " << cam_type << "\n";

  // Read the depth data, if present
  std::string depth_file = fs::path(image_file).replace_extension(".pc").string();
  if (fs::exists(depth_file)) {
    std::cout << "Reading: " << depth_file << std::endl;
    dense_map::readXyzImage(depth_file, depth_map[timestamp].image);
    depth_map[timestamp].name      = depth_file;
    depth_map[timestamp].timestamp = timestamp;
  }
}

void readDataFromList(// Inputs
                      std::string const& camera_poses_file, int ref_cam_type,
                      std::vector<std::string> const& cam_names,
                      // Outputs
                      std::vector<double>& ref_timestamps,
                      std::vector<Eigen::Affine3d>& world_to_ref,
                      std::vector<std::string> & ref_image_files,
                      std::vector<std::vector<ImageMessage>>& image_data,
                      std::vector<std::vector<ImageMessage>>& depth_data) {
  
  // Clear the outputs
  ref_timestamps.clear();
  world_to_ref.clear();
  ref_image_files.clear();
  image_data.clear();
  depth_data.clear();

  // Open the file
  std::cout << "Reading: " << camera_poses_file << std::endl;
  std::ifstream f;
  f.open(camera_poses_file.c_str(), std::ios::binary | std::ios::in);
  if (!f.is_open()) LOG(FATAL) << "Cannot open file for reading: " << camera_poses_file << "\n";

  // Read here temporarily the images and depth maps
  std::map<int, std::map<double, ImageMessage>> image_maps;
  std::map<int, std::map<double, ImageMessage>> depth_maps;

  std::string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::string image_file;
    std::istringstream iss(line);
    if (!(iss >> image_file))
      LOG(FATAL) << "Cannot parse the image file in: "
                 << camera_poses_file << "\n";

    // Read the camera to world transform
    Eigen::VectorXd vals(12);
    double val = -1.0;
    int count = 0;
    while (iss >> val) {
      if (count >= 12) break;
      vals[count] = val;
      count++;
    }

    if (count != 12)
      LOG(FATAL) << "Expecting 12 values for the transform on line:\n" << line << "\n";

    Eigen::Affine3d world_to_cam = vecToAffine(vals);
    readImageEntry(image_file, world_to_cam, cam_names,  
                   image_maps, depth_maps); // out 
  }

  // Put in vectors
  dense_map::ImageDataToVectors(// Inputs
                                ref_cam_type, image_maps,  depth_maps,
                                // Outputs
                                ref_timestamps, world_to_ref, ref_image_files,
                                image_data, depth_data);
  
  return;
}



// Read camera information and images from an NVM file, exported
// from Theia
// TODO(oalexan1): Move to utils
void readDataFromNvm(// Inputs
                     std::string const& nvm, int ref_cam_type,
                     std::vector<std::string> const& cam_names,
                     // Outputs
                     std::vector<double>& ref_timestamps,
                     std::vector<Eigen::Affine3d>& world_to_ref,
                     std::vector<std::string>    & ref_image_files,
                     std::vector<std::vector<ImageMessage>>& image_data,
                     std::vector<std::vector<ImageMessage>>& depth_data) {

  std::vector<Eigen::Matrix2Xd> cid_to_keypoint_map;
  std::vector<std::string> cid_to_filename;
  std::vector<std::map<int, int> > pid_to_cid_fid;
  std::vector<Eigen::Vector3d> pid_to_xyz;
  std::vector<Eigen::Affine3d> cid_to_cam_t_global;
  
  // cid_to_cam_t_global has world_to_cam
  dense_map::ReadNVM(nvm,  
                     &cid_to_keypoint_map,  
                     &cid_to_filename,  
                     &pid_to_cid_fid,  
                     &pid_to_xyz,  
                     &cid_to_cam_t_global);
  
  // Read here temporarily the images and depth maps
  std::map<int, std::map<double, dense_map::ImageMessage>> image_maps;
  std::map<int, std::map<double, dense_map::ImageMessage>> depth_maps;
  
  for (size_t it = 0; it < cid_to_filename.size(); it++) {

    // Aliases
    auto const& image_file = cid_to_filename[it];
    auto const& world_to_cam = cid_to_cam_t_global[it];

    readImageEntry(image_file, world_to_cam, cam_names,  
                   image_maps, depth_maps); // out 
  }

  // This entails some book-keeping
  // TODO(oalexan1): Just keep image_maps and depth_maps and change the book-keeping
  // to use std::map rather than std::vector iterators
  dense_map::ImageDataToVectors(// Inputs
                                ref_cam_type, image_maps, depth_maps,
                                // Outputs
                                ref_timestamps, world_to_ref, ref_image_files,
                                image_data, depth_data);
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

// TODO(oalexan1):  Move to utils!
// Given a set of points in 3D, heuristically estimate what it means
// for two points to be "not far" from each other. The logic is to
// find a bounding box of an inner cluster and multiply that by 0.2.
double estimateCloseDistance(std::vector<Eigen::Vector3d> const& vec) {
  Eigen::Vector3d range;
  int num_pts = vec.size();
  if (num_pts <= 0)
    LOG(FATAL) << "Empty set of points.\n";  // to avoid a segfault

  std::vector<double> vals(num_pts);
  for (int it = 0; it < range.size(); it++) {  // iterate in each coordinate
    // Sort all values in given coordinate
    for (int p = 0; p < num_pts; p++)
      vals[p] = vec[p][it];
    std::sort(vals.begin(), vals.end());

    // Find some percentiles
    int min_p = round(num_pts*0.25);
    int max_p = round(num_pts*0.75);
    if (min_p >= num_pts) min_p = num_pts - 1;
    if (max_p >= num_pts) max_p = num_pts - 1;
    double min_val = vals[min_p], max_val = vals[max_p];
    range[it] = 0.2*(max_val - min_val);
  }

  // Find the average of all ranges
  double range_val = 0.0;
  for (int it = 0; it < range.size(); it++)
    range_val += range[it];
  range_val /= range.size();

  return range_val;
}
  

// Given two sets of 3D points, find the rotation + translation + scale
// which best maps the first set to the second.
// Source: http://en.wikipedia.org/wiki/Kabsch_algorithm

// TODO(oalexan1): Move to utils!
// TODO(oalexan1): Use robust version!  
void Find3DAffineTransform(Eigen::Matrix3Xd const & in,
                           Eigen::Matrix3Xd const & out,
                           Eigen::Affine3d* result) {
  // Default output
  result->linear() = Eigen::Matrix3d::Identity(3, 3);
  result->translation() = Eigen::Vector3d::Zero();

  if (in.cols() != out.cols())
    throw "Find3DAffineTransform(): input data mis-match";

  // Local copies we can modify
  Eigen::Matrix3Xd local_in = in, local_out = out;

  // First find the scale, by finding the ratio of sums of some distances,
  // then bring the datasets to the same scale.
  double dist_in = 0, dist_out = 0;
  for (int col = 0; col < local_in.cols()-1; col++) {
    dist_in  += (local_in.col(col+1) - local_in.col(col)).norm();
    dist_out += (local_out.col(col+1) - local_out.col(col)).norm();
  }
  if (dist_in <= 0 || dist_out <= 0)
    return;
  double scale = dist_out/dist_in;
  local_out /= scale;

  // Find the centroids then shift to the origin
  Eigen::Vector3d in_ctr = Eigen::Vector3d::Zero();
  Eigen::Vector3d out_ctr = Eigen::Vector3d::Zero();
  for (int col = 0; col < local_in.cols(); col++) {
    in_ctr  += local_in.col(col);
    out_ctr += local_out.col(col);
  }
  in_ctr /= local_in.cols();
  out_ctr /= local_out.cols();
  for (int col = 0; col < local_in.cols(); col++) {
    local_in.col(col)  -= in_ctr;
    local_out.col(col) -= out_ctr;
  }

  // SVD
  Eigen::Matrix3d Cov = local_in * local_out.transpose();
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(Cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // Find the rotation
  double d = (svd.matrixV() * svd.matrixU().transpose()).determinant();
  if (d > 0)
    d = 1.0;
  else
    d = -1.0;
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity(3, 3);
  I(2, 2) = d;
  Eigen::Matrix3d R = svd.matrixV() * I * svd.matrixU().transpose();

  // The final transform
  result->linear() = scale * R;
  result->translation() = scale*(out_ctr - R*in_ctr);
}
  
// TODO(oalexan1): Move to utils
  
// Extract control points and the images they correspond 2 from
// a hugin project file
void ParseHuginControlPoints(std::string const& hugin_file,
                             std::vector<std::string> * images,
                             Eigen::MatrixXd * points) {
  
  // Initialize the outputs
  (*images).clear();
  *points = Eigen::MatrixXd(6, 1);

  std::ifstream hf(hugin_file.c_str());
  if (!hf.good())
    LOG(FATAL) << "ParseHuginControlPoints(): Could not open hugin file: " << hugin_file;

  int num_points = 0;
  std::string line;
  while (getline(hf, line)) {
    // Parse for images
    if (line.find("i ") == 0) {
      size_t it = line.find("n\"");
      if (it == std::string::npos)
        LOG(FATAL) << "ParseHuginControlPoints(): Invalid line: " << line;
      it += 2;
      std::string image;
      while (it < line.size() && line[it] != '"') {
        image += line[it];
        it++;
      }
      (*images).push_back(image);
    }

    // Parse control points
    if (line.find("c ") == 0) {
      // First wipe all letters
      std::string orig_line = line;
      char * ptr = const_cast<char*>(line.c_str());
      for (size_t i = 0; i < line.size(); i++) {
        // Wipe some extra chars
        if ( (ptr[i] >= 'a' && ptr[i] <= 'z') ||
             (ptr[i] >= 'A' && ptr[i] <= 'Z') )
          ptr[i] = ' ';
      }

      // Out of a line like:
      // c n0 N1 x367 y240 X144.183010710425 Y243.04008545843 t0
      // we store the numbers, 0, 1, 367, 240, 144.183010710425 243.04008545843
      // as a column.
      // The stand for left image index, right image index,
      // left image x, left image y, right image x, right image y.
      double a, b, c, d, e, f;
      if (sscanf(ptr, "%lf %lf %lf %lf %lf %lf", &a, &b, &c, &d, &e, &f) != 6)
        LOG(FATAL) << "ParseHuginControlPoints(): Could not scan line: " << line;

      // The left and right images must be different
      if (a == b)
        LOG(FATAL) << "The left and right images must be distinct. "
                   << "Offending line in " << hugin_file << " is:\n"
                   << orig_line << "\n";

      num_points++;
      (*points).conservativeResize(Eigen::NoChange_t(), num_points);
      (*points).col(num_points-1) << a, b, c, d, e, f;
    }
  }
}

// A little helper function
bool is_blank(std::string const& line) {
  return (line.find_first_not_of(" \t\n\v\f\r") == std::string::npos);
}

// Parse a file having on each line xyz coordinates
void ParseXYZ(std::string const& xyz_file,
                              Eigen::MatrixXd * xyz) {
  // Initialize the outputs
  *xyz = Eigen::MatrixXd(3, 1);

  std::ifstream hf(xyz_file.c_str());
  if (!hf.good())
    LOG(FATAL) << "ParseXYZ(): Could not open hugin file: " << xyz_file;

  int num_points = 0;
  std::string line;
  while (getline(hf, line)) {
    // Ignore lines starting with comments and empty lines
    if (line.find("#") == 0 || is_blank(line)) continue;

    // Apparently sometimes empty lines show up as if of length 1
    if (line.size() == 1)
      continue;

    // Replace commas with spaces
    char * ptr = const_cast<char*>(line.c_str());
    for (size_t c = 0; c < line.size(); c++)
      if (ptr[c] == ',') ptr[c] = ' ';
    double x, y, z;
    if (sscanf(line.c_str(), "%lf %lf %lf", &x, &y, &z) != 3)
      LOG(FATAL) << "ParseXYZ(): Could not scan line: '" << line << "'\n";

    num_points++;
    (*xyz).conservativeResize(Eigen::NoChange_t(), num_points);
    (*xyz).col(num_points-1) << x, y, z;
  }
}

// Apply a given transform to the given set of cameras.
// We assume that the transform is of the form
// T(x) = scale * rotation * x + translation
void TransformCameras(Eigen::Affine3d const& T, std::vector<Eigen::Affine3d> &world_to_cam) {
  
  // Inverse of rotation component
  double scale = pow(T.linear().determinant(), 1.0 / 3.0);
  Eigen::MatrixXd Tinv = (T.linear()/scale).inverse();

  for (size_t cid = 0; cid < world_to_cam.size(); cid++) {
    world_to_cam[cid].linear() = world_to_cam[cid].linear()*Tinv;
    world_to_cam[cid].translation() = scale*world_to_cam[cid].translation() -
      world_to_cam[cid].linear()*T.translation();
  }
}

// Apply same transform as above to points
void TransformPoints(Eigen::Affine3d const& T, std::vector<Eigen::Vector3d> *xyz) {
  for (size_t pid = 0; pid < (*xyz).size(); pid++)
    (*xyz)[pid] = T * (*xyz)[pid];
}

// Apply a registration transform to a rig. The only thing that
// changes is scale, as the rig transforms are between coordinate
// systems of various cameras.
void TransformRig(Eigen::Affine3d const& T, std::vector<Eigen::Affine3d> & ref_to_cam_trans) {
  double scale = pow(T.linear().determinant(), 1.0 / 3.0);
  for (size_t cam_type = 0; cam_type < ref_to_cam_trans.size(); cam_type++) 
    ref_to_cam_trans[cam_type].translation() *= scale;
}
  
// Register a map to world coordinates from user-supplied data, or simply
// verify how well the map performs with this data.
// It is assumed all images are from the reference camera
// TODO(oalexan1): This needs to be modularized.

Eigen::Affine3d registerTransforms(std::string const& hugin_file, std::string const& xyz_file,
                                  camera::CameraParameters const& ref_cam_params,
                                  std::vector<std::string> const& cid_to_filename,
                                  std::vector<Eigen::Affine3d>  & world_to_cam_trans) { 
  
  // Get the interest points in the images, and their positions in
  // the world coordinate system, as supplied by a user.
  // Parse and concatenate that information from multiple files.
  std::vector<std::string> images;
  Eigen::MatrixXd user_ip;
  Eigen::MatrixXd user_xyz;
  
  ParseHuginControlPoints(hugin_file, &images, &user_ip);
  ParseXYZ(xyz_file, &user_xyz);

  int num_points = user_ip.cols();
  if (num_points != user_xyz.cols())
    LOG(FATAL) << "Could not parse an equal number of control "
               << "points and xyz coordinates. Their numbers are "
               << num_points << " vs " << user_xyz.cols() << ".\n";


  std::map<std::string, int> filename_to_cid;
  for (size_t cid = 0; cid < cid_to_filename.size(); cid++)
    filename_to_cid[cid_to_filename[cid]] = cid;

  // Wipe images that are missing from the map
  std::map<int, int> cid2cid;
  int good_cid = 0;
  for (size_t cid = 0; cid < images.size(); cid++) {
    std::string image = images[cid];
    if (filename_to_cid.find(image) == filename_to_cid.end()) {
      LOG(WARNING) << "Will ignore image missing from map: " << image;
      continue;
    }
    cid2cid[cid] = good_cid;
    images[good_cid] = images[cid];
    good_cid++;
  }
  images.resize(good_cid);

  // Remove points corresponding to images missing from map
  int good_pid = 0;
  for (int pid = 0; pid < num_points; pid++) {
    int id1 = user_ip(0, pid);
    int id2 = user_ip(1, pid);
    if (cid2cid.find(id1) == cid2cid.end() || cid2cid.find(id2) == cid2cid.end()) {
      continue;
    }
    user_ip.col(good_pid) = user_ip.col(pid);
    user_xyz.col(good_pid) = user_xyz.col(pid);
    good_pid++;
  }
  user_ip.conservativeResize(Eigen::NoChange_t(), good_pid);
  user_xyz.conservativeResize(Eigen::NoChange_t(), good_pid);
  num_points = good_pid;
  for (int pid = 0; pid < num_points; pid++) {
    int id1 = user_ip(0, pid);
    int id2 = user_ip(1, pid);
    if (cid2cid.find(id1) == cid2cid.end() || cid2cid.find(id2) == cid2cid.end())
      LOG(FATAL) << "Book-keeping failure in registration.";
    user_ip(0, pid) = cid2cid[id1];
    user_ip(1, pid) = cid2cid[id2];
  }


  if (num_points < 3) 
    LOG(FATAL) << "Must have at least 3 points to apply registration. Got: "
               << num_points << "\n";
  
  // Iterate over the control points in the hugin file. Copy the
  // control points to the list of user keypoints, and create the
  // corresponding user_pid_to_cid_fid.
  std::vector<Eigen::Matrix2Xd> user_cid_to_keypoint_map;
  std::vector<std::map<int, int> > user_pid_to_cid_fid;
  user_cid_to_keypoint_map.resize(cid_to_filename.size());
  user_pid_to_cid_fid.resize(num_points);
  for (int pid = 0; pid < num_points; pid++) {
    // Left and right image indices
    int id1 = user_ip(0, pid);
    int id2 = user_ip(1, pid);

    // Sanity check
    if (id1 < 0 || id2 < 0 ||
        id1 >= static_cast<int>(images.size()) ||
        id2 >= static_cast<int>(images.size()) )
      LOG(FATAL) << "Invalid image indices in the hugin file: " << id1 << ' ' << id2;

    // Find the corresponding indices in the map where these keypoints will go to
    if (filename_to_cid.find(images[id1]) == filename_to_cid.end())
      LOG(FATAL) << "File missing from map: " << images[id1];
    if (filename_to_cid.find(images[id2]) == filename_to_cid.end())
      LOG(FATAL) << "File missing from map: " << images[id2];
    int cid1 = filename_to_cid[images[id1]];
    int cid2 = filename_to_cid[images[id2]];

    // Append to the keypoints for cid1
    Eigen::Matrix<double, 2, -1> &M1 = user_cid_to_keypoint_map[cid1];  // alias
    Eigen::Matrix<double, 2, -1> N1(M1.rows(), M1.cols()+1);
    N1 << M1, user_ip.block(2, pid, 2, 1);  // left image pixel x and pixel y
    M1.swap(N1);

    // Append to the keypoints for cid2
    Eigen::Matrix<double, 2, -1> &M2 = user_cid_to_keypoint_map[cid2];  // alias
    Eigen::Matrix<double, 2, -1> N2(M2.rows(), M2.cols()+1);
    N2 << M2, user_ip.block(4, pid, 2, 1);  // right image pixel x and pixel y
    M2.swap(N2);

    // The corresponding user_pid_to_cid_fid
    user_pid_to_cid_fid[pid][cid1] = user_cid_to_keypoint_map[cid1].cols()-1;
    user_pid_to_cid_fid[pid][cid2] = user_cid_to_keypoint_map[cid2].cols()-1;
  }

  // Apply undistortion
  Eigen::Vector2d output;
  for (size_t cid = 0; cid < user_cid_to_keypoint_map.size(); cid++) {
    for (int i = 0; i < user_cid_to_keypoint_map[cid].cols(); i++) {
      ref_cam_params.Convert<camera::DISTORTED, camera::UNDISTORTED_C>
        (user_cid_to_keypoint_map[cid].col(i), &output);
      user_cid_to_keypoint_map[cid].col(i) = output;
    }
  }

  // Triangulate to find the coordinates of the current points
  // in the virtual coordinate system
  std::vector<Eigen::Vector3d> unreg_pid_to_xyz;
  bool rm_invalid_xyz = false;  // there should be nothing to remove hopefully
  Triangulate(rm_invalid_xyz,
              ref_cam_params.GetFocalLength(),
              world_to_cam_trans,
              user_cid_to_keypoint_map,
              &user_pid_to_cid_fid,
              &unreg_pid_to_xyz);

  double mean_err = 0;
  for (int i = 0; i < user_xyz.cols(); i++) {
    Eigen::Vector3d a = unreg_pid_to_xyz[i];
    Eigen::Vector3d b = user_xyz.col(i);
    mean_err += (a-b).norm();
  }
  mean_err /= user_xyz.cols();
  std::cout << "Mean absolute error before registration: " << mean_err << " meters" << std::endl;
  std::cout << "Un-transformed computed xyz -- measured xyz -- error diff -- error norm (meters)"
            << std::endl;

  for (int i = 0; i < user_xyz.cols(); i++) {
    Eigen::Vector3d a = unreg_pid_to_xyz[i];
    Eigen::Vector3d b = user_xyz.col(i);
    std::cout << print_vec(a) << " -- "
              << print_vec(b) << " -- "
              << print_vec(a-b) << " -- "
              << print_vec((a - b).norm())
              << std::endl;
  }


  // Find the transform from the computed map coordinate system
  // to the world coordinate system.
  int np = unreg_pid_to_xyz.size();
  Eigen::Matrix3Xd in(3, np);
  for (int i = 0; i < np; i++)
    in.col(i) = unreg_pid_to_xyz[i];

  Eigen::Affine3d registration_trans;  
  Find3DAffineTransform(in, user_xyz, &registration_trans);

  // Transform the map to the world coordinate system
  TransformCameras(registration_trans, world_to_cam_trans);
  
  mean_err = 0.0;
  for (int i = 0; i < user_xyz.cols(); i++)
    mean_err += (registration_trans*in.col(i) - user_xyz.col(i)).norm();
  mean_err /= user_xyz.cols();

  // We don't use LOG(INFO) below, as it does not play well with
  // Eigen.
  double scale = pow(registration_trans.linear().determinant(), 1.0 / 3.0);
  std::cout << "Registration transform (to measured world coordinates)." << std::endl;
  std::cout << "Rotation:\n" << registration_trans.linear() / scale << std::endl;
  std::cout << "Scale:\n" << scale << std::endl;
  std::cout << "Translation:\n" << registration_trans.translation().transpose()
            << std::endl;

  std::cout << "Mean absolute error after registration: "
            << mean_err << " meters" << std::endl;

  std::cout << "Transformed computed xyz -- measured xyz -- "
            << "error diff - error norm (meters)" << std::endl;
  for (int i = 0; i < user_xyz.cols(); i++) {
    Eigen::Vector3d a = registration_trans*in.col(i);
    Eigen::Vector3d b = user_xyz.col(i);
    int id1 = user_ip(0, i);
    int id2 = user_ip(1, i);

    std::cout << print_vec(a) << " -- "
              << print_vec(b) << " -- "
              << print_vec(a - b) << " -- "
              << print_vec((a - b).norm()) << " -- "
              << images[id1] << ' '
              << images[id2] << std::endl;
  }


  return registration_trans;
}

}  // namespace dense_map

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
  // image_data[i] having data in increasing order of timestamp.  This
  // way it is fast to find next timestamps after a given one.
  // TODO(oalexan1): It would be better if image_data[i] was an
  // std::map<double, dense_map::ImageMessage> with the key being a
  // timestamp, and we would cache and advance the iterator for that key between
  // successive invocations, rather than caching and advancing the vector index.
  // Then looking up by a given timestamp would be more direct than now,
  // while looking up next timestamp after the given one would be as now.
  std::vector<double> ref_timestamps; // Timestamps for the ref cameras
  std::vector<std::vector<dense_map::ImageMessage>> image_data;
  std::vector<std::vector<dense_map::ImageMessage>> depth_data;
  std::vector<std::string> ref_image_files;
  if (FLAGS_camera_poses != "")
    dense_map::readDataFromList(FLAGS_camera_poses, ref_cam_type, cam_names, // in
                                ref_timestamps, world_to_ref, ref_image_files,
                                image_data, depth_data); // out
  else if (FLAGS_nvm != "") 
    dense_map::readDataFromNvm(FLAGS_nvm, ref_cam_type, cam_names, // in
                               ref_timestamps, world_to_ref, ref_image_files,
                               image_data, depth_data); // out

  // Keep here the images, timestamps, and bracketing information
  std::vector<dense_map::cameraImage> cams;
  //  The range of ref_to_cam_timestamp_offsets[cam_type] before
  //  getting out of the bracket.
  std::vector<double> min_timestamp_offset, max_timestamp_offset;

  // Select the images to use and bracket them using the ref cam
  // TODO(oalexan1): This should not be used without the rig
  // constraint. In that case we need to find from Theia which images to match.
  dense_map::lookupImagesAndBrackets(  // Inputs
    ref_cam_type, FLAGS_bracket_len, FLAGS_timestamp_offsets_max_change,
    FLAGS_max_image_to_depth_timestamp_diff, cam_names, cam_params,
    ref_timestamps, image_data, depth_data,
    ref_to_cam_timestamp_offsets,
    // Outputs
    cams, min_timestamp_offset, max_timestamp_offset);

  // If we have initial rig transforms, compute the transform from the
  // world to every camera based on the rig transforms and ref_to_cam
  // transforms. It assumes that world_to_ref and ref_to_cam
  // are up-to-date. Use the version of calc_world_to_cam_using_rig
  // without world_to_cam_vec, on input which was not computed yet.

  // TODO(oalexan1): Don't use this logic if we have no rig
  if (use_initial_rig_transforms) {
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
      = dense_map::registerTransforms(FLAGS_hugin_file, FLAGS_xyz_file,  
                                      cam_params[ref_cam_type],  
                                      ref_image_files,  
                                      world_to_ref);
    // The above transformed world_to_ref. Also transform world_to_cam.
    // TODO(oalexan1): Apply transform passed in from outside
    dense_map::TransformCameras(registration_trans, world_to_cam);
    dense_map::TransformRig(registration_trans, ref_to_cam_trans); // this uses different logic

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

  // Detect and match features
  std::vector<std::vector<std::pair<float, float>>> keypoint_vec;
  std::vector<std::map<int, int>> pid_to_cid_fid;
  dense_map::detectMatchFeatures(  // Inputs
    cams, cam_params, FLAGS_out_dir, FLAGS_save_matches, world_to_cam,
    FLAGS_num_overlaps, FLAGS_initial_max_reprojection_error, FLAGS_num_match_threads,
    FLAGS_verbose,
    // Outputs
    keypoint_vec, pid_to_cid_fid);

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

  for (int pass = 0; pass < FLAGS_calibrator_num_passes; pass++) {
    std::cout << "\nOptimization pass " << pass + 1 << " / " << FLAGS_calibrator_num_passes << "\n";

    // The transforms from the world to all cameras must be updated
    // given the current state of optimization
    // TODO(oalexan1): The call below is likely not necessary since this function
    // is already called earlier, and also whenever a pass finishes, see below.
    dense_map::calc_world_to_cam_rig_or_not(  // Inputs
      FLAGS_no_rig, cams, world_to_ref_vec, ref_timestamps, ref_to_cam_vec, world_to_cam_vec,
      ref_to_cam_timestamp_offsets,
      // Output
      world_to_cam);

    std::vector<Eigen::Vector3d> xyz_vec;
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
        // Update world_to_ref as well, as it is part of the sparse map
        if (cams[cid].camera_type == ref_cam_type) {
          int ref_cid = cams[cid].beg_ref_index;
          world_to_ref[ref_cid] = world_to_cam[cid];
          // For consistency, also keep world_to_ref_vec updated, even
          // if it is not used in this case
          dense_map::rigid_transform_to_array
            (world_to_ref[ref_cid], &world_to_ref_vec[dense_map::NUM_RIGID_PARAMS * ref_cid]);
        }
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

  dense_map::writeImageList(FLAGS_out_dir, cams, world_to_cam);
  
  bool model_rig = (!FLAGS_no_rig);
  dense_map::writeRigConfig(FLAGS_out_dir, model_rig, ref_cam_type, cam_names,
                            cam_params, ref_to_cam_trans, depth_to_image,
                            ref_to_cam_timestamp_offsets);

  if (FLAGS_export_to_voxblox)
    dense_map::exportToVoxblox(cam_names, cams, depth_to_image, world_to_cam, FLAGS_out_dir);

  if (FLAGS_save_transformed_depth_clouds)
    dense_map::saveTransformedDepthClouds(cam_names, cams, depth_to_image,
                                          world_to_cam, FLAGS_out_dir);

  return 0;
} // NOLINT // TODO(oalexan1): Remove this, after making the code more modular
