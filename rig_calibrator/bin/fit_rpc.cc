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

// TODO(oalexan1): Do not use -out_dir. Use -out_config.
// TODO(oalexan1): Print an underestimate and overestimate for the undistorted win.

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

#include <rig_calibrator/dense_map_utils.h>
#include <camera_model/rpc_distortion.h>
#include <camera_model/camera_params.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <Eigen/Geometry>
#include <Eigen/Core>

#include <boost/filesystem.hpp>

#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>

namespace fs = boost::filesystem;

// TODO(oalexan1): Must have sensor name as an option. For now it defaults to sensor 0.

DEFINE_int32(rpc_degree, -1,
             "The degree of the RPC model to fit.");

DEFINE_int32(num_samples, -1,
             "The number of row and column samples to use to fit the RPC model.");

DEFINE_int32(num_iterations, 20, "How many solver iterations to perform in calibration.");

DEFINE_int32(num_opt_threads, 16, "How many threads to use in the optimization.");

DEFINE_double(parameter_tolerance, 1e-12, "Stop when the optimization variables change by "
              "less than this.");

DEFINE_string(camera_config, "",
              "Read the camera configuration from this file.");

DEFINE_string(out_dir, "",
              "Write here the camera configuration having the RPC fit.");

DEFINE_bool(verbose, false,
            "Print more information about what the tool is doing.");


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_camera_config.empty())
    LOG(FATAL) << "Camera config file was not specified.";
  
  if (FLAGS_out_dir.empty())
    LOG(FATAL) << "Output camera config directory was not specified.";

  if (FLAGS_rpc_degree <= 0)
    LOG(FATAL) << "The RPC degree must be positive.";

  if (FLAGS_num_samples <= 0)
      LOG(FATAL) << "The number of samples must be positive.";

  int ref_cam_type = 0; // dictated by the API
  std::vector<std::string> cam_names;
  std::vector<Eigen::Affine3d> depth_to_image;
  std::vector<camera::CameraParameters> cam_params;
  std::vector<Eigen::Affine3d>          ref_to_cam_trans;
  std::vector<double>                   ref_to_cam_timestamp_offsets;
  bool use_initial_rig_transforms = true; // dictated by the api
  dense_map::readRigConfig(FLAGS_camera_config, use_initial_rig_transforms, ref_cam_type, cam_names,
                           cam_params, ref_to_cam_trans, depth_to_image,
                           ref_to_cam_timestamp_offsets);
  
  std::cout << "Focal length is " << cam_params[0].GetFocalVector().transpose() << std::endl;

  Eigen::VectorXd rpc_dist_coeffs;
  dense_map::fitRpcDist(FLAGS_rpc_degree, FLAGS_num_samples,
                        cam_params[0],
                        FLAGS_num_opt_threads, FLAGS_num_iterations,
                        FLAGS_parameter_tolerance,
                        FLAGS_verbose,
                        // Output
                        rpc_dist_coeffs);
  
  Eigen::VectorXd rpc_undist_coeffs;
  dense_map::fitRpcUndist(rpc_dist_coeffs,
                          FLAGS_num_samples,
                          cam_params[0],
                          FLAGS_num_opt_threads, FLAGS_num_iterations,
                          FLAGS_parameter_tolerance,
                          FLAGS_verbose,
                          // Output
                          rpc_undist_coeffs);

  dense_map::RPCLensDistortion rpc;
  rpc.set_distortion_parameters(rpc_dist_coeffs);
  rpc.set_undistortion_parameters(rpc_undist_coeffs);

  dense_map::evalRpcDistUndist(FLAGS_num_samples, cam_params[0], rpc);

  // Create the model with RPC distortion. Note how we pass both the distortion
  // and undistortion RPC coefficients.
  cam_params[0].SetDistortion(rpc.dist_undist_params());

  dense_map::writeRigConfig(FLAGS_out_dir, use_initial_rig_transforms, ref_cam_type, cam_names,
                            cam_params, ref_to_cam_trans, depth_to_image,
                            ref_to_cam_timestamp_offsets);
  
  return 0;
}
