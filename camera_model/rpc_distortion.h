/* Copyright (c) 2017, United States Government, as represented by the
 * Administrator of the National Aeronautics and Space Administration.
 * 
 * All rights reserved.
 * 
 * The Astrobee platform is licensed under the Apache License, Version 2.0
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

#ifndef RIG_CALIBRATOR_RPC_DISTORTION_H
#define RIG_CALIBRATOR_RPC_DISTORTION_H

#include <Eigen/Core>

#include <string>
#include <vector>

namespace camera {
  // forward declaration
  class CameraParameters;
}

namespace dense_map {
class RPCLensDistortion {
  int m_rpc_degree;
  Eigen::VectorXd m_distortion, m_undistortion;

  // This variable signals that the coefficients needed to perform undistortion
  // have been computed.
  bool m_can_undistort;

 public:
  explicit RPCLensDistortion(); // NOLINT
  explicit RPCLensDistortion(Eigen::VectorXd const& params);
  void reset(int rpc_degree);  // Form the identity transform
  Eigen::VectorXd distortion_parameters() const;
  Eigen::VectorXd undistortion_parameters() const;
  void set_image_size(Eigen::Vector2i const& image_size);
  void set_distortion_parameters(Eigen::VectorXd const& params);
  void set_undistortion_parameters(Eigen::VectorXd const& params);
  int num_dist_params() const { return m_distortion.size(); }

  Eigen::Vector2d distorted_coordinates(Eigen::Vector2d const& p) const;
  Eigen::Vector2d undistorted_coordinates(Eigen::Vector2d const& p) const;

  bool has_fast_distort  () const {return true;}
  bool has_fast_undistort() const {return true;}

  void write(std::ostream& os) const;
  void read(std::istream& os);

  static  std::string class_name()       { return "RPC"; }
  std::string name      () const { return class_name();  }

  void scale(double scale);

  bool can_undistort() const { return m_can_undistort; }
  static void init_as_identity(Eigen::VectorXd & params);
  static void increment_degree(Eigen::VectorXd & params);

  Eigen::VectorXd dist_undist_params();

  void set_dist_undist_params(Eigen::VectorXd const& dist_undist);

  Eigen::Vector2d distort_centered(Eigen::Vector2d const& p) const;
  Eigen::Vector2d undistort_centered(Eigen::Vector2d const& p) const;
  
};

void fitRpcDist(int rpc_degree, int num_samples, int num_exclude_boundary_pixels,
                camera::CameraParameters const& cam_params,
                int num_opt_threads, int num_iterations, double parameter_tolerance,
                bool verbose,
                // Output
                Eigen::VectorXd & rpc_dist_coeffs);

void fitRpcUndist(Eigen::VectorXd const & rpc_dist_coeffs,
                  int num_samples, int num_exclude_boundary_pixels,
                  camera::CameraParameters const& cam_params,
                  int num_opt_threads, int num_iterations, double parameter_tolerance,
                  bool verbose,
                  // output
                  Eigen::VectorXd & rpc_undist_coeffs);

void evalRpcDistUndist(int num_samples, int num_exclude_boundary_pixels,
                       camera::CameraParameters const& cam_params,
                       RPCLensDistortion const& rpc);
  
}
  
#endif  // RIG_CALIBRATOR_RPC_DISTORTION_H
