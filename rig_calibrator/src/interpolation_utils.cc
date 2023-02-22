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

#include <rig_calibrator/interpolation_utils.h>
#include <glog/logging.h>

namespace dense_map {

// Given two poses aff0 and aff1, and 0 <= alpha <= 1, do linear interpolation.
Eigen::Affine3d linearInterp(double alpha, Eigen::Affine3d const& aff0,
                             Eigen::Affine3d const& aff1) {
  Eigen::Quaternion<double> rot0(aff0.linear());
  Eigen::Quaternion<double> rot1(aff1.linear());
  
  Eigen::Vector3d trans0 = aff0.translation();
  Eigen::Vector3d trans1 = aff1.translation();

  Eigen::Affine3d result;

  result.translation() = (1.0 - alpha) * trans0 + alpha * trans1;
  result.linear() = rot0.slerp(alpha, rot1).toRotationMatrix();

  return result;
}

// Given two poses aff0 and aff1, and t0 <= t <= t1, do linear interpolation.
Eigen::Affine3d linearInterp(double t0, double t, double t1, Eigen::Affine3d const& aff0,
                             Eigen::Affine3d const& aff1) {

  bool is_good = ((t0 <= t) && (t <= t1));
  if (!is_good) 
    LOG(FATAL) << "Incorrect bounds in interpolation.\n";
  
  double den = t1 - t0;
  if (den == 0.0)
    den = 1.0;
  
  double alpha = (t - t0) / den;
  if (alpha < 0.0 || alpha > 1.0) LOG(FATAL) << "Out of bounds in interpolation.\n";

  return dense_map::linearInterp(alpha, aff0, aff1);
}
  
// Given a set of poses indexed by timestamp in an std::map, find the
// interpolated pose at desired timestamp. This is efficient
// only for very small maps. Else use the StampedPoseStorage class.
bool findInterpPose(double desired_time, std::map<double, Eigen::Affine3d> const& poses,
                      Eigen::Affine3d& interp_pose) {
  double left_time = std::numeric_limits<double>::max();
  double right_time = -left_time;
  for (auto it = poses.begin(); it != poses.end(); it++) {
    double curr_time = it->first;
    if (curr_time <= desired_time) {
      left_time = curr_time;  // this can only increase
    }
    if (curr_time >= desired_time) {
      // Here an "if" was used rather than "else", to be able to
      // handle the case when left_time == curr_time == right_time.
      right_time = curr_time;
      break;  // just passed the desired time, can stop now
    }
  }

  if (left_time > right_time) {
    // Could not bracket the desired time
    return false;
  }

  double alpha = (desired_time - left_time) / (right_time - left_time);
  if (left_time == right_time) alpha = 0.0;  // handle division by 0
  interp_pose = linearInterp(alpha, poses.find(left_time)->second, poses.find(right_time)->second);
  return true;
}

}  // end namespace dense_map
