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

#include <iostream>

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

// Given a set of poses indexed by time, interpolate or extrapolate
// (within range of extrap_len) at a set of target timestamps. Go
// forward in time both in the input and the target, which makes the
// complexity linear rather than quadratic.
void interpOrExtrap(std::map<double, Eigen::Affine3d> const& input_poses,
                    std::map<double, std::string> const& target,
                    double extrap_len,  bool nearest_neighbor_interp,
                    // Outputs
                    std::vector<std::string> & found_images,
                    std::vector<Eigen::Affine3d> & found_poses) {

  found_images.clear();
  found_poses.clear();
  
  double beg_timestamp = input_poses.begin()->first;
  double end_timestamp = input_poses.rbegin()->first;
  
  // Iterate over all timestamps to interpolate at, and simultaneously
  // over the existing images to interpolate into, both in increasing
  // order of time.  This makes complexity linear rather than
  // quadratic. This is possible because both the inputs and outputs
  // are sorted by time. Some care is needed.

  auto start = input_poses.begin(); // track where we are in the input 
  for (auto curr_it = target.begin(); curr_it != target.end(); curr_it++) {
    
    double curr_timestamp = curr_it->first;
    std::string const& image_file = curr_it->second;
    
    if (curr_timestamp < beg_timestamp - extrap_len ||
        curr_timestamp > end_timestamp + extrap_len) {
      std::cout << "Warning: Cannot find camera pose for image " << image_file
               << " as it is too far in time from existing images. Ignoring it.\n";
      continue;
    }
    
    Eigen::Affine3d found_pose;
    if (curr_timestamp <= beg_timestamp) {
      // Use extrapolation
      found_pose = input_poses.begin()->second; // pose for earliest time
    } else if (curr_timestamp >= end_timestamp) {
      found_pose = input_poses.rbegin()->second; // pose for the latest tame
    } else {
      // Use interpolation
      bool success = false;
      for (auto map_it = start; map_it != input_poses.end(); map_it++) {
        
        // Find the right bracketing iterator
        auto right_map_it = map_it;
        right_map_it++;
        if (right_map_it == input_poses.end())
          right_map_it = map_it; // fall back to left if at the end
        
        double left_timestamp = map_it->first;
        double right_timestamp = right_map_it->first;

        if (left_timestamp > curr_timestamp) {
          // Went too far
          break;
        }

        // Update this for next time. It always moves forward in time,
        // and points to a location left of current timestamp.
        start = map_it;
        
        bool is_good = (curr_timestamp <= right_timestamp);
        if (!is_good) 
          continue; // too early

        if (nearest_neighbor_interp) {
          // Use the nearest neighbor
          if (std::abs(left_timestamp - curr_timestamp) <=
              std::abs(right_timestamp - curr_timestamp)) {
            curr_timestamp = left_timestamp;
          } else {
            curr_timestamp = right_timestamp;
          }
        }
        
        // Interpolate at desired time
        found_pose
          = dense_map::linearInterp(left_timestamp, curr_timestamp, right_timestamp,
                                    map_it->second, right_map_it->second);
        
        found_images.push_back(image_file);
        found_poses.push_back(found_pose);
        success = true;
        break;
      }
      
      if (!success) {
        std::cout << "Warning: Cannot compute camera pose for image " << image_file
                  << ". Ignoring it.\n";
      }
    }
  }  
}
  
}  // end namespace dense_map
