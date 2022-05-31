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

// This header file will be included when building without ROS and
// other Astrobee dependencies.  It will make some empty classes to
// make compilation succeed.

#ifndef DENSE_MAP_NOROS_UTILS_H_
#define DENSE_MAP_NOROS_UTILS_H_

#include <opencv2/imgproc.hpp>
#include <glog/logging.h>
#include <camera_model/camera_params.h>
#include <vector>

namespace rosbag {
  struct MessageInstance {};
}

namespace dense_map {

inline bool lookupImage(double desired_time, std::vector<rosbag::MessageInstance> const& bag_msgs,
                        bool save_grayscale, cv::Mat& image, int& bag_pos, double& found_time) {
  LOG(FATAL) << "This version of lookupImage() is not meant to be called without Astrobee/ROS. "
             << "There is a logic error somewhere.\n";
}

inline bool lookupCloud(double desired_time, std::vector<rosbag::MessageInstance> const& bag_msgs,
                        double max_time_diff, cv::Mat& cloud, int& bag_pos, double& found_time) {
  LOG(FATAL) << "This version of lookupCloud() is not meant to be called without Astrobee/ROS. "
             << "There is a logic error somewhere.\n";
}

inline void readLuaConfig(bool & have_rig_transforms, int & ref_cam_type,
                           std::vector<std::string> & cam_names,
                           std::vector<camera::CameraParameters> & cam_params,
                           std::vector<Eigen::Affine3d> & ref_to_cam_trans,
                           std::vector<Eigen::Affine3d> & depth_to_image,
                           std::vector<double> & ref_to_cam_timestamp_offsets) {
  LOG(FATAL) << "This version of readLuaConfig() is not meant to be called without Astrobee/ROS. "
             << "There is a logic error somewhere.\n";
}

// This is a wrapper for writeLuaConfig() hiding some things
inline void writeLuaConfig                                                 // NOLINT
(std::vector<std::string>              const& cam_names,                    // NOLINT
 std::vector<camera::CameraParameters> const& cam_params,                   // NOLINT
 std::vector<Eigen::Affine3d>          const& ref_to_cam_trans,             // NOLINT
 std::vector<double>                   const& ref_to_cam_timestamp_offset,  // NOLINT
 std::vector<Eigen::Affine3d>          const& depth_to_image_trans) {       // NOLINT
  LOG(FATAL) << "This version of readLuaConfig() is not meant to be called without Astrobee/ROS. "
             << "There is a logic error somewhere.\n";
}

}  // end namespace dense_map

#endif  // DENSE_MAP_NOROS_UTILS_H_
