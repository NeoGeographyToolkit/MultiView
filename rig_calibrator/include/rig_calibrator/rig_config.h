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

#ifndef RIG_CALIBRATOR_RIG_CONFIG_H
#define RIG_CALIBRATOR_RIG_CONFIG_H

// A structure to hold info about a set of rigs

#include <camera_model/camera_model.h>

#include <map>
#include <vector>
#include <glog/logging.h>

namespace dense_map {

  struct RigSet {

    // For rig i, sensor_names[i] will list the sensor names in that rig.
    // sensor_names[i][0] is the reference sensor. All sensor
    // names are unique.
    std::vector<std::vector<std::string>> sensor_names;

    // A transform from a reference sensor to a sensor in the rig
    // (including to itself, that's the identity transform). This
    // concatenates all such transforms for all rigs, in the order
    // given by concatenating sensor_names[0], sensor_names[1], etc.
    std::vector<Eigen::Affine3d> ref_to_cam_trans;

    // Depth-to-image transform for each sensor. It is the identity
    // if there are no depth transforms. One per sensor. All
    // concatenated.
    std::vector<Eigen::Affine3d> depth_to_image;

    // The value to add to each ref sensor time to get a given sensor
    // time. One per sensor. All concatenated.
    std::vector<double> ref_to_cam_timestamp_offsets;
    
    // Each sensor's intrinsics. All concatenated. 
    std::vector<camera::CameraParameters> cam_params;

    // If this sensor is a reference sensor for one of the rig.
    bool isRefSensor(std::string const& sensor_name);

    // Sanity checks
    void validate();
  };
  
}  // end namespace dense_map

#endif  // RIG_CALIBRATOR_RIG_CONFIG_H
