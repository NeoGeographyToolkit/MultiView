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

#include <rig_calibrator/rig_config.h>

#include <iostream>
#include <set>
#include <fstream>
#include <iomanip>

namespace dense_map {

  void RigSet::validate() {

    if (sensor_names.empty()) 
      LOG(FATAL) << "Found an empty set of rigs.\n";

    size_t num_sensors = 0;
    std::set<std::string> all_sensors; // checks for duplicates
    for (size_t rig_it = 0; rig_it < sensor_names.size(); rig_it++) {
      if (sensor_names[rig_it].empty()) 
        LOG(FATAL) << "Found a rig with no sensors.\n";

      num_sensors += sensor_names[rig_it].size();
      for (size_t sensor_it = 0; sensor_it < sensor_names[rig_it].size(); sensor_it++)
        all_sensors.insert(sensor_names[rig_it][sensor_it]);
    }

    if (num_sensors != all_sensors.size())
      LOG(FATAL) << "Found a duplicate sensor name in the rig set.\n";

    if (num_sensors != ref_to_cam_trans.size()) 
      LOG(FATAL) << "Number of sensors is not equal to number of ref-to-cam transforms.\n";
    
    if (num_sensors != depth_to_image.size()) 
      LOG(FATAL) << "Number of sensors is not equal to number of depth-to-image transforms.\n";

    if (num_sensors != ref_to_cam_timestamp_offsets.size()) 
      LOG(FATAL) << "Number of sensors is not equal to number of ref-to-cam timestamp offsets.\n";
    
    if (num_sensors != cam_params.size()) 
      LOG(FATAL) << "Number of sensors is not equal to number of camera models.\n";
  }
  
  // A ref sensor is the first sensor on each rig
  bool RigSet::isRefSensor(std::string const& sensor_name) {
    for (size_t rig_it = 0; rig_it < sensor_names.size(); rig_it++) 
      if (sensor_names[rig_it][0] == sensor_name) 
        return true;
    return false;
  }
}  // end namespace dense_map
