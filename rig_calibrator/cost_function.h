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

#ifndef RIG_COST_FUNCTION_H_
#define RIG_COST_FUNCTION_H_

#include <vector>

namespace dense_map {

// Forward declarations
class cameraImage;
class RigSet;
 
// Find pointers to the camera and reference images that bracket the
// camera image. Great care is needed here. Two cases are considered,
// if there is a rig or not. If no_rig is true, then the reference images are
// the same as the camera images. 
void calcBracketing(// Inputs
                  bool no_rig, int cid, int cam_type,
                  std::vector<dense_map::cameraImage> const& cams,
                  std::vector<double> const& ref_timestamps,
                  dense_map::RigSet   const& R,
                  // Will not be changed but need access
                  std::vector<double> & world_to_cam_vec,
                  std::vector<double> & world_to_ref_vec,
                  std::vector<double> & ref_to_cam_vec,
                  std::vector<double> & ref_identity_vec,
                  std::vector<double> & right_identity_vec,
                  // Outputs
                  double* & beg_cam_ptr, 
                  double* & end_cam_ptr, 
                  double* & ref_to_cam_ptr,
                  double  & beg_ref_timestamp, 
                  double  & end_ref_timestamp,
                  double  & cam_timestamp);

}  // namespace dense_map

#endif  // RIG_COST_FUNCTION_H_
