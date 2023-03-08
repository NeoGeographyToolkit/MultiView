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

#ifndef TRANSFORM_UTILS_H_
#define TRANSFORM_UTILS_H_

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace dense_map {

// Save an affine transform represented as a matrix to a string.
std::string affineToStr(Eigen::Affine3d const& M);

// Form an affine transform from 12 values
Eigen::Affine3d vecToAffine(Eigen::VectorXd const& vals);

// Calculate interpolated world to reference sensor transform. Take into account
// that the timestamp is for a sensor which may not be the reference one, so
// an offset needs to be applied.
Eigen::Affine3d calc_interp_world_to_ref(const double* beg_world_to_ref_t,
                                         const double* end_world_to_ref_t,
                                         double beg_ref_stamp,
                                         double end_ref_stamp,
                                         double ref_to_cam_offset,
                                         double cam_stamp);

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
                                        double cam_stamp);

void affine_transform_to_array(Eigen::Affine3d const& aff, double* arr);
void array_to_affine_transform(Eigen::Affine3d& aff, const double* arr);
  
// Extract a rigid transform to an array of length NUM_RIGID_PARAMS
void rigid_transform_to_array(Eigen::Affine3d const& aff, double* arr);

// Convert an array of length NUM_RIGID_PARAMS to a rigid
// transform. Normalize the quaternion to make it into a rotation.
void array_to_rigid_transform(Eigen::Affine3d& aff, const double* arr);

// Given two sets of 3D points, find the rotation + translation + scale
// which best maps the first set to the second.
// Source: http://en.wikipedia.org/wiki/Kabsch_algorithm
// TODO(oalexan1): Use the version robust to outliers!  
void Find3DAffineTransform(Eigen::Matrix3Xd const & in,
                           Eigen::Matrix3Xd const & out,
                           Eigen::Affine3d* result);
  
}  // end namespace dense_map

#endif  // TRANSFORM_UTILS_H_
