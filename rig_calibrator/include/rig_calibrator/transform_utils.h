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
  
}  // end namespace dense_map

#endif  // TRANSFORM_UTILS_H_
