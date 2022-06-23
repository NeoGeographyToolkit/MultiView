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

#include <rig_calibrator/transform_utils.h>

#include <glog/logging.h>

#include <string>
#include <vector>

namespace dense_map {

// Save an affine transform represented as a matrix to a string.
std::string affineToStr(Eigen::Affine3d const& M) {
  Eigen::MatrixXd T = M.matrix();
  std::ostringstream os;
  os.precision(17);
  os << T(0, 0) << " " << T(0, 1) << " " << T(0, 2) << " "
     << T(1, 0) << " " << T(1, 1) << " " << T(1, 2) << " "
     << T(2, 0) << " " << T(2, 1) << " " << T(2, 2) << " "
     << T(0, 3) << " " << T(1, 3) << " " << T(2, 3);

  return os.str();
}

// TODO(oalexan1): Move to utils
// Form an affine transform from 12 values
Eigen::Affine3d vecToAffine(Eigen::VectorXd const& vals) {
  if (vals.size() != 12)
    LOG(FATAL) << "An affine transform must have 12 parameters.\n";

  Eigen::Affine3d M = Eigen::Affine3d::Identity();
  Eigen::MatrixXd T = M.matrix();

  int count = 0;

  // linear part
  T(0, 0) = vals[count++];
  T(0, 1) = vals[count++];
  T(0, 2) = vals[count++];
  T(1, 0) = vals[count++];
  T(1, 1) = vals[count++];
  T(1, 2) = vals[count++];
  T(2, 0) = vals[count++];
  T(2, 1) = vals[count++];
  T(2, 2) = vals[count++];

  // translation part
  T(0, 3) = vals[count++];
  T(1, 3) = vals[count++];
  T(2, 3) = vals[count++];

  M.matrix() = T;

  return M;
}
   
}  // end namespace dense_map
