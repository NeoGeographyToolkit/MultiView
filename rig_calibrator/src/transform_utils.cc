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
#include <rig_calibrator/interpolation_utils.h>

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

// Calculate interpolated world to reference sensor transform. Take into account
// that the timestamp is for a sensor which may not be the reference one, so
// an offset needs to be applied.
Eigen::Affine3d calc_interp_world_to_ref(const double* beg_world_to_ref_t,
                                         const double* end_world_to_ref_t,
                                         double beg_ref_stamp,
                                         double end_ref_stamp,
                                         double ref_to_cam_offset,
                                         double cam_stamp) {
    Eigen::Affine3d beg_world_to_ref_aff;
    array_to_rigid_transform(beg_world_to_ref_aff, beg_world_to_ref_t);

    Eigen::Affine3d end_world_to_ref_aff;
    array_to_rigid_transform(end_world_to_ref_aff, end_world_to_ref_t);

    // Handle the degenerate case
    if (end_ref_stamp == beg_ref_stamp) 
      return beg_world_to_ref_aff;
    
    // Covert from cam time to ref time and normalize. It is very
    // important that below we subtract the big numbers from each
    // other first, which are the timestamps, then subtract whatever
    // else is necessary. Otherwise we get problems with numerical
    // precision with CERES.
    double alpha = ((cam_stamp - beg_ref_stamp) - ref_to_cam_offset)
        / (end_ref_stamp - beg_ref_stamp);
    
    if (alpha < 0.0 || alpha > 1.0) LOG(FATAL) << "Out of bounds in interpolation.\n";

    // Interpolate at desired time
    Eigen::Affine3d interp_world_to_ref_aff = dense_map::linearInterp(alpha, beg_world_to_ref_aff,
                                                                      end_world_to_ref_aff);

    return interp_world_to_ref_aff;
}
  
// Calculate interpolated world to camera transform. Use the
// convention that if beg_ref_stamp == end_ref_stamp, then this is the
// reference camera, and then only beg_world_to_ref_t is used, while
// end_world_to_ref_t is undefined. For the reference camera it is
// also expected that ref_to_cam_aff is the identity. This saves some
// code duplication later as the ref cam need not be treated
// separately.
// TODO(oalexan1): There is a bug! Just because beg_ref_stamp and end_ref_stamp
// are equal, it should not mean that ref_to_cam_aff is the identity!
Eigen::Affine3d calc_world_to_cam_trans(const double* beg_world_to_ref_t,
                                        const double* end_world_to_ref_t,
                                        const double* ref_to_cam_trans,
                                        double beg_ref_stamp,
                                        double end_ref_stamp,
                                        double ref_to_cam_offset,
                                        double cam_stamp) {

  Eigen::Affine3d interp_world_to_cam_aff;
  if (beg_ref_stamp == end_ref_stamp) {
    Eigen::Affine3d beg_world_to_ref_aff;
    array_to_rigid_transform(beg_world_to_ref_aff, beg_world_to_ref_t);
    interp_world_to_cam_aff = beg_world_to_ref_aff;
  } else {

    Eigen::Affine3d ref_to_cam_aff;
    array_to_rigid_transform(ref_to_cam_aff, ref_to_cam_trans);

    Eigen::Affine3d interp_world_to_ref_aff =
      calc_interp_world_to_ref(beg_world_to_ref_t, end_world_to_ref_t,  
                               beg_ref_stamp,  
                               end_ref_stamp,  ref_to_cam_offset,  
                               cam_stamp);
    
    interp_world_to_cam_aff = ref_to_cam_aff * interp_world_to_ref_aff;
  }

  return interp_world_to_cam_aff;
}

// Extract a affine transform to an array of length NUM_AFFINE_PARAMS
void affine_transform_to_array(Eigen::Affine3d const& aff, double* arr) {
  Eigen::MatrixXd M = aff.matrix();
  int count = 0;
  // The 4th row always has 0, 0, 0, 1
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 4; col++) {
      arr[count] = M(row, col);
      count++;
    }
  }
}

// Convert an array of length NUM_AFFINE_PARAMS to a affine
// transform. Normalize the quaternion to make it into a rotation.
void array_to_affine_transform(Eigen::Affine3d& aff, const double* arr) {
  Eigen::MatrixXd M = Eigen::Matrix<double, 4, 4>::Identity();

  int count = 0;
  // The 4th row always has 0, 0, 0, 1
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 4; col++) {
      M(row, col) = arr[count];
      count++;
    }
    }

  aff.matrix() = M;
}


// Extract a rigid transform to an array of length NUM_RIGID_PARAMS
void rigid_transform_to_array(Eigen::Affine3d const& aff, double* arr) {
  for (size_t it = 0; it < 3; it++) arr[it] = aff.translation()[it];

  Eigen::Quaterniond R(aff.linear());
  arr[3] = R.x();
  arr[4] = R.y();
  arr[5] = R.z();
  arr[6] = R.w();
}

// Convert an array of length NUM_RIGID_PARAMS to a rigid
// transform. Normalize the quaternion to make it into a rotation.
void array_to_rigid_transform(Eigen::Affine3d& aff, const double* arr) {
  for (size_t it = 0; it < 3; it++) aff.translation()[it] = arr[it];

  Eigen::Quaterniond R(arr[6], arr[3], arr[4], arr[5]);
  R.normalize();

  aff = Eigen::Affine3d(Eigen::Translation3d(arr[0], arr[1], arr[2])) * Eigen::Affine3d(R);
}
  
}  // end namespace dense_map
