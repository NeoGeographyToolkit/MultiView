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

#include <camera_model/camera_params.h>
#include <camera_model/rpc_distortion.h>

#include <Eigen/Dense>
#include <gflags/gflags.h>

#include <iostream>

#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <iostream>

camera::CameraParameters::CameraParameters(Eigen::Vector2i const& image_size,
    Eigen::Vector2d const& focal_length,
    Eigen::Vector2d const& optical_center,
    Eigen::VectorXd const& distortion,
    DistortionType distortion_type) {
  SetDistortedSize(image_size);
  SetDistortedCropSize(image_size);
  SetUndistortedSize(image_size);
  m_focal_length = focal_length;
  m_optical_offset = optical_center;
  m_crop_offset.setZero();
  SetDistortion(distortion);
  m_distortion_type = distortion_type;
}

void camera::CameraParameters::SetDistortedSize(Eigen::Vector2i const& image_size) {
  m_distorted_image_size = image_size;
  m_distorted_half_size = image_size.cast<double>() / 2.0;
}

const Eigen::Vector2i& camera::CameraParameters::GetDistortedSize() const {
  return m_distorted_image_size;
}

const Eigen::Vector2d& camera::CameraParameters::GetDistortedHalfSize() const {
  return m_distorted_half_size;
}

void camera::CameraParameters::SetDistortedCropSize(Eigen::Vector2i const& crop_size) {
  m_distorted_crop_size = crop_size;
}

const Eigen::Vector2i& camera::CameraParameters::GetDistortedCropSize() const {
  return m_distorted_crop_size;
}

void camera::CameraParameters::SetUndistortedSize(Eigen::Vector2i const& image_size) {
  m_undistorted_size = image_size;
  m_undistorted_half_size = image_size.cast<double>() / 2.0;
}

const Eigen::Vector2i& camera::CameraParameters::GetUndistortedSize() const {
  return m_undistorted_size;
}

const Eigen::Vector2d& camera::CameraParameters::GetUndistortedHalfSize() const {
  return m_undistorted_half_size;
}

void camera::CameraParameters::SetCropOffset(Eigen::Vector2i const& crop) {
  m_crop_offset = crop;
}

const Eigen::Vector2i& camera::CameraParameters::GetCropOffset() const {
  return m_crop_offset;
}

void camera::CameraParameters::SetOpticalOffset(Eigen::Vector2d const& offset) {
  m_optical_offset = offset;
}

const Eigen::Vector2d& camera::CameraParameters::GetOpticalOffset() const {
  return m_optical_offset;
}

void camera::CameraParameters::SetFocalLength(Eigen::Vector2d const& f) {
  m_focal_length = f;
}

double camera::CameraParameters::GetFocalLength() const {
  return m_focal_length.mean();
}

const Eigen::Vector2d& camera::CameraParameters::GetFocalVector() const {
  return m_focal_length;
}

void camera::CameraParameters::SetDistortion(Eigen::VectorXd const& distortion) {

  // Reset this. Will be needed only with RPC distortion.
  m_rpc = dense_map::RPCLensDistortion(); 
  
  m_distortion_coeffs = distortion;

  // Ensure variables are initialized
  m_distortion_precalc1 = 0;
  m_distortion_precalc2 = 0;
  m_distortion_precalc3 = 0;

  switch (m_distortion_coeffs.size()) {
  case 0:
    // No lens distortion!
    break;
  case 1:
    // FOV model
    // inverse alpha
    m_distortion_precalc1 = 1 / distortion[0];
    // Inside tangent function
    m_distortion_precalc2 = 2 * tan(distortion[0] / 2);
    break;
  case 4:
    // Fall through intended.
  case 5:
    // Tsai model
    // There doesn't seem like there are any precalculations we can use.
    break;
  default:
    // Try to do RPC
    try { 
      m_rpc.set_dist_undist_params(m_distortion_coeffs);
    } catch(std::exception const& e) {
      LOG(FATAL) << "Recieved irregular distortion vector size. Size = "
                 << m_distortion_coeffs.size() << "\n"
                 << "Additional message: " << e.what() << "\n";
    }
  }
}

// This must be called before a model having RPC distortion can be used
// for undistortion. Here it is assumed that the distortion component
// of m_distortion_coeffs is up-to-date, and its undistortion component
// must be updated.
void camera::CameraParameters::updateRpcUndistortion(int num_threads) {
  int num_samples = 400; // in each of rows and columns; should be enough
  bool verbose = false;
  int num_iterations = 100; // should be plenty
  double parameter_tolerance = 1e-12; // should be enough

  std::cout << "Finding RPC undistortion. Using " << num_samples
            << " samples in width and height, "
            << num_iterations << " iterations, and "
            << num_threads << " threads." << std::endl;
  
  if (m_distortion_coeffs.size() % 2 != 0) 
    LOG(FATAL) << "Must have an even number of RPC distortion coefficients.\n";

  // m_distortion_coeffs stores both distortion and undistortion rpc coeffs. Get
  // the distortion ones, and update the undistortion ones.
  // This is quite confusing, but an outside user of this class need not know
  // these details
  int num_dist = m_distortion_coeffs.size()/2;
  Eigen::VectorXd rpc_dist_coeffs(num_dist);
  for (int it = 0; it < num_dist; it++)
    rpc_dist_coeffs[it] = m_distortion_coeffs[it];

  Eigen::VectorXd rpc_undist_coeffs;
  dense_map::fitRpcUndist(rpc_dist_coeffs, num_samples,
                          *this,
                          num_threads, num_iterations,
                          parameter_tolerance,
                          verbose,
                          // Output
                          rpc_undist_coeffs);

  dense_map::RPCLensDistortion rpc;
  rpc.set_distortion_parameters(rpc_dist_coeffs);
  rpc.set_undistortion_parameters(rpc_undist_coeffs);
  dense_map::evalRpcDistUndist(num_samples, *this, rpc);

  // Copy back the updated values
  for (int it = 0; it < num_dist; it++)
    m_distortion_coeffs[it + num_dist] = rpc_undist_coeffs[it];
}

const Eigen::VectorXd& camera::CameraParameters::GetDistortion() const {
  return m_distortion_coeffs;
}

void camera::CameraParameters::DistortCentered(Eigen::Vector2d const& undistorted_c,
                                               Eigen::Vector2d* distorted_c) const {
  // We assume that input x and y are pixel values that have
  // undistorted_len_x/2.0 and undistorted_len_y/2.0 subtracted from
  // them. The outputs will have distorted_len_x/2.0 and
  // distorted_len_y/2.0 subtracted from them.
  if (m_distortion_coeffs.size() == 0) {
    // There is no distortion
    *distorted_c = undistorted_c + m_optical_offset - m_distorted_half_size;
  } else if (m_distortion_coeffs.size() == 1) {
    // This is the FOV model
    Eigen::Vector2d norm = undistorted_c.cwiseQuotient(m_focal_length);
    double ru = norm.norm();
    double rd = atan(ru * m_distortion_precalc2) * m_distortion_precalc1;
    double conv;
    if (ru > 1e-5) {
      conv = rd / ru;
    } else {
      conv = 1;
    }
    *distorted_c = (m_optical_offset - m_distorted_half_size) +
      conv * norm.cwiseProduct(m_focal_length);
  } else if (m_distortion_coeffs.size() == 4 ||
             m_distortion_coeffs.size() == 5) {
    // Tsai lens distortion
    double k1 = m_distortion_coeffs[0];
    double k2 = m_distortion_coeffs[1];
    double p1 = m_distortion_coeffs[2];
    double p2 = m_distortion_coeffs[3];
    double k3 = 0;
    if (m_distortion_coeffs.size() == 5)
      k3 = m_distortion_coeffs[4];

    // To relative coordinates
    Eigen::Vector2d norm = undistorted_c.cwiseQuotient(m_focal_length);
    double r2 = norm.squaredNorm();

    // Radial distortion
    double radial_dist = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    *distorted_c = radial_dist * norm;

    // Tangential distortion
    *distorted_c +=
      Eigen::Vector2d(2 * p1 * norm[0] * norm[1] + p2 * (r2 + 2 * norm[0] * norm[0]),
                      p1 * (r2 + 2 * norm[1] * norm[1]) + 2 * p2 * norm[0] * norm[1]);

    // Back to absolute coordinates.
    *distorted_c = distorted_c->cwiseProduct(m_focal_length) +
      (m_optical_offset - m_distorted_half_size);
  } else {
    // If we got so far, we validated that RPC distortion should work
    *distorted_c = m_rpc.distort_centered(undistorted_c);
    //LOG(ERROR) << "Unknown distortion vector size.";
  }
}

void camera::CameraParameters::UndistortCentered(Eigen::Vector2d const& distorted_c,
                                                 Eigen::Vector2d *undistorted_c) const {
  // We assume that input x and y are pixel values that have
  // distorted_len_x and distorted_len_y subtracted from them. The
  // outputs will have undistorted_len_x and undistorted_len_y
  // subtracted from them.
  if (m_distortion_coeffs.size() == 0) {
    // No lens distortion
    *undistorted_c = distorted_c - (m_optical_offset - m_distorted_half_size);
  } else if (m_distortion_coeffs.size() == 1) {
    // FOV lens distortion
    Eigen::Vector2d norm =
      (distorted_c - (m_optical_offset - m_distorted_half_size)).cwiseQuotient(m_focal_length);
    double rd = norm.norm();
    double ru = tan(rd * m_distortion_coeffs[0]) / m_distortion_precalc2;
    double conv = 1.0;
    if (rd > 1e-5)
      conv = ru / rd;
    *undistorted_c = conv * norm.cwiseProduct(m_focal_length);
  } else if (m_distortion_coeffs.size() == 4 ||
             m_distortion_coeffs.size() == 5) {
    // Tsai lens distortion
    cv::Mat src(1, 1, CV_64FC2);
    cv::Mat dst(1, 1, CV_64FC2);
    Eigen::Map<Eigen::Vector2d> src_map(src.ptr<double>()), dst_map(dst.ptr<double>());
    cv::Mat dist_int_mat(3, 3, cv::DataType<double>::type),
      undist_int_mat(3, 3, cv::DataType<double>::type);
    cv::Mat cvdist;
    cv::eigen2cv(m_distortion_coeffs, cvdist);
    cv::eigen2cv(GetIntrinsicMatrix<DISTORTED>(), dist_int_mat);
    cv::eigen2cv(GetIntrinsicMatrix<UNDISTORTED>(), undist_int_mat);
    src_map = distorted_c + m_distorted_half_size;
    cv::undistortPoints(src, dst, dist_int_mat, cvdist, cv::Mat(), undist_int_mat);
    *undistorted_c = dst_map - m_undistorted_half_size;
  } else {
    // If we got so far, we validated that RPC distortion should work
    *undistorted_c = m_rpc.undistort_centered(distorted_c);
    //LOG(ERROR) << "Unknown distortion vector size.";
  }
}

// The 'scale' variable is useful when we have the distortion model for a given
// image, and want to apply it to a version of that image at a different resolution,
// with 'scale' being the ratio of the width of the image at different resolution
// and the one at the resolution at which the distortion model is computed.
void camera::CameraParameters::GenerateRemapMaps(cv::Mat* remap_map, double scale) {
  remap_map->create(scale*m_undistorted_size[1], scale*m_undistorted_size[0], CV_32FC2);
  Eigen::Vector2d undistorted, distorted;
  for (undistorted[1] = 0; undistorted[1] < scale*m_undistorted_size[1]; undistorted[1]++) {
    for (undistorted[0] = 0; undistorted[0] < scale*m_undistorted_size[0]; undistorted[0]++) {
      Convert<UNDISTORTED, DISTORTED>(undistorted/scale, &distorted);
      remap_map->at<cv::Vec2f>(undistorted[1], undistorted[0])[0] = scale*distorted[0];
      remap_map->at<cv::Vec2f>(undistorted[1], undistorted[0])[1] = scale*distorted[1];
    }
  }
}


namespace camera {

  // Conversion function helpers
#define DEFINE_CONVERSION(TYPEA, TYPEB) \
  template <> \
  void camera::CameraParameters::Convert<TYPEA, TYPEB>(Eigen::Vector2d const& input, Eigen::Vector2d *output) const

  DEFINE_CONVERSION(RAW, DISTORTED) {
    *output = input - m_crop_offset.cast<double>();
  }
  DEFINE_CONVERSION(DISTORTED, RAW) {
    *output = input + m_crop_offset.cast<double>();
  }
  DEFINE_CONVERSION(UNDISTORTED_C, DISTORTED_C) {
    DistortCentered(input, output);
  }
  DEFINE_CONVERSION(DISTORTED_C, UNDISTORTED_C) {
    UndistortCentered(input, output);
  }
  DEFINE_CONVERSION(UNDISTORTED, UNDISTORTED_C) {
    *output = input - m_undistorted_half_size;
  }
  DEFINE_CONVERSION(UNDISTORTED_C, UNDISTORTED) {
    *output = input + m_undistorted_half_size;
  }
  DEFINE_CONVERSION(DISTORTED, UNDISTORTED) {
    Convert<DISTORTED_C, UNDISTORTED_C>(input - m_distorted_half_size, output);
    *output += m_undistorted_half_size;
  }
  DEFINE_CONVERSION(UNDISTORTED, DISTORTED) {
    Eigen::Vector2d centered_output;
    Convert<UNDISTORTED, UNDISTORTED_C>(input, output);
    Convert<UNDISTORTED_C, DISTORTED_C>(*output, &centered_output);
    *output = centered_output + m_distorted_half_size;
  }
  DEFINE_CONVERSION(DISTORTED, UNDISTORTED_C) {
    Convert<DISTORTED_C, UNDISTORTED_C>(input - m_distorted_half_size, output);
  }
  DEFINE_CONVERSION(UNDISTORTED_C, DISTORTED) {
    Convert<UNDISTORTED_C, DISTORTED_C>(input, output);
    *output += m_distorted_half_size;
  }

#undef DEFINE_CONVERSION

  // Helper functions to give the intrinsic matrix
#define DEFINE_INTRINSIC(TYPE) \
  template <> \
  Eigen::Matrix3d camera::CameraParameters::GetIntrinsicMatrix<TYPE>() const

  DEFINE_INTRINSIC(RAW) {
    Eigen::Matrix3d k = m_focal_length.homogeneous().asDiagonal();
    k.block<2, 1>(0, 2) = m_optical_offset + m_crop_offset.cast<double>();
    return k;
  }
  DEFINE_INTRINSIC(DISTORTED) {
    Eigen::Matrix3d k = m_focal_length.homogeneous().asDiagonal();
    k.block<2, 1>(0, 2) = m_optical_offset;
    return k;
  }
  DEFINE_INTRINSIC(DISTORTED_C) {
    Eigen::Matrix3d k = m_focal_length.homogeneous().asDiagonal();
    k.block<2, 1>(0, 2) = m_optical_offset - m_distorted_half_size;
    return k;
  }
  DEFINE_INTRINSIC(UNDISTORTED) {
    Eigen::Matrix3d k = m_focal_length.homogeneous().asDiagonal();
    k.block<2, 1>(0, 2) = m_undistorted_half_size;
    return k;
  }
  DEFINE_INTRINSIC(UNDISTORTED_C) {
    Eigen::Matrix3d k = m_focal_length.homogeneous().asDiagonal();
    return k;
  }

#undef DEFINE_INTRINSIC

}  // end namespace camera

