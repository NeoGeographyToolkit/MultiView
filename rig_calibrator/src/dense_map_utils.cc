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

#include <glog/logging.h>

#include <rig_calibrator/system_utils.h>
#include <rig_calibrator/dense_map_utils.h>
#include <rig_calibrator/camera_image.h>

#include <boost/filesystem.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
// TODO(oalexan1): This file needs to be broken up

namespace dense_map {

// Two minor and local utility functions
std::string print_vec(double a) {
  char st[256];
  snprintf(st, sizeof(st), "%7.4f", a);
  return std::string(st);
}
std::string print_vec(Eigen::Vector3d a) {
  char st[256];
  snprintf(st, sizeof(st), "%7.4f %7.4f %7.4f", a[0], a[1], a[2]);
  return std::string(st);
}

// A little function to replace separators with space  
void replace_separators_with_space(std::string & str) {
  std::string sep = ":, \t\r\n";
  for (size_t it = 0; it < sep.size(); it++) 
    std::replace(str.begin(), str.end(), sep[it], ' ');
}
  
// A function to parse a string like
// 'cam1:focal_length,optical_center,distortion cam2:focal_length' and
// extract the intrinsics to float. Separators can be space, comma,
// colon.
void parse_intrinsics_to_float(std::string const& intrinsics_to_float_str,
                               std::vector<std::string> const& cam_names,
                               std::vector<std::set<std::string>>& intrinsics_to_float) {
  // Wipe the output
  intrinsics_to_float.clear();

  // Replace all separators with spaces
  std::string sep = ":, \t\r\n";

  std::string input_str = intrinsics_to_float_str; // so we can edit it
  replace_separators_with_space(input_str);

  std::istringstream iss(input_str);
  std::string curr_cam = "";
  std::string val;
  
  // Temporary map of sets for collection. This will ensure variable order
  // of inputs is supported.
  std::map<std::string, std::set<std::string>> local_map;
  while (iss >> val) {
    // See if this is a camera name
    bool have_cam_name = false;
    for (size_t it = 0; it < cam_names.size(); it++) {
      if (val == cam_names[it]) {
        curr_cam = val;
        have_cam_name = true;
        break;
      }
    }

    if (have_cam_name) // recorded the camera name
      continue;
    
    if (val != "focal_length" && val != "optical_center" && val != "distortion")
      LOG(FATAL) << "Unexpected value when parsing intrinsics to float: " << val << "\n";

    if (curr_cam == "") 
      LOG(FATAL) << "Incorrectly set option for floating intrinsics.\n";

    local_map[curr_cam].insert(val);
  }

  // Export this
  intrinsics_to_float.resize(cam_names.size());
  for (size_t it = 0; it < cam_names.size(); it++)
    intrinsics_to_float[it] = local_map[cam_names[it]];
}

// A  function to split a string like 'haz_cam sci_cam' into
// its two constituents and validate against the list of known cameras.
void parse_camera_names(std::vector<std::string> const& cam_names,
                                              std::string const&
                                              depth_to_image_transforms_to_float_str,
                                              std::set<std::string>&
                                              depth_to_image_transforms_to_float) {
  // Wipe the output
  depth_to_image_transforms_to_float.clear();

  std::string input_str = depth_to_image_transforms_to_float_str; // so we can edit it
  replace_separators_with_space(input_str);
  
  std::istringstream iss(input_str);
  std::string curr_cam = "";
  std::string val;
  
  while (iss >> val) {
    bool have_cam_name = false;
    for (size_t it = 0; it < cam_names.size(); it++) {
      if (val == cam_names[it]) {
        have_cam_name = true;
        break;
      }
    }
    
    if (!have_cam_name) 
      LOG(FATAL) << "Error: A specified sensor name is not among the known sensors.\n";
    
    depth_to_image_transforms_to_float.insert(val);
  }

  return;
}  
  
// A  function to split a string like 'haz_cam sci_cam' into
// its two constituents and validate against the list of known cameras.
// Do not allow to float the transform from ref cam to itself, as that
// is the identity.
void parse_rig_transforms_to_float(std::vector<std::string> const& cam_names,
                                   int ref_cam_type,
                                   std::string const& rig_transforms_to_float_str,
                                   std::set<std::string>& rig_transforms_to_float) {

  // Reuse earlier logic
  parse_camera_names(cam_names, 
                     rig_transforms_to_float_str,  
                     rig_transforms_to_float);

  // Additional sanity check
  for (auto it = rig_transforms_to_float.begin(); it != rig_transforms_to_float.end(); it++)
    if (*it == cam_names[ref_cam_type]) 
      LOG(FATAL) << "Cannot float the rig transform from reference camera to itself.\n";

  return;
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

// Read a 4x4 pose matrix of doubles from disk
void readPoseMatrix(cv::Mat& pose, std::string const& filename) {
  pose = cv::Mat::zeros(4, 4, CV_64F);
  std::ifstream ifs(filename.c_str(), std::ifstream::in);
  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 4; col++) {
      double val;
      if (!(ifs >> val)) LOG(FATAL) << "Could not read a 4x4 matrix from: " << filename;
      pose.at<double>(row, col) = val;
    }
  }
}

// Read an affine matrix with double values
bool readAffine(Eigen::Affine3d& T, std::string const& filename) {
  Eigen::MatrixXd M(4, 4);

  std::ifstream ifs(filename.c_str(), std::ifstream::in);
  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 4; col++) {
      double val;
      if (!(ifs >> val)) return false;

      M(row, col) = val;
    }
  }

  T.linear() = M.block<3, 3>(0, 0);
  T.translation() = M.block<3, 1>(0, 3);

  return true;
}

// Write a matrix with double values
void writeMatrix(Eigen::MatrixXd const& M, std::string const& filename) {
  std::cout << "Writing: " << filename << std::endl;
  std::ofstream ofs(filename.c_str());
  ofs.precision(17);
  ofs << M << "\n";
  ofs.close();
}

// Save a file with x, y, z rows if point_size is 3, and also a color
// if point_size is 4.
void writeCloud(std::vector<float> const& points, size_t point_size, std::string const& filename) {
  size_t num_points = points.size() / point_size;
  if (point_size * num_points != points.size()) LOG(FATAL) << "Book-keeping failure.";

  std::cout << "Writing: " << filename << "\n";
  std::ofstream fh(filename.c_str());
  fh.precision(17);
  for (size_t it = 0; it < num_points; it++) {
    for (size_t ch = 0; ch < point_size; ch++) {
      fh << points[point_size * it + ch];
      if (ch + 1 < point_size)
        fh << " ";
      else
        fh << "\n";
    }
  }
  fh.close();
}

// Return the type of an opencv matrix
std::string matType(cv::Mat const& mat) {
  int type = mat.type();
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}

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

// Implement some heuristic to find the maximum rotation angle that can result
// from applying the given transform. It is assumed that the transform is not
// too different from the identity.
double maxRotationAngle(Eigen::Affine3d const& T) {
  Eigen::Vector3d angles = T.linear().eulerAngles(0, 1, 2);

  // Angles close to +/-pi can result even if the matrix is close to identity
  for (size_t it = 0; it < 3; it++)
    angles[it] = std::min(std::abs(angles[it]), std::abs(M_PI - std::abs(angles[it])));
  double angle_norm = (180.0 / M_PI) * angles.norm();
  return angle_norm;
}

void StampedPoseStorage::addPose(Eigen::Affine3d const& pose, double timestamp) {
  int bin_index = floor(timestamp);
  m_poses[bin_index][timestamp] = pose;
}

bool StampedPoseStorage::interpPose(double input_timestamp, double max_gap, Eigen::Affine3d& out_pose) const {
  bool is_success = false;

  if (m_poses.empty()) return is_success;

  // Look for the nearest pose with timestamp <= input_timestamp.
  double low_timestamp = -1.0;
  Eigen::Affine3d low_pose;
  // Traverse the bins in decreasing order of bin key.
  for (int bin_iter = floor(input_timestamp); bin_iter >= m_poses.begin()->first; bin_iter--) {
    auto bin_ptr = m_poses.find(bin_iter);
    if (bin_ptr == m_poses.end()) continue;  // empty bin

    // Found a bin. Study it in decreasing order of timestamps.
    auto& bin = bin_ptr->second;
    for (auto it = bin.rbegin(); it != bin.rend(); it++) {
      double timestamp = it->first;
      if (timestamp <= input_timestamp) {
        low_timestamp = timestamp;
        low_pose = it->second;
        is_success = true;
        break;
      }
    }
    if (is_success) break;
  }

  if (!is_success) return false;  // Failed

  // Found the lower bound. Now go forward in time. Here the logic is
  // the reverse of the above.
  is_success = false;
  double high_timestamp = -1.0;
  Eigen::Affine3d high_pose;
  for (int bin_iter = floor(input_timestamp); bin_iter <= m_poses.rbegin()->first; bin_iter++) {
    auto bin_ptr = m_poses.find(bin_iter);
    if (bin_ptr == m_poses.end()) continue;  // empty bin

    // Found a bin. Study it in increasing order of timestamps.
    auto& bin = bin_ptr->second;
    for (auto it = bin.begin(); it != bin.end(); it++) {
      double timestamp = it->first;
      if (timestamp >= input_timestamp) {
        high_timestamp = timestamp;
        high_pose = it->second;
        is_success = true;
        break;
      }
    }
    if (is_success) break;
  }

  if (!is_success || high_timestamp - low_timestamp > max_gap) {
    return false;  // Failed
  }

  if (!(low_timestamp <= input_timestamp && input_timestamp <= high_timestamp))
    LOG(FATAL) << "Book-keeping failure in pose interpolation.";

  double alpha = (input_timestamp - low_timestamp) / (high_timestamp - low_timestamp);
  if (high_timestamp == low_timestamp) alpha = 0.0;  // handle division by zero

  out_pose = dense_map::linearInterp(alpha, low_pose, high_pose);

  return is_success;
}

void StampedPoseStorage::clear() { m_poses.clear(); }

bool StampedPoseStorage::empty() const { return m_poses.empty(); }

// Compute the azimuth and elevation for a (normal) vector
void normalToAzimuthAndElevation(Eigen::Vector3d const& normal, double& azimuth, double& elevation) {
  if (normal.x() == 0 && normal.y() == 0) {
    azimuth = 0.0;
    if (normal.z() >= 0.0)
      elevation = M_PI / 2.0;
    else
      elevation = -M_PI / 2.0;
  } else {
    azimuth = atan2(normal.y(), normal.x());
    elevation = atan2(normal.z(), Eigen::Vector2d(normal.x(), normal.y()).norm());
  }
}

// Compute a normal vector based on the azimuth and elevation angles
void azimuthAndElevationToNormal(Eigen::Vector3d& normal, double azimuth, double elevation) {
  double ca = cos(azimuth), sa = sin(azimuth);
  double ce = cos(elevation), se = sin(elevation);
  normal = Eigen::Vector3d(ca * ce, sa * ce, se);
}

// Snap the normal to the plane (and the plane itself) to make
// all angles multiple of 45 degrees with the coordinate axes.
void snapPlaneNormal(Eigen::Vector3d& plane_normal) {
  double azimuth, elevation;
  normalToAzimuthAndElevation(plane_normal, azimuth, elevation);

  // Snap to multiple of 45 degrees
  double radian45 = M_PI / 4.0;
  azimuth = radian45 * round(azimuth / radian45);
  elevation = radian45 * round(elevation / radian45);

  azimuthAndElevationToNormal(plane_normal, azimuth, elevation);
}

// Find the best fitting plane to a set of points
void bestFitPlane(const std::vector<Eigen::Vector3d>& points, Eigen::Vector3d& centroid,
                  Eigen::Vector3d& plane_normal) {
  // Copy coordinates to  matrix in Eigen format
  size_t num_points = points.size();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coord(3, num_points);

  for (size_t i = 0; i < num_points; i++) coord.col(i) = points[i];

  // calculate centroid
  centroid = Eigen::Vector3d(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());

  // subtract centroid
  for (size_t it = 0; it < 3; it++) coord.row(it).array() -= centroid(it);

  // We only need the left-singular matrix here
  // https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
  auto svd = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

  plane_normal = svd.matrixU().rightCols<1>();
}

// Extract from a string of the form someDir/1234.5678.jpg the number 123.456.
double fileNameToTimestamp(std::string const& file_name) {
  size_t beg = file_name.rfind("/");
  size_t end = file_name.rfind(".");
  if (beg == std::string::npos || end == std::string::npos || beg > end) {
    std::cout << "Could not parse file name: " + file_name;
    exit(1);
  }

  std::string frameStr = file_name.substr(beg + 1, end - beg - 1);
  return atof(frameStr.c_str());
}

// Minor utilities for converting values to a string below

// Convert a string of values separated by spaces to a vector of doubles.
std::vector<double> string_to_vector(std::string const& str) {
  std::istringstream iss(str);
  std::vector<double> vals;
  double val;
  while (iss >> val)
    vals.push_back(val);
  return vals;
}

void readCameraPoses(std::string const& filename,
                     std::map<double, double>& haz_depth_to_image_timestamps,
                     std::map<std::string, std::map<double, Eigen::Affine3d> >&
                     world_to_cam_poses) {
  haz_depth_to_image_timestamps.clear();
  world_to_cam_poses.clear();

  std::ifstream ifs(filename.c_str());
  std::string line;
  while (std::getline(ifs, line)) {
    std::istringstream is(line);

    std::string str;
    if (!(is >> str)) continue;

    if (str == "nav_cam" || str == "sci_cam" || str == "haz_cam") {
      double timestamp;
      if (!(is >> timestamp)) continue;

      Eigen::MatrixXd M(4, 4);
      for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
          if (!(is >> M(row, col))) LOG(FATAL) << "Failure reading: " << filename;
        }
      }

      world_to_cam_poses[str][timestamp].matrix() = M;

    } else if (str == "haz_depth_to_image") {
      double depth_time, image_time;
      if (!(is >> depth_time >> image_time)) LOG(FATAL) << "Failure reading: " << filename;

      haz_depth_to_image_timestamps[depth_time] = image_time;
    }
  }

  ifs.close();
}

// Gamma correction for x between 0 and 1.
// https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
double gamma(double x) {
  // return pow(x, 1.0/2.6);

  if (x <= 0.0031308) return 12.92 * x;

  return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

double inv_gamma(double x) {
  // return pow(x, 2.6);

  if (x <= 0.04045) return x / 12.92;

  return pow((x + 0.055) / (1.055), 2.4);
}

// Apply the inverse gamma transform to images, multiply them by
// max_iso_times_exposure/ISO/exposure_time to adjust for
// lightning differences, then apply the gamma transform back.
void exposureCorrection(double max_iso_times_exposure, double iso, double exposure,
                        cv::Mat const& input_image, cv::Mat& output_image) {
  double scale = max_iso_times_exposure / iso / exposure;

  // Make an image of the same type
  input_image.copyTo(output_image);

  // Apply the inverse gamma correction, multiply by scale,
  // and apply the correction back
#pragma omp parallel for
  for (int row = 0; row < input_image.rows; row++) {
    for (int col = 0; col < input_image.cols; col++) {
      cv::Vec3b b = input_image.at<cv::Vec3b>(row, col);

      cv::Vec3b c;
      for (int color = 0; color < 3; color++) {
        double x = 255.0 * gamma(inv_gamma(static_cast<double>(b[color]) / 255.0) * scale);
        c[color] = std::min(round(x), 255.0);
      }
      output_image.at<cv::Vec3b>(row, col) = c;
    }
  }
}

// Scale an image to correct for lightning variations by taking into
// account that JPEG images have gamma correction applied to them.
// See https://en.wikipedia.org/wiki/Gamma_correction.
void scaleImage(double max_iso_times_exposure, double iso, double exposure, cv::Mat const& input_image,
                cv::Mat& output_image) {
  double scale = pow(max_iso_times_exposure / iso / exposure, 1.0 / 2.2);
  int same_type = -1;
  double offset = 0.0;
  input_image.convertTo(output_image, same_type, scale, offset);
}

// Given two bounds, pick two timestamps within these bounds, the one
// closest to the left bound and the one to the right bound. Take into
// account that the timestamps may need to have an offset added to
// them. Assume that the input timestamps are sorted in increasing order.
// TODO(oalexan1): May have to add a constraint to only pick
// a timestamp if not further from the bound than a given value.
void pickTimestampsInBounds(std::vector<double> const& timestamps,
                            double left_bound, double right_bound, double offset,
                            std::vector<double>& out_timestamps) {
  out_timestamps.clear();

  // Start by simply collecting all timestamps between the given
  // bounds. Much easier to understand than if doing something more
  // fancy.
  std::vector<double> local_timestamps;
  for (size_t it = 0; it < timestamps.size(); it++) {
    double timestamp = timestamps[it];
    if (timestamp + offset >= left_bound && timestamp + offset < right_bound) {
      local_timestamps.push_back(timestamp);
    }
  }

  if (local_timestamps.size() < 1) {
    // Nothing to pick
    return;
  }

  if (local_timestamps.size() == 1) {
    // Only one is present
    out_timestamps.push_back(local_timestamps[0]);
    return;
  }

  // Add the ones at the ends
  out_timestamps.push_back(local_timestamps[0]);
  out_timestamps.push_back(local_timestamps.back());

  return;
}

// A debug utility for saving a camera in a format ASP understands.
// Need to expose the sci cam intrinsics.
void saveTsaiCamera(Eigen::MatrixXd const& desired_cam_to_world_trans,
                        std::string const& output_dir,
                      double curr_time, std::string const& suffix) {
  char filename_buffer[1000];
  auto T = desired_cam_to_world_trans;
  double shift = 6378137.0;  // planet radius, pretend this is a satellite camera
  snprintf(filename_buffer, sizeof(filename_buffer), "%s/%10.7f_%s.tsai",
           output_dir.c_str(), curr_time,
           suffix.c_str());
  std::cout << "Writing: " << filename_buffer << std::endl;
  std::ofstream ofs(filename_buffer);
  ofs.precision(18);
  ofs << "VERSION_3\n";
  ofs << "fu = 1138.4943\n";
  ofs << "fv = 1138.4943\n";
  ofs << "cu = 680.36447\n";
  ofs << "cv = 534.00133\n";
  ofs << "u_direction = 1 0 0\n";
  ofs << "v_direction = 0 1 0\n";
  ofs << "w_direction = 0 0 1\n";
  ofs << "C = " << T(0, 3) + shift << ' ' << T(1, 3) << ' ' << T(2, 3) << "\n";
  ofs << "R = " << T(0, 0) << ' ' << T(0, 1) << ' ' << T(0, 2) << ' ' << T(1, 0) << ' ' << T(1, 1) << ' ' << T(1, 2)
      << ' ' << T(2, 0) << ' ' << T(2, 1) << ' ' << T(2, 2) << "\n";
  ofs << "pitch = 1\n";
  ofs << "NULL\n";
  ofs.close();
}

// Write an image with 3 floats per pixel. OpenCV's imwrite() cannot do that.
void saveXyzImage(std::string const& filename, cv::Mat const& img) {
  if (img.depth() != CV_32F)
    LOG(FATAL) << "Expecting an image with float values\n";
  if (img.channels() != 3) LOG(FATAL) << "Expecting 3 channels.\n";

  std::ofstream f;
  f.open(filename.c_str(), std::ios::binary | std::ios::out);
  if (!f.is_open()) LOG(FATAL) << "Cannot open file for writing: " << filename << "\n";

  // Assign these to explicit variables so we know their type and size in bytes
  int rows = img.rows, cols = img.cols, channels = img.channels();

  // TODO(oalexan1): Avoid C-style cast. Test if
  // reinterpret_cast<char*> does the same thing.
  f.write((char*)(&rows), sizeof(rows));         // NOLINT
  f.write((char*)(&cols), sizeof(cols));         // NOLINT
  f.write((char*)(&channels), sizeof(channels)); // NOLINT

  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      cv::Vec3f const& P = img.at<cv::Vec3f>(row, col);  // alias
      // TODO(oalexan1): See if using reinterpret_cast<char*> does the same
      // thing.
      for (int c = 0; c < channels; c++)
        f.write((char*)(&P[c]), sizeof(P[c])); // NOLINT
    }
  }

  return;
}

// Read an image with 3 floats per pixel. OpenCV's imread() cannot do that.
void readXyzImage(std::string const& filename, cv::Mat & img) {
  std::ifstream f;
  f.open(filename.c_str(), std::ios::binary | std::ios::in);
  if (!f.is_open()) LOG(FATAL) << "Cannot open file for reading: " << filename << "\n";

  int rows, cols, channels;
  f.read((char*)(&rows), sizeof(rows));         // NOLINT
  f.read((char*)(&cols), sizeof(cols));         // NOLINT
  f.read((char*)(&channels), sizeof(channels)); // NOLINT

  img = cv::Mat::zeros(rows, cols, CV_32FC3);

  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      cv::Vec3f P;
      // TODO(oalexan1): See if using reinterpret_cast<char*> does the same
      // thing.
      for (int c = 0; c < channels; c++)
        f.read((char*)(&P[c]), sizeof(P[c])); // NOLINT
      img.at<cv::Vec3f>(row, col) = P;
    }
  }

  return;
}

// Create the image and depth cloud file names
void genImageAndDepthFileNames(  // Inputs
  std::vector<dense_map::cameraImage> const& cams, std::vector<std::string> const& cam_names,
  std::string const& out_dir,
  // Outputs
  std::vector<std::string>& image_files, std::vector<std::string>& depth_files) {
  // Wipe the output
  image_files.clear();

  char buffer[1000];
  for (size_t it = 0; it < cams.size(); it++) {
    // Use the timestamp as image and depth cloud names, unless those
    // names already exist. Note that several sensors can produce
    // images with same timestamp, so also keep track of camera name.
    std::string cam_dir = out_dir + "/" + cam_names[cams[it].camera_type];
    std::string image_name;
    // Will use .tif images, as those are lossless
    snprintf(buffer, sizeof(buffer), "%10.7f.tif", cams[it].timestamp);
    image_name = buffer;
    image_name = cam_dir + "/" + image_name;
    image_files.push_back(image_name);

    // We count having on the depth file having precisely the same
    // name as the image file, with only a different extension
    std::string depth_name;
    snprintf(buffer, sizeof(buffer), "%10.7f.pc", cams[it].timestamp);
    depth_name = buffer;
    depth_name = cam_dir + "/" + depth_name;
    depth_files.push_back(depth_name);
  }
}

// Save images and depth clouds to disk
void saveImagesAndDepthClouds(std::vector<dense_map::cameraImage> const& cams,
                              std::vector<std::string> const& image_files,
                              std::vector<std::string> const& depth_files) {
  if (cams.size() != image_files.size())
    LOG(FATAL) << "There must be as many image files as cameras.\n";

  for (size_t it = 0; it < cams.size(); it++) {
    // Create the directory for the image file
    std::string out_dir = boost::filesystem::path(image_files[it]).parent_path().string();
    if (out_dir != "") dense_map::createDir(out_dir);

    std::cout << "Writing: " << image_files[it] << std::endl;
    cv::imwrite(image_files[it], cams[it].image);

    if (cams[it].depth_cloud.cols > 0 && cams[it].depth_cloud.rows > 0) {
      // Create the directory for the depth cloud
      out_dir = boost::filesystem::path(depth_files[it]).parent_path().string();
      if (out_dir != "") dense_map::createDir(out_dir);

      std::cout << "Writing: " << depth_files[it] << std::endl;
      dense_map::saveXyzImage(depth_files[it], cams[it].depth_cloud);
    }
  }

  return;
}

// Find an image at the given timestamp or right after it. We assume
// that during repeated calls to this function we always travel
// forward in time, and we keep track of where we are in the bag using
// the variable start_pos that we update as we go.
bool lookupImage(double desired_time, std::vector<ImageMessage> const& msgs,
                 cv::Mat& image, int& start_pos, double& found_time) {
  // Initialize the outputs.
  image = cv::Mat();
  found_time = -1.0;

  int num_msgs = msgs.size();
  double prev_image_time = -1.0;

  for (int local_pos = start_pos; local_pos < num_msgs; local_pos++) {
    start_pos = local_pos;  // save this for exporting

    found_time = msgs[local_pos].timestamp;

    // Sanity check: We must always travel forward in time
    if (found_time < prev_image_time) {
      LOG(FATAL) << "Found images not in chronological order.\n"
                 << std::fixed << std::setprecision(17)
                 << "Times in wrong order: " << prev_image_time << ' ' << found_time << ".\n";
      continue;
    }

    prev_image_time = found_time;

    if (found_time >= desired_time) {
      // Found the desired data. Do a deep copy, to not depend on the original structure.
      msgs[local_pos].image.copyTo(image);
      return true;
    }
  }
  return false;
}

}  // end namespace dense_map
