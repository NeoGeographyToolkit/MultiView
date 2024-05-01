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

#include <rig_calibrator/basic_algs.h>

#include <glog/logging.h>
#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace dense_map {

// Find the file extension (the part after the last dot).  
std::string file_extension(std::string const& file) {
  size_t it = file.find_last_of(".");

  if (it == std::string::npos || it + 1 >= file.size())
    return "";

  return file.substr(it + 1);
}

// Convert keypoints to Eigen format
void vec2eigen(std::vector<std::pair<float, float>> const& vec,
               Eigen::Matrix2Xd & mat) {

  mat = Eigen::MatrixXd(2, vec.size());
  for (size_t it = 0; it < vec.size(); it++) {
    mat.col(it) = Eigen::Vector2d(vec[it].first, vec[it].second);
  }
}

// Convert keypoints from Eigen format
void eigen2vec(Eigen::Matrix2Xd const& mat,
               std::vector<std::pair<float, float>> & vec) {

  vec.clear();
  vec.resize(mat.cols());
  for (size_t it = 0; it < vec.size(); it++)
    vec.at(it) = std::make_pair<float, float>(mat(0, it), mat(1, it));
}

// Read a vector of strings from a file, with spaces and newlines
// acting as separators.  Store them in a set.
void readList(std::string const& file, std::set<std::string> & list) {
  list.clear();
  std::ifstream fh(file);
  std::string val;
  while (fh >> val)
    list.insert(val);
  fh.close();
}

// The cam name is the subdir having the images.
// Example: mydir/nav_cam/file.jpg has nav_cam as the cam name.
std::string camName(std::string const& image_file) {
  return fs::path(image_file).parent_path().filename().string();
}

// Find cam type based on cam name
void camTypeFromName(std::string const& cam_name,
                     std::vector<std::string> const& cam_names,
                     int& cam_type) {
  cam_type = 0; // initialize
  for (size_t cam_it = 0; cam_it < cam_names.size(); cam_it++) {
    if (cam_names[cam_it] == cam_name) {
      cam_type = cam_it;
      return;
    }
  }

  LOG(FATAL) << "Could not determine the sensor type for: " << cam_name << "\n";
}
  
// Given a file with name <path to>/<cam name>/<digits>.<digits>.jpg,
// find the cam name, then look up the cam type. Also find the timestamp.
void findCamTypeAndTimestamp(std::string const& image_file,
                             std::vector<std::string> const& cam_names,
                             // Outputs
                             int    & cam_type,
                             double & timestamp) {

  // Initialize the outputs
  cam_type = 0;
  timestamp = 0.0;
  
  // The cam name is the subdir having the images
  std::string cam_name = camName(image_file);
    
  std::string basename = fs::path(image_file).filename().string();

  // Remove anything after <digits>.<digits> 
  bool have_dot = false;
  std::string timestamp_str;
  for (size_t it = 0; it < basename.size(); it++) {

    if (basename[it] == '.') {
      if (have_dot) 
        break; // We have seen a dot already, ignore the rest
      have_dot = true;
      timestamp_str += basename[it];
      continue;
    }

    if (basename[it] < '0' || basename[it] > '9') 
      break; // Not a digit, ignore the rest
    
    timestamp_str += basename[it];
  }

  if (timestamp_str.empty())
    LOG(FATAL) << "Image name (without directory) must have digits as part of "
               << "their name, which will be converted to a timestamp. Got: "
               << basename << "\n";

  // Having the timestamp extracted from the image name is convenient though it
  // requires some care. This is well-documented.
  timestamp = atof(timestamp_str.c_str());
  
  // Infer cam type from cam name
  camTypeFromName(cam_name, cam_names, cam_type);
}

// Replace .<extension> with <suffix>  
std::string changeFileSuffix(std::string filename, std::string new_suffix) {
  // Find the last '.' character in the filename
  size_t last_dot = filename.find_last_of(".");
  if (last_dot == std::string::npos) {
    // No dot found, return original filename
    return filename;
  } else {
    // Replace extension with new suffix
    std::string new_filename = filename.substr(0, last_dot) + new_suffix;
    return new_filename;
  }
}  
  
}  // end namespace dense_map
