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
#include <rig_calibrator/transform_utils.h>
#include <rig_calibrator/system_utils.h>
#include <iostream>
#include <set>
#include <fstream>
#include <iomanip>

namespace dense_map {

void RigSet::validate() const {
  
  if (cam_set.empty()) 
    LOG(FATAL) << "Found an empty set of rigs.\n";
  
  size_t num_cams = 0;
  std::set<std::string> all_cams; // checks for duplicates
  for (size_t rig_it = 0; rig_it < cam_set.size(); rig_it++) {
    if (cam_set[rig_it].empty()) 
      LOG(FATAL) << "Found a rig with no cams.\n";
    
    num_cams += cam_set[rig_it].size();
    for (size_t cam_it = 0; cam_it < cam_set[rig_it].size(); cam_it++)
      all_cams.insert(cam_set[rig_it][cam_it]);
  }
  
  if (num_cams != all_cams.size() || num_cams != cam_names.size())
    LOG(FATAL) << "Found a duplicate sensor name in the rig set.\n";
  
  if (num_cams != ref_to_cam_trans.size()) 
    LOG(FATAL) << "Number of sensors is not equal to number of ref-to-cam transforms.\n";
  
  if (num_cams != depth_to_image.size()) 
    LOG(FATAL) << "Number of sensors is not equal to number of depth-to-image transforms.\n";
  
  if (num_cams != ref_to_cam_timestamp_offsets.size()) 
    LOG(FATAL) << "Number of sensors is not equal to number of ref-to-cam timestamp offsets.\n";
  
  if (num_cams != cam_params.size()) 
    LOG(FATAL) << "Number of sensors is not equal to number of camera models.\n";
  
  for (size_t cam_it = 0; cam_it < cam_names.size(); cam_it++) {
    if (isRefSensor(cam_names[cam_it]) && ref_to_cam_timestamp_offsets[cam_it] != 0) 
      LOG(FATAL) << "The timestamp offsets for the reference sensors must be always 0.\n";
  }
}
  
// A ref sensor is the first sensor on each rig
bool RigSet::isRefSensor(std::string const& cam_name) const {
  for (size_t rig_it = 0; rig_it < cam_set.size(); rig_it++) 
    if (cam_set[rig_it][0] == cam_name) 
      return true;
  return false;
}

// Return the id of the rig given the index of the sensor
// in cam_names.
int RigSet::rigId(int cam_id) const {
  if (cam_id < 0 || cam_id >= cam_names.size()) 
    LOG(FATAL) << "Out of bounds sensor id.\n";
  
  std::string cam_name = cam_names[cam_id];
  
  for (size_t rig_it = 0; rig_it < cam_set.size(); rig_it++) {
    for (size_t cam_it = 0; cam_it < cam_set[rig_it].size(); cam_it++) {
      if (cam_set[rig_it][cam_it] == cam_name) {
        return rig_it;
      }
    }
  }

  // Should not arrive here
  LOG(FATAL) << "Could not look up in the rig the sensor: " << cam_name << "\n";
  return -1;
}

// The name of the ref sensor for the rig having the given sensor id
std::string RigSet::refSensor(int cam_id) const {
  return cam_set[rigId(cam_id)][0];
}
  
// Index in the list of sensors of the sensor with given name
int RigSet::sensorIndex(std::string const& sensor_name) const {
  auto it = std::find(cam_names.begin(), cam_names.end(), sensor_name);
  if (it == cam_names.end()) 
    LOG(FATAL) << "Could not find sensor in rig. That is unexpected. Offending sensor: "
               << sensor_name << ".\n";
  return it - cam_names.begin();
}
  
// Create a rig set having a single rig  
RigSet RigSet::subRig(int rig_id) const {

  if (rig_id < 0 || rig_id >= cam_set.size()) 
    LOG(FATAL) << "Out of range in rig set.\n";

  RigSet sub_rig;
  sub_rig.cam_set.push_back(cam_set[rig_id]);

  // Add the relevant portion of each rig member
  for (size_t subrig_it = 0; subrig_it < cam_set[rig_id].size(); subrig_it++) {
    
    std::string sensor_name = cam_set[rig_id][subrig_it];
    int rig_index = sensorIndex(sensor_name);

    sub_rig.cam_names.push_back(cam_names[rig_index]);
    sub_rig.ref_to_cam_trans.push_back(ref_to_cam_trans[rig_index]);
    sub_rig.depth_to_image.push_back(depth_to_image[rig_index]);
    sub_rig.ref_to_cam_timestamp_offsets.push_back(ref_to_cam_timestamp_offsets[rig_index]);
    sub_rig.cam_params.push_back(cam_params[rig_index]);
  }
  sub_rig.validate();

  return sub_rig;
}
  
const std::string FISHEYE_DISTORTION = "fisheye";
const std::string RADTAN_DISTORTION  = "radtan";
const std::string RPC_DISTORTION     = "rpc";
const std::string NO_DISTORION       = "no_distortion";

// Save the optimized rig configuration
void writeRigConfig(std::string const& out_dir, bool model_rig, RigSet const& R) {

  R.validate();
  
  dense_map::createDir(out_dir);
  std::string rig_config = out_dir + "/rig_config.txt";
  std::cout << "Writing: " << rig_config << std::endl;

  std::ofstream f;
  f.open(rig_config.c_str(), std::ios::binary | std::ios::out);
  if (!f.is_open()) LOG(FATAL) << "Cannot open file for writing: " << rig_config << "\n";
  f.precision(17);

  for (size_t cam_type = 0; cam_type < R.cam_params.size(); cam_type++) {

    if (R.isRefSensor(R.cam_names[cam_type])) {
      if (cam_type > 0)
        f << "\n"; // add an empty line for clarity
      
      f << "ref_sensor_name: " << R.cam_names[cam_type] << "\n";
    }
    
    f << "\n";
    f << "sensor_name: "  << R.cam_names[cam_type] << "\n";
    f << "focal_length: " << R.cam_params[cam_type].GetFocalLength() << "\n";

    Eigen::Vector2d c = R.cam_params[cam_type].GetOpticalOffset();
    f << "optical_center: " << c[0] << " " << c[1] << "\n";

    Eigen::VectorXd D = R.cam_params[cam_type].GetDistortion();

    f << "distortion_coeffs: ";
    for (int d = 0; d < D.size(); d++) {
      f << D[d];
      if (d + 1 < D.size()) f << " ";
    }
    f << "\n";

    if (D.size() == 0) f << "distortion_type: " << dense_map::NO_DISTORION << "\n";
    else if (D.size() == 1)
      f << "distortion_type: " << dense_map::FISHEYE_DISTORTION << "\n";
    else if (D.size() >= 4 && D.size() <= 5)
      f << "distortion_type: " << dense_map::RADTAN_DISTORTION << "\n";
    else if (D.size() > 5)
      f << "distortion_type: " << dense_map::RPC_DISTORTION << "\n";
    else
      LOG(FATAL) << "Expecting 0, 1, 4, 5, or more distortion coefficients. Got: "
                 << D.size() << ".\n";

    Eigen::Vector2i image_size = R.cam_params[cam_type].GetDistortedSize();
    f << "image_size: " << image_size[0] << ' ' << image_size[1] << "\n";

    Eigen::Vector2i distorted_crop_size = R.cam_params[cam_type].GetDistortedCropSize();
    f << "distorted_crop_size: " << distorted_crop_size[0] << ' ' << distorted_crop_size[1] << "\n";

    Eigen::Vector2i undist_size = R.cam_params[cam_type].GetUndistortedSize();
    f << "undistorted_image_size: " << undist_size[0] << ' ' << undist_size[1] << "\n";

    Eigen::Affine3d T;
    if (model_rig)
      T = R.ref_to_cam_trans[cam_type];
    else
      T = Eigen::Affine3d::Identity(); // write something valid

    f << "ref_to_sensor_transform: " << dense_map::affineToStr(T) << "\n";

    f << "depth_to_image_transform: " << dense_map::affineToStr(R.depth_to_image[cam_type]) << "\n";

    f << "ref_to_sensor_timestamp_offset: " << R.ref_to_cam_timestamp_offsets[cam_type] << "\n";
  }

  f.close();
}

// Read real values after given tag. Ignore comments, so any line starting
// with #, and empty lines. If desired_num_vals >=0, validate that we
// read the desired number.
void readConfigVals(std::ifstream & f, std::string const& tag, int desired_num_vals,
                    Eigen::VectorXd & vals) {
  // Clear the output
  vals.resize(0);

  std::vector<double> local_vals;  // std::vector has push_back()
  std::string line;
  while (getline(f, line)) {

    // Remove everything after any point sign
    bool have_comment = (line.find('#') != line.npos);
    if (have_comment) {
      std::string new_line;
      for (size_t c = 0; c < line.size(); c++) {
        if (line[c] == '#') 
          break; // got to the pound sign
        
        new_line += line[c];
      }

      line = new_line;
    }
    
    // Here must remove anything after the pound sign
    
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    std::string token;
    iss >> token;
    double val = 0.0;
    while (iss >> val) {
      local_vals.push_back(val);
    }

    if (token == "") 
      continue; // likely just whitespace is present on the line
    
    if (token != tag) throw std::runtime_error("Could not read value for: " + tag);

    // Copy to Eigen::VectorXd
    vals.resize(local_vals.size());
    for (int it = 0; it < vals.size(); it++) vals[it] = local_vals[it];

    if (desired_num_vals >= 0 && vals.size() != desired_num_vals)
      throw std::runtime_error("Read an incorrect number of values for: " + tag);

    return;
  }

  throw std::runtime_error("Could not read value for: " + tag);
}
  
// Read strings separated by spaces after given tag. Ignore comments, so any line starting
// with #, and empty lines. If desired_num_vals >=0, validate that we
// read the desired number.
void readConfigVals(std::ifstream & f, std::string const& tag, int desired_num_vals,
                    std::vector<std::string> & vals) {
  // Clear the output
  vals.resize(0);

  std::string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    std::string token;
    iss >> token;
    std::string val;
    while (iss >> val)
      vals.push_back(val);

    if (token != tag) throw std::runtime_error("Could not read value for: " + tag);

    if (desired_num_vals >= 0 && vals.size() != desired_num_vals)
      throw std::runtime_error("Read an incorrect number of values for: " + tag);

    return;
  }

  throw std::runtime_error("Could not read value for: " + tag);
}

// Read a rig configuration. Check if the transforms among the sensors
// on the rig is not 0, in that case will use it.
void readRigConfig(std::string const& rig_config, bool have_rig_transforms, RigSet & R) {
  try {
    // Initialize the outputs
    R = RigSet();

    // Open the file
    std::cout << "Reading: " << rig_config << std::endl;
    std::ifstream f;
    f.open(rig_config.c_str(), std::ios::in);
    if (!f.is_open()) LOG(FATAL) << "Cannot open file for reading: " << rig_config << "\n";

    int ref_sensor_count = 0;
    Eigen::VectorXd vals;
    std::vector<std::string> str_vals;

    // Read each sensor
    int sensor_it = -1;
    while (1) {
      sensor_it++;

      std::string ref_sensor_name;
      int curr_pos = f.tellg(); // where we are in the file
      // Read the reference sensor
      try {
        readConfigVals(f, "ref_sensor_name:", 1, str_vals);
        ref_sensor_name = str_vals[0];
        ref_sensor_count++; // found a ref sensor
        R.cam_set.resize(ref_sensor_count);
      } catch (...) {
        // No luck, go back to the line we tried to read, and continue reading other fields
        f.seekg(curr_pos, std::ios::beg);
      }
      
      try {
        readConfigVals(f, "sensor_name:", 1, str_vals);
      } catch(...) {
        // Likely no more sensors
        return;
      }
      std::string sensor_name = str_vals[0];

      // It is convenient to store each sensor in cam_set, which has the rig set structure,
      // and in R.cam_names, which is enough for almost all processing.
      R.cam_set.back().push_back(sensor_name);
      R.cam_names.push_back(sensor_name);
      
      readConfigVals(f, "focal_length:", 1, vals);
      Eigen::Vector2d focal_length(vals[0], vals[0]);

      readConfigVals(f, "optical_center:", 2, vals);
      Eigen::Vector2d optical_center(vals[0], vals[1]);

      readConfigVals(f, "distortion_coeffs:", -1, vals);
      if (vals.size() != 0 && vals.size() != 1 && vals.size() != 4 && vals.size() < 5)
        LOG(FATAL) << "Expecting 0, 1, 4, 5, or more distortion coefficients.\n";
      
      Eigen::VectorXd distortion = vals;
      
      readConfigVals(f, "distortion_type:", 1, str_vals);
      if (distortion.size() == 0 && str_vals[0] != dense_map::NO_DISTORION)
        LOG(FATAL) << "When there are no distortion coefficients, distortion type must be: "
                   << dense_map::NO_DISTORION << "\n";
      if (distortion.size() == 1 && str_vals[0] != dense_map::FISHEYE_DISTORTION)
        LOG(FATAL) << "When there is 1 distortion coefficient, distortion type must be: "
                   << dense_map::FISHEYE_DISTORTION << "\n";
      if ((distortion.size() == 4 || distortion.size() == 5) &&
          str_vals[0] != dense_map::RADTAN_DISTORTION)
        LOG(FATAL) << "When there are 4 or 5 distortion coefficient, distortion type must be: "
                   << dense_map::RADTAN_DISTORTION << "\n";
      if ((distortion.size() > 5) &&
          str_vals[0] != dense_map::RPC_DISTORTION)
        LOG(FATAL) << "When there are more than 5 distortion coefficients, distortion "
                   << "type must be: " << dense_map::RPC_DISTORTION << "\n";
      
      readConfigVals(f, "image_size:", 2, vals);
      Eigen::Vector2i image_size(vals[0], vals[1]);

      readConfigVals(f, "distorted_crop_size:", 2, vals);
      Eigen::Vector2i distorted_crop_size(vals[0], vals[1]);

      readConfigVals(f, "undistorted_image_size:", 2, vals);
      Eigen::Vector2i undistorted_image_size(vals[0], vals[1]);

      camera::CameraParameters params(image_size, focal_length, optical_center, distortion);
      params.SetDistortedCropSize(distorted_crop_size);
      params.SetUndistortedSize(undistorted_image_size);
      R.cam_params.push_back(params);

      readConfigVals(f, "ref_to_sensor_transform:", 12, vals);
      R.ref_to_cam_trans.push_back(vecToAffine(vals));

      // Sanity check
      if (have_rig_transforms &&
          R.ref_to_cam_trans.back().matrix() == 0 * R.ref_to_cam_trans.back().matrix()) {
        LOG(FATAL) << "Failed to read valid transforms among the sensors on the rig\n";
      }

      readConfigVals(f, "depth_to_image_transform:", 12, vals);
      R.depth_to_image.push_back(vecToAffine(vals));

      readConfigVals(f, "ref_to_sensor_timestamp_offset:", 1, vals);
      double timestamp_offset = vals[0];
      R.ref_to_cam_timestamp_offsets.push_back(timestamp_offset);
    }

    // Sanity check
    if (have_rig_transforms) {
      for (size_t cam_it = 0; cam_it < R.cam_names.size(); cam_it++) {
        if (R.isRefSensor(R.cam_names[cam_it]) &&
            R.ref_to_cam_trans[cam_it].matrix() != Eigen::Affine3d::Identity().matrix())
          LOG(FATAL) << "The transform from the reference sensor to itself must be the identity.\n";
      }
    }
    
    R.validate();
    
  } catch(std::exception const& e) {
    LOG(FATAL) << e.what() << "\n";
  }

  return;
}
  
}  // end namespace dense_map
