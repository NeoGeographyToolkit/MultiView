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

#include <rig_calibrator/nvm.h>
#include <camera_model/camera_params.h>
#include <rig_calibrator/camera_image.h>
#include <rig_calibrator/basic_algs.h>
#include <rig_calibrator/system_utils.h>

#include <boost/filesystem.hpp>
#include <glog/logging.h>

#include <iostream>
#include <fstream>

namespace fs = boost::filesystem;

namespace dense_map {

// Reads the NVM control network format. The interest points may or may not
// be shifted relative to optical center. The user is responsible for knowing that.
void ReadNvm(std::string               const & input_filename,
             std::vector<Eigen::Matrix2Xd>   & cid_to_keypoint_map,
             std::vector<std::string>        & cid_to_filename,
             std::vector<std::map<int, int>> & pid_to_cid_fid,
             std::vector<Eigen::Vector3d>    & pid_to_xyz,
             std::vector<Eigen::Affine3d>    & cid_to_cam_t_global) {

  std::cout << "Reading: " << input_filename << std::endl;
  std::ifstream f(input_filename, std::ios::in);
  std::string token;
  std::getline(f, token);
  
  // Assert that we start with our NVM token
  if (token.compare(0, 6, "NVM_V3") != 0) {
    LOG(FATAL) << "File doesn't start with NVM token.";
  }

  // Read the number of cameras
  ptrdiff_t number_of_cid;
  f >> number_of_cid;
  if (number_of_cid < 1) {
    LOG(FATAL) << "NVM file is missing cameras.";
  }

  // Resize all our structures to support the number of cameras we now expect
  cid_to_keypoint_map.resize(number_of_cid);
  cid_to_filename.resize(number_of_cid);
  cid_to_cam_t_global.resize(number_of_cid);
  for (ptrdiff_t cid = 0; cid < number_of_cid; cid++) {
    // Clear keypoints from map. We'll read these in shortly
    cid_to_keypoint_map.at(cid).resize(Eigen::NoChange_t(), 2);

    // Read the line that contains camera information
    double focal, dist1, dist2;
    Eigen::Quaterniond q;
    Eigen::Vector3d c;
    f >> token >> focal;
    f >> q.w() >> q.x() >> q.y() >> q.z();
    f >> c[0] >> c[1] >> c[2] >> dist1 >> dist2;
    cid_to_filename.at(cid) = token;

    // Solve for t, which is part of the affine transform
    Eigen::Matrix3d r = q.matrix();
    cid_to_cam_t_global.at(cid).linear() = r;
    cid_to_cam_t_global.at(cid).translation() = -r * c;
  }

  // Read the number of points
  ptrdiff_t number_of_pid;
  f >> number_of_pid;
  if (number_of_pid < 1) {
    LOG(FATAL) << "The NVM file has no triangulated points.";
  }

  // Read the point
  pid_to_cid_fid.resize(number_of_pid);
  pid_to_xyz.resize(number_of_pid);
  Eigen::Vector3d xyz;
  Eigen::Vector3i color;
  Eigen::Vector2d pt;
  ptrdiff_t cid, fid;
  for (ptrdiff_t pid = 0; pid < number_of_pid; pid++) {
    pid_to_cid_fid.at(pid).clear();

    ptrdiff_t number_of_measures;
    f >> xyz[0] >> xyz[1] >> xyz[2] >>
      color[0] >> color[1] >> color[2] >> number_of_measures;
    pid_to_xyz.at(pid) = xyz;
    for (ptrdiff_t m = 0; m < number_of_measures; m++) {
      f >> cid >> fid >> pt[0] >> pt[1];

      pid_to_cid_fid.at(pid)[cid] = fid;

      if (cid_to_keypoint_map.at(cid).cols() <= fid) {
        cid_to_keypoint_map.at(cid).conservativeResize(Eigen::NoChange_t(), fid + 1);
      }
      cid_to_keypoint_map.at(cid).col(fid) = pt;
    }

    if (!f.good())
      LOG(FATAL) << "Unable to correctly read PID: " << pid;
  }
}

// Write the inliers in nvm format. The keypoints are shifted relative to the optical
// center, as written by Theia if shift_keypoints is specified.
// We handle properly the case when a (cid, fid) shows up in many tracks
// (this was a bug).
void writeInliersToNvm
(std::string                                       const& nvm_file,
 bool                                                     shift_keypoints, 
 std::vector<camera::CameraParameters>             const& cam_params,
 std::vector<dense_map::cameraImage>               const& cams,
 std::vector<Eigen::Affine3d>                      const& world_to_cam,
 std::vector<std::vector<std::pair<float, float>>> const& keypoint_vec,
 std::vector<std::map<int, int>>                   const& pid_to_cid_fid,
 std::vector<std::map<int, std::map<int, int>>>    const& pid_cid_fid_inlier,
 std::vector<Eigen::Vector3d>                      const& xyz_vec) {
  
  // Sanity checks
  if (world_to_cam.size() != cams.size())
    LOG(FATAL) << "Expecting as many world-to-camera transforms as cameras.\n";
  if (world_to_cam.size() != keypoint_vec.size()) 
    LOG(FATAL) << "Expecting as many sets of keypoints as cameras.\n";  
  if (pid_to_cid_fid.size() != pid_cid_fid_inlier.size())
    LOG(FATAL) << "Expecting as many inlier flags as there are tracks.\n";
  if (pid_to_cid_fid.size() != xyz_vec.size()) 
    LOG(FATAL) << "Expecting as many tracks as there are triangulated points.\n";

  // Initialize the keypoints in expected format. Copy the filenames
  // and focal lengths.
  std::vector<Eigen::Matrix2Xd> cid_to_keypoint_map(keypoint_vec.size());
  std::vector<std::string> cid_to_filename(keypoint_vec.size());
  for (size_t cid = 0; cid < cams.size(); cid++) {
    cid_to_keypoint_map[cid] = Eigen::MatrixXd(2, keypoint_vec[cid].size());
    cid_to_filename[cid] = cams[cid].image_name;
  }
  
  // Copy over only inliers, and tracks of length >= 2.
  std::vector<std::map<int, int>> nvm_pid_to_cid_fid;
  std::vector<Eigen::Vector3d> nvm_pid_to_xyz;

  for (size_t pid = 0; pid < pid_to_cid_fid.size(); pid++) {

    std::map<int, int> nvm_cid_fid;
    for (auto cid_fid = pid_to_cid_fid[pid].begin();
         cid_fid != pid_to_cid_fid[pid].end(); cid_fid++) {
      int cid = cid_fid->first;
      int fid = cid_fid->second;

      // Keep inliers only
      if (!dense_map::getMapValue(pid_cid_fid_inlier, pid, cid, fid))
        continue;

      Eigen::Vector2d dist_ip(keypoint_vec[cid][fid].first,
                              keypoint_vec[cid][fid].second);

      // Offset relative to the optical center
      if (shift_keypoints) 
        dist_ip -= cam_params[cams[cid].camera_type].GetOpticalOffset();

      // Add this to the keypoint map for cid in the location at fid_count[cid]
      cid_to_keypoint_map.at(cid).col(fid) = dist_ip;
      nvm_cid_fid[cid] = fid;
    }

    // Keep only tracks with at least two points
    if (nvm_cid_fid.size() > 1) {
      nvm_pid_to_cid_fid.push_back(nvm_cid_fid);
      nvm_pid_to_xyz.push_back(xyz_vec[pid]);
    }
  }
  
  WriteNvm(cid_to_keypoint_map, cid_to_filename, nvm_pid_to_cid_fid,  
           nvm_pid_to_xyz, world_to_cam, nvm_file);

  // Write the optical center per image
  if (shift_keypoints) {
    int file_len = nvm_file.size(); // cast to int to make subtraction safe
    // Remove .nvm and add new suffix
    std::string offset_path = nvm_file.substr(0, std::max(file_len - 4, 0))
      + "_offsets.txt";
    std::cout << "Writing optical center per image: " << offset_path << std::endl;
    std::ofstream offset_fh(offset_path.c_str());
    offset_fh.precision(17);   // Save with the highest precision
    for (size_t cid = 0; cid < cid_to_filename.size(); cid++) {
      auto const& optical_center = cam_params[cams[cid].camera_type].GetOpticalOffset();
      offset_fh << cid_to_filename[cid] << " "
                << optical_center[0] << " " << optical_center[1] << "\n";
    }
  }
}

// Write an nvm file. Keypoints may or may not be shifted relative to the optical center.
// The focal length is set to 0. Intrinsics need to be saved in some other data structure.
void WriteNvm(std::vector<Eigen::Matrix2Xd> const& cid_to_keypoint_map,
              std::vector<std::string> const& cid_to_filename,
              std::vector<std::map<int, int>> const& pid_to_cid_fid,
              std::vector<Eigen::Vector3d> const& pid_to_xyz,
              std::vector<Eigen::Affine3d> const& cid_to_cam_t_global,
              std::string const& output_filename) {

  // Ensure that the output directory exists
  std::string out_dir = fs::path(output_filename).parent_path().string();
  dense_map::createDir(out_dir);

  std::cout << "Writing: " << output_filename << std::endl;
  std::fstream f(output_filename, std::ios::out);
  f.precision(17); // double precision
  f << "NVM_V3\n";

  CHECK(cid_to_filename.size() == cid_to_keypoint_map.size())
    << "Unequal number of filenames and keypoints";
  CHECK(pid_to_cid_fid.size() == pid_to_xyz.size())
    << "Unequal number of pid_to_cid_fid and xyz measurements";
  CHECK(cid_to_filename.size() == cid_to_cam_t_global.size())
    << "Unequal number of filename and camera transforms";

  // Write camera information
  f << cid_to_filename.size() << std::endl;
  for (size_t cid = 0; cid < cid_to_filename.size(); cid++) {

    // World-to-camera rotation quaternion
    Eigen::Quaterniond q(cid_to_cam_t_global[cid].rotation());

    // Camera center in world coordinates
    Eigen::Vector3d t(cid_to_cam_t_global[cid].translation());
    Eigen::Vector3d camera_center =
      - cid_to_cam_t_global[cid].rotation().inverse() * t;

    f << cid_to_filename[cid] << " " << 0 // focal length is set to 0, not used
      << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
      << camera_center[0] << " " << camera_center[1] << " "
      << camera_center[2] << " " << "0 0\n"; // zero distortion, not used
  }

  // Write the number of points
  f << pid_to_cid_fid.size() << std::endl;

  for (size_t pid = 0; pid < pid_to_cid_fid.size(); pid++) {
    f << pid_to_xyz[pid][0] << " " << pid_to_xyz[pid][1] << " "
      << pid_to_xyz[pid][2] << " 0 0 0 "
      << pid_to_cid_fid[pid].size();

    CHECK(pid_to_cid_fid[pid].size() > 1)
      << "PID " << pid << " has " << pid_to_cid_fid[pid].size() << " measurements";

    for (std::map<int, int>::const_iterator it = pid_to_cid_fid[pid].begin();
         it != pid_to_cid_fid[pid].end(); it++) {
      f << " " << it->first << " " << it->second << " "
        << cid_to_keypoint_map[it->first].col(it->second)[0] << " "
        << cid_to_keypoint_map[it->first].col(it->second)[1];
    }
    f << std::endl;
  }

  // Close the file
  f.flush();
  f.close();
}
  
}  // end namespace dense_map
