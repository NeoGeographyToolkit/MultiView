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

// Logic for nvm files and the sfm solution
#ifndef NVM_H_
#define NVM_H_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <vector>
#include <map>

namespace camera {
  // Forward declaration
  class CameraParameters;
}

namespace dense_map {

// Forward declarations
class cameraImage;
class ImageMessage;
}

namespace dense_map {

struct nvmData {
  std::vector<Eigen::Matrix2Xd>    cid_to_keypoint_map;
  std::vector<std::string>         cid_to_filename;
  std::vector<std::map<int, int>>  pid_to_cid_fid;
  std::vector<Eigen::Vector3d>     pid_to_xyz;
  std::vector<Eigen::Affine3d>     cid_to_cam_t_global;
};

// Read cameras and interest points from an nvm file  
void ReadNVM(std::string const& input_filename,
             std::vector<Eigen::Matrix2Xd> * cid_to_keypoint_map,
             std::vector<std::string> * cid_to_filename,
             std::vector<std::map<int, int>> * pid_to_cid_fid,
             std::vector<Eigen::Vector3d> * pid_to_xyz,
             std::vector<Eigen::Affine3d> *
             cid_to_cam_t_global);

// Write the inliers in nvm format. The keypoints are shifted relative to the optical
// center, as written by Theia.
void writeInliersToNvm(std::string                                       const& nvm_file,
                       bool                                                     shift_keypoints,
                       std::vector<camera::CameraParameters>             const& cam_params,
                       std::vector<dense_map::cameraImage>               const& cams,
                       std::vector<Eigen::Affine3d>                      const& world_to_cam,
                       std::vector<std::vector<std::pair<float, float>>> const& keypoint_vec,
                       std::vector<std::map<int, int>>                   const& pid_to_cid_fid,
                       std::vector<std::map<int, std::map<int, int>>>    const& pid_cid_fid_inlier,
                       std::vector<Eigen::Vector3d>                      const& xyz_vec);
  
// Write an nvm file. Note that a single focal length is assumed and no distortion.
// Those are ignored, and only camera poses, matches, and keypoints are used.
// Write an nvm file. Note that a single focal length is assumed and no distortion.
// Those are ignored, and only camera poses, matches, and keypoints are used.
// It is assumed the interest point matches are shifted relative to the optical center.
void writeNvm(std::vector<Eigen::Matrix2Xd> const& cid_to_keypoint_map,
              std::vector<std::string> const& cid_to_filename,
              std::vector<double> const& focal_lengths,
              std::vector<std::map<int, int>> const& pid_to_cid_fid,
              std::vector<Eigen::Vector3d> const& pid_to_xyz,
              std::vector<Eigen::Affine3d> const& cid_to_cam_t_global,
              std::string const& output_filename);

}  // namespace dense_map

#endif  // NVM_H_
