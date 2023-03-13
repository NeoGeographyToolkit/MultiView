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

#ifndef IMAGE_LOOKUP_H_
#define IMAGE_LOOKUP_H_

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <boost/shared_ptr.hpp>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace camera {
  // Forward declaration
  class CameraParameters;
}

namespace dense_map {

// Forward declarations  
class cameraImage;
class ImageMessage;
class RigSet;

// Read a vector of strings from a file, with spaces and newlines
// acting as separators.  Store them in a set.
void readList(std::string const& file, std::set<std::string> & list);

// The cam name is the subdir having the images.
// Example: mydir/nav_cam/file.jpg has nav_cam as the cam name.
std::string camName(std::string const& image_file);
  
// Given a file with name <path to>/<cam name>/<digits>.<digits>.jpg,
// find the cam name, then look up the cam type. Also find the timestamp.
void findCamTypeAndTimestamp(std::string const& image_file,
                             std::vector<std::string> const& cam_names,
                             // Outputs
                             int    & cam_type,
                             double & timestamp);
  
// Look up images, with or without the rig constraint. See individual
// functions below for more details.
typedef std::map<double, dense_map::ImageMessage> MsgMap;
typedef MsgMap::const_iterator MsgMapIter;
void lookupImages(// Inputs
                  bool no_rig, double bracket_len,
                  double timestamp_offsets_max_change,
                  dense_map::RigSet const& R,
                  std::vector<MsgMap> const& image_maps,
                  std::vector<MsgMap> const& depth_maps,
                  // Outputs
                  std::vector<double>                 & ref_timestamps,
                  std::vector<Eigen::Affine3d>        & world_to_ref,
                  std::vector<dense_map::cameraImage> & cams,
                  std::vector<Eigen::Affine3d>        & world_to_cam,
                  std::vector<double>                 & min_timestamp_offset,
                  std::vector<double>                 & max_timestamp_offset);
  
}  // namespace dense_map

#endif  // IMAGE_LOOKUP_H_
