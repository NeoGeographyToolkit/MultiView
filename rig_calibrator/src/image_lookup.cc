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

#include <rig_calibrator/camera_image.h>
#include <rig_calibrator/rig_config.h>
#include <rig_calibrator/image_lookup.h>
#include <camera_model/camera_params.h>

#include <glog/logging.h>

#include <iostream>
#include <fstream>
#include <iomanip>

namespace dense_map {
  
// Sort by timestamps adjusted to be relative to the ref camera clock
bool timestampLess(cameraImage i, cameraImage j) {
  return (i.ref_timestamp < j.ref_timestamp);
}

// The images from the bag may need to be resized to be the same
// size as in the calibration file. Sometimes the full-res images
// can be so blurry that interest point matching fails, hence the
// resizing.
// Similar logic to deal with differences between image size and calibrated size
// is used further down this code.
void adjustImageSize(camera::CameraParameters const& cam_params, cv::Mat & image) {
  int64_t raw_image_cols = image.cols;
  int64_t raw_image_rows = image.rows;
  int64_t calib_image_cols = cam_params.GetDistortedSize()[0];
  int64_t calib_image_rows = cam_params.GetDistortedSize()[1];
  int64_t factor = raw_image_cols / calib_image_cols;

  if ((raw_image_cols != calib_image_cols * factor) ||
      (raw_image_rows != calib_image_rows * factor)) {
    LOG(FATAL) << "Image width and height are: " << raw_image_cols << ' ' << raw_image_rows
               << "\n"
               << "Calibrated image width and height are: "
               << calib_image_cols << ' ' << calib_image_rows << "\n"
               << "These must be equal up to an integer factor.\n";
  }

  if (factor != 1) {
    // TODO(oalexan1): This kind of resizing may be creating aliased images.
    cv::Mat local_image;
    cv::resize(image, local_image, cv::Size(), 1.0/factor, 1.0/factor, cv::INTER_AREA);
    local_image.copyTo(image);
  }

  // Check
  if (image.cols != calib_image_cols || image.rows != calib_image_rows)
    LOG(FATAL) << "Sci cam images have the wrong size.";
}

// Find an image at the given timestamp or right after it. We assume
// that during repeated calls to this function we always travel
// forward in time, and we keep track of where we are in the bag using
// the variable start_pos that we update as we go.
// TODO(oalexan1): Wipe this!
bool lookupImage(// Inputs
                 double desired_time, std::vector<ImageMessage> const& msgs,
                 // Outputs
                 cv::Mat& image, std::string & image_name,
                 int& start_pos, double& found_time) {
  // Initialize the outputs. Note that start_pos is passed in from outside.
  image = cv::Mat();
  image_name = "";
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
      // Found the desired data. Do a deep copy, to not depend on the
      // original structure.
      msgs[local_pos].image.copyTo(image);
      image_name = msgs[local_pos].name;
      return true;
    }
  }
  return false;
}

  
// Find an image at the given timestamp or right after it. We assume
// that during repeated calls to this function we always travel
// forward in time, and we keep track of where we are in the bag using
// the variable start_pos that we update as we go.
bool lookupImage(// Inputs
                 double desired_time, MsgMap const& msgs,
                 // Outputs
                 cv::Mat& image, std::string & image_name,
                 MsgMapIter& start_pos, double& found_time) {
  // Initialize the outputs. Note that start_pos is passed in from outside.
  image = cv::Mat();
  image_name = "";
  found_time = -1.0;

   int num_msgs = msgs.size();
   double prev_image_time = -1.0;

   for (auto local_pos = start_pos; local_pos != msgs.end(); local_pos++) {
     start_pos = local_pos;  // save this for exporting

     dense_map::ImageMessage const& imgMsg = local_pos->second; // alias
     found_time = imgMsg.timestamp;

     // Sanity check: We must always travel forward in time
     if (found_time < prev_image_time) {
       LOG(FATAL) << "Found images not in chronological order.\n"
                  << std::fixed << std::setprecision(17)
                  << "Times in wrong order: " << prev_image_time << ' ' << found_time << ".\n";
       continue;
     }

     prev_image_time = found_time;

     if (found_time >= desired_time) {
       // Found the desired data. Do a deep copy, to not depend on the
       // original structure.
       imgMsg.image.copyTo(image);
       image_name = imgMsg.name;
       return true;
     }
   }
  return false;
}

// A function to extract poses, filenames, and timestamps from read data.
void lookupFilesPoses(// Inputs
                      dense_map::RigSet const& R,
                      std::vector<std::map<double, dense_map::ImageMessage>> const& image_maps,
                      std::vector<std::map<double, dense_map::ImageMessage>> const& depth_maps,
                      // Outputs
                      std::vector<double>& ref_timestamps,
                      std::vector<Eigen::Affine3d> & world_to_ref) {

  // Sanity checks
  if (image_maps.size() != depth_maps.size() || image_maps.size() != R.cam_names.size())
    LOG(FATAL) << "Bookkeeping failure in lookupFilesPoses()\n";
  
  // Wipe the outputs
  ref_timestamps.clear();
  world_to_ref.clear();

  int num_cams = R.cam_names.size();
  for (size_t cam_type = 0; cam_type < image_maps.size(); cam_type++) {

    auto const& image_map = image_maps[cam_type];

    for (auto it = image_map.begin(); it != image_map.end(); it++) {
      // Collect the ref cam timestamps, world_to_ref, in chronological order
      if (R.isRefSensor(R.cam_names[cam_type])) {
        world_to_ref.push_back(it->second.world_to_cam);
        ref_timestamps.push_back(it->second.timestamp);
      }
    }
  }
}
  
// Look up each ref cam image by timestamp, with the rig
// assumption. In between any two ref cam timestamps, which are no
// further from each other than the bracket length, look up an image
// of each of the other camera types in the rig. If more than one choice, try to
// stay as close as possible to the midpoint of the two bracketing ref
// cam timestamps. This way there's more wiggle room later if one
// attempts to modify the timestamp offset. If there is no rig, keep
// all the images.
void lookupImagesAndBrackets(// Inputs
                             double bracket_len,
                             double timestamp_offsets_max_change,
                             dense_map::RigSet   const& R,
                             std::vector<double> const& ref_timestamps,
                             std::vector<MsgMap> const& image_data,
                             std::vector<MsgMap> const& depth_data,
                             // Outputs
                             std::vector<dense_map::cameraImage>& cams,
                             std::vector<double>& min_timestamp_offset,
                             std::vector<double>& max_timestamp_offset) {

  std::cout << "Looking up the images and bracketing the timestamps." << std::endl;

  int num_ref_cams = ref_timestamps.size();
  int num_cam_types = R.cam_names.size();

  // Sanity checks
  if (R.cam_names.size() != image_data.size()) 
    LOG(FATAL) << "Expecting as many sensors as image datasets for them.\n";
  if (R.cam_names.size() != depth_data.size()) 
    LOG(FATAL) << "Expecting as many sensors as depth datasets for them.\n";
    
  // Initialize the outputs
  cams.clear();
  min_timestamp_offset.resize(num_cam_types, -1.0e+100);
  max_timestamp_offset.resize(num_cam_types,  1.0e+100);

  // A lot of care is needed with positions. This remembers how we travel in time
  // for each camera type so we have fewer messages to search.
  // But if a mistake is done below it will mess up this bookkeeping.
  std::vector<MsgMapIter> image_start_positions(num_cam_types);
  std::vector<MsgMapIter> depth_start_positions(num_cam_types);
  for (int cam_it = 0; cam_it < num_cam_types; cam_it++) {
    image_start_positions[cam_it] = image_data[cam_it].begin();
    depth_start_positions[cam_it] = depth_data[cam_it].begin();
  }
  
  // Populate the data for each camera image
  for (int beg_ref_it = 0; beg_ref_it < num_ref_cams; beg_ref_it++) {

    // For when we have last ref timestamp and last other cam timestamp and they are equal
    int end_ref_it = beg_ref_it + 1;
    bool last_timestamp = (end_ref_it == num_ref_cams);
    if (last_timestamp) end_ref_it = beg_ref_it;

    for (int cam_type = 0; cam_type < num_cam_types; cam_type++) {
      dense_map::cameraImage cam;
      bool success = false;

      // The ref cam does not need bracketing, but the others need to be bracketed
      // by ref cam, so there are two cases to consider.

      if (R.isRefSensor(R.cam_names[cam_type])) {
        cam.camera_type   = cam_type;
        cam.timestamp     = ref_timestamps[beg_ref_it];
        cam.ref_timestamp = cam.timestamp;  // the time offset is 0 between ref and itself
        cam.beg_ref_index = beg_ref_it;
        cam.end_ref_index = beg_ref_it;  // same index for beg and end

        // Start looking up the image timestamp from this position. Some care
        // is needed here as we advance in time in image_start_positions[cam_type].
        double found_time = -1.0;
        // This has to succeed since this timestamp came from an existing image
        bool have_lookup =  
          dense_map::lookupImage(cam.timestamp, image_data[cam_type],
                                 // Outputs
                                 cam.image, cam.image_name, 
                                 image_start_positions[cam_type],  // this will move forward
                                 found_time);
        
        if (!have_lookup)
          LOG(FATAL) << std::fixed << std::setprecision(17)
                     << "Cannot look up camera at time " << cam.timestamp << ".\n";

        // The exact time is expected
        if (found_time != cam.timestamp)
          LOG(FATAL) << std::fixed << std::setprecision(17)
                     << "Cannot look up camera at time " << cam.timestamp << ".\n";

        success = true;

      } else {
        // Need care here since sometimes ref_cam and current cam can have
        // exactly the same timestamp, so then bracketing should succeed.

        // Convert the bracketing timestamps to current cam's time
        double ref_to_cam_offset = R.ref_to_cam_timestamp_offsets[cam_type];
        double beg_timestamp     = ref_timestamps[beg_ref_it] + ref_to_cam_offset;
        double end_timestamp     = ref_timestamps[end_ref_it] + ref_to_cam_offset;
        if (end_timestamp == beg_timestamp && last_timestamp)  // necessary adjustment
          end_timestamp = std::nextafter(end_timestamp, end_timestamp + 1.0); 

        if (end_timestamp <= beg_timestamp)
          LOG(FATAL) << "Ref timestamps must be in strictly increasing order.\n";

        // Find the image timestamp closest to the midpoint of the brackets. This will give
        // more room to vary the timestamp later.
        double mid_timestamp = (beg_timestamp + end_timestamp)/2.0;

        // Search forward in time from image_start_positions[cam_type].
        // We will update that too later. One has to be very careful
        // with it so it does not go too far forward in time
        // so that at the next iteration we are passed what we
        // search for.
        MsgMapIter start_pos = image_start_positions[cam_type]; // care here
        double curr_timestamp = beg_timestamp;                  // start here
        cv::Mat best_image;
        std::string best_image_name;
        double best_dist = 1.0e+100;
        double best_time = -1.0, found_time = -1.0;
        while (1) {
          if (found_time > end_timestamp) break;  // out of range

          cv::Mat image;
          std::string image_name;
          bool have_lookup =
            dense_map::lookupImage(curr_timestamp, image_data[cam_type],
                                   // Outputs
                                   image, image_name,
                                   // care here, start_pos moves forward
                                   start_pos,
                                   // found_time will be updated now
                                   found_time);

          if (!have_lookup)
            break;  // Need not succeed, but then there's no need to go on as we are at the end

          // Check if the found time is in the bracket
          bool is_in_bracket = (beg_timestamp <= found_time && found_time < end_timestamp);
          double curr_dist = std::abs(found_time - mid_timestamp);

          if (curr_dist < best_dist && is_in_bracket) {
            best_dist = curr_dist;
            best_time = found_time;
            // Update the start position for the future only if this is a good
            // solution. Otherwise we may have moved too far.
            image_start_positions[cam_type] = start_pos;
            image.copyTo(best_image);
            best_image_name = image_name;
          }

          // Go forward in time. We count on the fact that
          // lookupImage() looks forward from given guess.
          // Careful here with the api of std::nextafter().
          curr_timestamp = std::nextafter(found_time, found_time + 1.0);
        }

        if (best_time < 0.0) continue;  // bracketing failed

        if (best_time > beg_timestamp && end_timestamp - beg_timestamp > bracket_len)
          continue;  // Must respect the bracket length, unless best time equals beg time
        
        // Note how we allow best_time == beg_timestamp if there's no other choice
        if (best_time < beg_timestamp || best_time >= end_timestamp)
          continue;  // no luck

        cam.camera_type   = cam_type;
        cam.timestamp     = best_time;
        cam.ref_timestamp = best_time - ref_to_cam_offset;
        cam.beg_ref_index = beg_ref_it;
        cam.end_ref_index = end_ref_it;
        cam.image         = best_image;
        cam.image_name    = best_image_name;

        success = true;
      }

      if (!success) continue;

      if (!R.isRefSensor(R.cam_names[cam_type])) { // Not a ref sensor
        double ref_to_cam_offset = R.ref_to_cam_timestamp_offsets[cam_type];

        // cam.timestamp was chosen as centrally as possible so that
        // ref_timestamps[beg_ref_it] + ref_to_cam_offset <= cam.timestamp
        // and
        // cam.timestamp <= ref_timestamps[end_ref_it] + ref_to_cam_offset
        // Find the range of potential future values of ref_to_cam_offset so that
        // cam.timestamp still respects these bounds.
        min_timestamp_offset[cam_type]
          = std::max(min_timestamp_offset[cam_type], cam.timestamp - ref_timestamps[end_ref_it]);
        max_timestamp_offset[cam_type]
          = std::min(max_timestamp_offset[cam_type], cam.timestamp - ref_timestamps[beg_ref_it]);
      }

      // Look up the closest depth in time (either before or after cam.timestamp)
      // This need not succeed.
      cam.cloud_timestamp = -1.0;  // will change
      if (!depth_data.empty()) 
        dense_map::lookupImage(cam.timestamp,  // start looking from this time forward
                               depth_data[cam_type],
                               // Outputs
                               cam.depth_cloud, cam.depth_name, 
                               depth_start_positions[cam_type],  // this will move forward
                               cam.cloud_timestamp);             // found time
      
      cams.push_back(cam);
    }  // end loop over camera types
  }    // end loop over ref images

  std::cout << "Timestamp offset allowed ranges based on current bracketing:\n";
  // Adjust for timestamp_offsets_max_change
  for (int cam_type = 0; cam_type < num_cam_types; cam_type++) {
    if (R.isRefSensor(R.cam_names[cam_type]))
      continue;  // bounds don't make sense here
    min_timestamp_offset[cam_type] = std::max(min_timestamp_offset[cam_type],
                                              R.ref_to_cam_timestamp_offsets[cam_type]
                                              - timestamp_offsets_max_change);
    max_timestamp_offset[cam_type] = std::min(max_timestamp_offset[cam_type],
                                              R.ref_to_cam_timestamp_offsets[cam_type]
                                              + timestamp_offsets_max_change);

    // Tighten a bit to ensure we don't exceed things when we add
    // and subtract timestamps later. Note that timestamps are
    // measured in seconds and fractions of a second since epoch and
    // can be quite large so precision loss can easily happen.
    min_timestamp_offset[cam_type] += 1.0e-5;
    max_timestamp_offset[cam_type] -= 1.0e-5;
    std::cout << std::setprecision(8) << R.cam_names[cam_type]
              << ": [" << min_timestamp_offset[cam_type]
              << ", " << max_timestamp_offset[cam_type] << "]\n";
  }

}

// Assuming that the rig constraint is not used, initialize the 'cams' structure
// by copying each image and its other data in that structure as expected
// by later code. See also lookupImagesAndBrackets() when some selection based
// on bracketing takes place.
void lookupImagesNoBrackets(// Inputs
                            dense_map::RigSet const& R,
                            std::vector<MsgMap> const& image_data,
                            std::vector<MsgMap> const& depth_data,
                            // Outputs
                            std::vector<dense_map::cameraImage>& cams,
                            std::vector<double>& min_timestamp_offset,
                            std::vector<double>& max_timestamp_offset) {

  std::cout << "Looking up the images." << std::endl;
  int num_cam_types = R.cam_names.size();
  
  // Initialize the outputs
  cams.clear();
  min_timestamp_offset.resize(num_cam_types, -1.0e+100);
  max_timestamp_offset.resize(num_cam_types,  1.0e+100);

  // A lot of care is needed with positions. This remembers how we travel in time
  // for each camera type so we have fewer messages to search.
  // But if a mistake is done below it will mess up this bookkeeping.
  std::vector<MsgMapIter> image_start_positions(num_cam_types);
  std::vector<MsgMapIter> depth_start_positions(num_cam_types);
  for (int cam_it = 0; cam_it < num_cam_types; cam_it++) {
    image_start_positions[cam_it] = image_data[cam_it].begin();
    depth_start_positions[cam_it] = depth_data[cam_it].begin();
  }

  // Populate the data for each camera image
  for (int cam_type = 0; cam_type < num_cam_types; cam_type++) {

    int cam_it = -1;
    for (auto map_it = image_data[cam_type].begin(); map_it != image_data[cam_type].end();
         map_it++) {
      cam_it++;
      
      dense_map::cameraImage cam;
      cam.camera_type   = cam_type;
      cam.timestamp     = (map_it->second).timestamp;
      cam.ref_timestamp = cam.timestamp; // no rig, so no timestamp offset
      // These two values below should not be needed with no rig
      cam.beg_ref_index = cam_it;
      cam.end_ref_index = cam_it;

      // Start looking up the image timestamp from this position. Some care
      // is needed here as we advance in time in image_start_positions[cam_type].
      double found_time = -1.0;
      // This has to succeed since this timestamp originally came from an existing image
      bool have_lookup =  
        dense_map::lookupImage(cam.timestamp, image_data[cam_type],
                               // Outputs
                               cam.image, cam.image_name, 
                               image_start_positions[cam_type],  // this will move forward
                               found_time);
      if (!have_lookup)
        LOG(FATAL) << std::fixed << std::setprecision(17)
                   << "Cannot look up camera at time " << cam.timestamp << ".\n";

      // The exact time is expected
      if (found_time != cam.timestamp)
        LOG(FATAL) << std::fixed << std::setprecision(17)
                   << "Cannot look up camera at time " << cam.timestamp << ".\n";
      
      // Look up the closest cloud in time (either before or after cam.timestamp)
      // This need not succeed.
      cam.cloud_timestamp = -1.0;  // will change
      if (!depth_data.empty()) 
        dense_map::lookupImage(cam.timestamp,  // start looking from this time forward
                               depth_data[cam_type],
                               // Outputs
                               cam.depth_cloud, cam.depth_name, 
                               depth_start_positions[cam_type],  // this will move forward
                               cam.cloud_timestamp);             // found time

      // Accept this camera
      cams.push_back(cam);
    }  // end loop over camera types
  }    // end loop over ref images

  return;
}

// Look up images, with or without the rig constraint. See individual functions
// below for more details.
void lookupImagesOneRig(// Inputs
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
                        std::vector<double>                 & max_timestamp_offset) {
  
  dense_map::lookupFilesPoses(// Inputs
                              R, image_maps, depth_maps,
                              // Outputs
                              ref_timestamps, world_to_ref);
  
  if (!no_rig) 
    lookupImagesAndBrackets(// Inputs
                            bracket_len,  
                            timestamp_offsets_max_change,  
                            R, ref_timestamps,  image_maps, depth_maps,  
                            // Outputs
                            cams, min_timestamp_offset, max_timestamp_offset);
  else
    lookupImagesNoBrackets(// Inputs
                           R, image_maps, depth_maps,  
                           // Outputs
                           cams, min_timestamp_offset, max_timestamp_offset);
  
  // See how many timestamps we have for each camera
  std::map<int, int> num_images;
  int num_cam_types = R.cam_names.size();
  for (int cam_type_it = 0; cam_type_it < num_cam_types; cam_type_it++)
    num_images[cam_type_it] = 0;
  for (size_t cam_it = 0; cam_it < cams.size(); cam_it++)
    num_images[cams[cam_it].camera_type]++;
  bool is_good = true;
  for (int cam_type_it = 0; cam_type_it < num_cam_types; cam_type_it++) {
    std::cout << "Number of images for sensor: " << R.cam_names[cam_type_it] << ": "
              << num_images[cam_type_it] << std::endl;

    if (num_images[cam_type_it] == 0)
      is_good = false;
  }
  if (!is_good)
    LOG(FATAL) << "Could not find images for all sensors. Cannot continue.\n";

  // The images may need to be resized to be the same
  // size as in the calibration file. Sometimes the full-res images
  // can be so blurry that interest point matching fails, hence the
  // resizing.
  for (size_t it = 0; it < cams.size(); it++)
    dense_map::adjustImageSize(R.cam_params[cams[it].camera_type], cams[it].image);

  // Sort by the timestamp in reference camera time. This is essential
  // for matching each image to other images close in time. Note
  // that this does not affect the book-keeping of beg_ref_index
  // and end_ref_it in this vector because those indices point to
  // world_to_ref and ref_timestamp, which do not change.
  // TODO(oalexan1): This will be wrong with multiple rigs.
  // Matches should be among images on same rig?
  std::sort(cams.begin(), cams.end(), dense_map::timestampLess);

  // Parse the transform from the world to each cam, which were known
  // on input. Later, if use_initial_rig_transform is specified,
  // these will be computed based on the rig.  Since
  // image_maps[cam_type] is sorted chronologically, travel in time
  // along at as we move along the cams array. Use the two arrays
  // below to remember where we left off.
  // TODO(oalexan1): This is fragile. It relies on cams being sorted by time.
  // Make the cams array have a world_to_cam entry and remove the loop below.
  world_to_cam.resize(cams.size());
  std::vector<MsgMapIter> beg_pos(num_cam_types); 
  std::vector<MsgMapIter> end_pos(num_cam_types); 
  for (int cam_type = 0; cam_type < num_cam_types; cam_type++) {
    beg_pos[cam_type] = image_maps[cam_type].begin();
    end_pos[cam_type] = image_maps[cam_type].end();
  }
  for (size_t cam_it = 0; cam_it < cams.size(); cam_it++) {
    int cam_type = cams[cam_it].camera_type;
    for (auto pos = beg_pos[cam_type]; pos != end_pos[cam_type]; pos++) {
      if (cams[cam_it].timestamp == pos->first) {
        world_to_cam[cam_it] = (pos->second).world_to_cam;
        beg_pos[cam_type] = pos;  // save for next time
        break;
      }
    }
  }  
  return; 
}

// Look up images for a set of rigs. This requires looking up images for individual rigs,
// then concatenating the results and adjusting the book-keeping.
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
                  std::vector<double>                 & max_timestamp_offset) {

  // Wipe the outputs
  ref_timestamps.clear();
  world_to_ref.clear();
  cams.clear();
  world_to_cam.clear();
  min_timestamp_offset.clear();
  max_timestamp_offset.clear();

  for (size_t rig_id = 0; rig_id < R.cam_set.size(); rig_id++) {

    // Create a single rig
    std::cout << "--rig id is " << rig_id << std::endl;
    dense_map::RigSet sub_rig = R.subRig(rig_id);

    // Prepare the inputs for the subrig
    std::vector<MsgMap> sub_image_maps;
    std::vector<MsgMap> sub_depth_maps;
    for (size_t sub_it = 0; sub_it < sub_rig.cam_names.size(); sub_it++) {
      std::string sensor_name = sub_rig.cam_names[sub_it];
      std::cout << "sensor name " << sensor_name << std::endl;
      int rig_set_it = R.sensorIndex(sensor_name); // index in the larger rig
      std::cout << "---got " << sub_it << ' ' << sensor_name << ' ' << rig_set_it << std::endl;
      sub_image_maps.push_back(image_maps[rig_set_it]);
      sub_depth_maps.push_back(depth_maps[rig_set_it]);
    }

    std::vector<double>                 sub_ref_timestamps;
    std::vector<Eigen::Affine3d>        sub_world_to_ref;
    std::vector<dense_map::cameraImage> sub_cams;
    std::vector<Eigen::Affine3d>        sub_world_to_cam;
    std::vector<double>                 sub_min_timestamp_offset;
    std::vector<double>                 sub_max_timestamp_offset;

    // Do the work for the subrig
    lookupImagesOneRig(// Inputs
                       no_rig, bracket_len, timestamp_offsets_max_change, sub_rig,  
                       sub_image_maps, sub_depth_maps,  
                       // Outputs
                       sub_ref_timestamps, sub_world_to_ref, sub_cams,  
                       sub_world_to_cam, sub_min_timestamp_offset, sub_max_timestamp_offset);

    // Save the endpoints for ref timestamps and all cams, before concatenation
    size_t prev_ref_end = ref_timestamps.size();
    size_t prev_end = cams.size();

    std::cout << "--prev ref end " << prev_ref_end << std::endl;
    std::cout << "--prev end " << prev_end << std::endl;

    // Append the answers
    ref_timestamps.insert(ref_timestamps.end(), sub_ref_timestamps.begin(),
                          sub_ref_timestamps.end());
    world_to_ref.insert(world_to_ref.end(), sub_world_to_ref.begin(), sub_world_to_ref.end());
    cams.insert(cams.end(), sub_cams.begin(), sub_cams.end());
    world_to_cam.insert(world_to_cam.end(), sub_world_to_cam.begin(), sub_world_to_cam.end());
    min_timestamp_offset.insert(min_timestamp_offset.end(), sub_min_timestamp_offset.begin(),
                                sub_min_timestamp_offset.end());
    max_timestamp_offset.insert(max_timestamp_offset.end(), sub_max_timestamp_offset.begin(),
                                sub_max_timestamp_offset.end());

    std::cout << "--curr ref end " << ref_timestamps.size() << std::endl;
    std::cout << "--curr end " << cams.size() << std::endl;
    
    // Update the bookkeeping in 'cams'
    for (size_t cam_it = prev_end; cam_it < cams.size(); cam_it++) {

      // Find the current sensor index in the larger rig set
      int subrig_sensor_index = cams[cam_it].camera_type;
      std::string subrig_sensor = sub_rig.cam_names[subrig_sensor_index];
      int rig_sensor_index = R.sensorIndex(subrig_sensor);
      cams[cam_it].camera_type = rig_sensor_index;

      std::cout << "--sub and full sensor index " << subrig_sensor_index << ' ' << rig_sensor_index
                << std::endl;
      
      // Update the pointers to indices in ref_timestamps
      std::cout << "--sub ref beg end " << cams[cam_it].beg_ref_index << ' ' << cams[cam_it].end_ref_index << std::endl;
      cams[cam_it].beg_ref_index += prev_ref_end;     
      cams[cam_it].end_ref_index += prev_ref_end;

      std::cout << "--full ref beg end " << cams[cam_it].beg_ref_index << ' ' << cams[cam_it].end_ref_index << std::endl;
    }
  }

  return;
}
  
}  // end namespace dense_map
