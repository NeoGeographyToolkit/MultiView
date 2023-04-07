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
#include <rig_calibrator/thread.h>
#include <rig_calibrator/tensor.h>
#include <rig_calibrator/sparse_map.h>
#include <rig_calibrator/sparse_mapping.h>
#include <rig_calibrator/interest_point.h>
#include <rig_calibrator/reprojection.h>
#include <rig_calibrator/nvm.h>
#include <rig_calibrator/rig_config.h>
#include <rig_calibrator/merge_maps.h>
#include <rig_calibrator/basic_algs.h>

#include <opencv2/features2d/features2d.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <thread>

// Merge n maps by merging second into the first, then the third into
// the merged map, etc. It works by finding matches among the maps
// using -num_image_overlaps_at_endpoints and then bringing the second
// map in the coordinate system of the first map. It is suggested that
// bundle adjustment and registration to real-world coordinate systems
// be (re-)done after maps are merged, using rig_calibrator.

DEFINE_string(rig_config, "",
              "Read the configuration of sensors from this file in the format used for "
              "rig_calibrator, even though this tool does not use the rig structure. "
              "The output of this program can be passed back to rig_calibrator, "
              "with or without the rig constraint.");

DEFINE_string(output_map, "",
              "Output file containing the merged map.");

DEFINE_int32(num_image_overlaps_at_endpoints, 10,
             "Search this many images at the beginning and end of the first map "
             "for matches to this many images at the beginning and end of the "
             "second map.");

DEFINE_bool(fix_first_map, false,
            "If true, after merging the maps and reconciling the camera poses for the "
            "shared images, overwrite the shared poses with those from the first map.");

DEFINE_bool(fast_merge, false,
            "When merging maps that have shared images, use their camera poses to "
            "find the transform from other maps to first map, and skip finding "
            "additional matches among the images.");

DEFINE_bool(no_shift, false,
            "Assume that in the input .nvm files the features are not shifted "
            "relative to the optical center. The merged map will then be saved "
            "the same way. The usual behavior is that .nvm file features are "
            "shifted, then this tool internally undoes the shift.");

DEFINE_bool(no_transform, false,
            "Do not compute and apply a transform from the other "
            "maps to the first one. This keeps the camera poses as "
            "they are (shared poses and features will be reconciled). "
            "This will succeed even when the two maps do not overlap.");

DEFINE_double(close_dist, -1.0,
              "Two triangulated points are considered to be close if no further "
              "than this distance, in meters. Used as inlier threshold when "
              "identifying triangulated points after the maps are "
              "aligned. Auto-computed, taking into account the extent of "
              "a tight subset of the triangulated points and printed on screen if "
              "not set. This is an advanced option. ");

void parameterValidation(int argc, char** argv) {

  if (argc < 3)
    LOG(FATAL) << "Usage: " << argv[0] << " <input maps> -output-map <output map>";

  // Ensure we don't over-write one of the inputs
  for (int i = 1; i < argc; i++) {
    if (argv[i] == FLAGS_output_map)
      LOG(FATAL) << "The input and output maps must have different names.";
  }
  
  if (FLAGS_rig_config == "")
    LOG(FATAL) << "The rig configuration was not specified.";

  if (FLAGS_output_map == "")
    LOG(FATAL) << "No output map was specified.";
  
  if (!FLAGS_fast_merge && FLAGS_num_image_overlaps_at_endpoints <= 0)
    LOG(FATAL) << "Must have num_image_overlaps_at_endpoints > 0.";

  if (FLAGS_fix_first_map && argc != 3)
    LOG(FATAL) << "Keeping the first map fixed works only when there are two input maps.";
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  parameterValidation(argc, argv);

  dense_map::RigSet R;
  bool use_initial_rig_transforms = false; // not used, part of the API
  dense_map::readRigConfig(FLAGS_rig_config, use_initial_rig_transforms, R);

  dense_map::nvmData in0;
  dense_map::ReadNvm(argv[1],
                     in0.cid_to_keypoint_map,  
                     in0.cid_to_filename,  
                     in0.pid_to_cid_fid,  
                     in0.pid_to_xyz,  
                     in0.cid_to_cam_t_global);
  if (!FLAGS_no_shift) {
    bool undo_shift = true; // remove the shift relative to the optical center
    dense_map::shiftKeypoints(undo_shift, R, in0);
  }
  
  // Successively append the maps
  dense_map::nvmData out_map;
  for (int i = 2; i < argc; i++) {
    
    dense_map::nvmData in1;
    dense_map::ReadNvm(argv[i],
                       in1.cid_to_keypoint_map,  
                       in1.cid_to_filename,  
                       in1.pid_to_cid_fid,  
                       in1.pid_to_xyz,  
                       in1.cid_to_cam_t_global);
    if (!FLAGS_no_shift) {
      bool undo_shift = true; // remove the shift relative to the optical center
      dense_map::shiftKeypoints(undo_shift, R, in1);
    }
    
    // TODO(oalexan1): Add flag to not have to transform second map, then use
    // this code to merge the theia nvm and produced nvm.
    dense_map::MergeMaps(in0, in1, R,
                         FLAGS_num_image_overlaps_at_endpoints,
                         FLAGS_fast_merge,
                         FLAGS_no_transform,
                         FLAGS_close_dist,
                         out_map);
    
    if (i + 1 < argc) {
      // There are more maps to marge. Let in0 be what we have so far,
      // and will merge onto it at the next iteration
      // TODO(oalexan1): Test this!
      in0 = out_map;
    }
  }

  if (FLAGS_fix_first_map) {
    // TODO(oalexan1): Make this work with N maps
    // Poses shared among the two input maps were averaged after
    // merging in out_map. Now, replace all poses in out_map which
    // exist in the first map with the originals from that map.
    std::map<std::string, int> in0_name_to_cid;
    for (size_t in0_cid = 0; in0_cid < in0.cid_to_filename.size(); in0_cid++)
      in0_name_to_cid[in0.cid_to_filename[in0_cid]] = in0_cid;
    
    for (size_t out_cid = 0; out_cid < out_map.cid_to_filename.size(); out_cid++) {
      auto pos = in0_name_to_cid.find(out_map.cid_to_filename[out_cid]);
      if (pos == in0_name_to_cid.end()) continue;  // was not in in0
      
      int in0_cid = pos->second;
      out_map.cid_to_cam_t_global[out_cid] = in0.cid_to_cam_t_global[in0_cid];
    }
  }

  if (!FLAGS_no_shift) {
    bool undo_shift = false; // put back the shift
    dense_map::shiftKeypoints(undo_shift, R, out_map);
  }
  
  // TODO(oalexan1): Throw out outliers!
  dense_map::WriteNvm(out_map.cid_to_keypoint_map,
                      out_map.cid_to_filename,
                      out_map.pid_to_cid_fid,
                      out_map.pid_to_xyz,
                      out_map.cid_to_cam_t_global,
                      FLAGS_output_map);

  // Save the optical offsets
  if (!FLAGS_no_shift)
    dense_map::writeOpticalCenters(FLAGS_output_map, out_map.cid_to_filename,
                                   R.cam_names, R.cam_params);
  
  return 0;
}

