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

#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
// Get rid of warning beyond our control
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic push
#include <openMVG/multiview/projection.hpp>
#include <openMVG/multiview/triangulation_nview.hpp>
#include <openMVG/numeric/numeric.h>
#include <openMVG/tracks/tracks.hpp>
#pragma GCC diagnostic pop

#include <rig_calibrator/essential.h>
#include <rig_calibrator/sparse_map.h>
#include <rig_calibrator/merge_maps.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <set>
#include <thread>
#include <vector>
#include <mutex>
namespace sparse_mapping {

}

