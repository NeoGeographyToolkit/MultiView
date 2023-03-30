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

// TODO(oalexan1): This file must be broken up into several files, for example,
// ba.cc, tracks.cc, interest_point.cc, triangulation.cc, sfm_merge.cc, etc.

#include <rig_calibrator/tensor.h>
#include <rig_calibrator/ransac.h>
#include <rig_calibrator/reprojection.h>
#include <rig_calibrator/sparse_mapping.h>
#include <rig_calibrator/sparse_map.h>
#include <rig_calibrator/essential.h>
#include <rig_calibrator/matching.h>
#include <rig_calibrator/transform_utils.h>
#include <rig_calibrator/basic_algs.h>
#include <rig_calibrator/thread.h>
#include <rig_calibrator/nvm.h>
#include <rig_calibrator/rig_config.h>
#include <rig_calibrator/camera_image.h>
#include <rig_calibrator/image_lookup.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
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

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sys/stat.h>
#include <fstream>
#include <set>
#include <thread>
#include <vector>
#include <mutex>
#include <functional>
#include <cstdio>

DEFINE_int32(min_valid, 20,
             "Minimum number of valid inlier matches required to keep matches for "
             "a given image pair.");

DEFINE_int32(max_pairwise_matches, 2000,
             "Maximum number of pairwise matches in an image pair to keep.");

DEFINE_int32(num_subsequent_images, std::numeric_limits<int32_t>::max()/2,  // avoid overflow
             "When no vocabulary tree is provided, match every image against this "
             "many subsequent images.");
DEFINE_int32(match_all_rate, -1,  // avoid overflow
             "If nonnegative, match one of every match_all_rate images to every other image.");
DEFINE_bool(skip_filtering, false,
            "Skip filtering of outliers after bundle adjustment.");
DEFINE_bool(skip_adding_new_matches_on_merging, false,
            "When merging maps, do not take advantage of performed matching to add new tracks.");
DEFINE_double(reproj_thresh, 5.0,
              "Filter points with re-projection error higher than this.");

// bundle adjustment phase parameters
DECLARE_int32(num_threads); // defined externally

DEFINE_int32(max_num_iterations, 1000,
             "Maximum number of iterations for bundle adjustment solver.");
DEFINE_int32(num_ba_passes, 5,
             "How many times to run bundle adjustment, removing outliers each time.");
DEFINE_string(cost_function, "Cauchy",
              "Choose a bundle adjustment cost function from: Cauchy, PseudoHuber, Huber, L1, L2.");
DEFINE_double(cost_function_threshold, 2.0,
              "Threshold to use with some cost functions, e.g., Cauchy.");
DEFINE_int32(first_ba_index, 0,
             "Vary only cameras starting with this index during bundle adjustment.");
DEFINE_int32(last_ba_index, std::numeric_limits<int>::max(),
             "Vary only cameras ending with this index during bundle adjustment.");
DEFINE_bool(silent_matching, false,
            "Do not print a lot of verbose info when matching.");

namespace sparse_mapping {
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

// Read matches from disk into OpenMVG's format
int ReadMatches(std::string const& matches_file,
                openMVG::matching::PairWiseMatches * match_map) {
  (*match_map).clear();
  std::ifstream imfile(matches_file.c_str());
  if (!imfile.good())
    LOG(FATAL) << "Could not read: " << matches_file << ". The matching step needs to be redone.";

  LOG(INFO) << "Reading: " << matches_file;
  std::string line;
  int num_matches = 0;
  while (std::getline(imfile, line)) {
    // replace '_' with ' '
    char * ptr = const_cast<char*>(line.c_str());
    for (size_t it = 0; it < line.size(); it++)
      if (ptr[it] == '_') ptr[it] = ' ';
    int i0, mi0, j0, mj0;
    if (sscanf(line.c_str(), "%d %d %d %d\n", &i0, &mi0, &j0, &mj0) != 4) continue;
    size_t i = i0, mi = mi0, j = j0, mj = mj0;
    std::pair<size_t, size_t> P(i, j);
    if ((*match_map).find(P) == (*match_map).end())
      (*match_map)[P] = std::vector<openMVG::matching::IndMatch>();
    openMVG::matching::IndMatch M(mi, mj);
    (*match_map)[P].push_back(M);
    num_matches++;
  }
  return num_matches;
}

void WriteMatches(openMVG::matching::PairWiseMatches const& match_map,
                  std::string const& matches_file) {
  // Save the matches to disk in the format: cidi_fidi cidj_fidj
  LOG(INFO) << "Writing: " << matches_file;
  std::ofstream mfile(matches_file.c_str());
  for (openMVG::matching::PairWiseMatches::const_iterator iter = match_map.begin();
       iter != match_map.end(); ++iter) {
    const size_t & I = iter->first.first;
    const size_t & J = iter->first.second;
    const std::vector<openMVG::matching::IndMatch> & matchVec = iter->second;
    // We have correspondences between I and J image indices
    for (size_t k = 0; k < matchVec.size(); ++k) {
      mfile << I << "_" << matchVec[k].i_ << " "
            << J << "_" << matchVec[k].j_ << std::endl;
    }
  }
  mfile.close();
}

// Filter the matches by a geometric constraint. Compute the essential matrix.
void BuildMapFindEssentialAndInliers(Eigen::Matrix2Xd const& keypoints1,
                                     Eigen::Matrix2Xd const& keypoints2,
                                     std::vector<cv::DMatch> const& matches,
                                     camera::CameraParameters const& camera_params,
                                     bool compute_inliers_only,
                                     size_t cam_a_idx, size_t cam_b_idx,
                                     std::mutex * match_mutex,
                                     CIDPairAffineMap * relative_affines,
                                     std::vector<cv::DMatch> * inlier_matches,
                                     bool compute_rays_angle,
                                     double * rays_angle) {
  // Initialize the outputs
  inlier_matches->clear();
  if (compute_rays_angle)
    *rays_angle = 0.0;

  int pt_count = matches.size();
  Eigen::MatrixXd observationsa(2, pt_count);
  Eigen::MatrixXd observationsb(2, pt_count);
  for (int i = 0; i < pt_count; i++) {
    observationsa.col(i) = keypoints1.col(matches[i].queryIdx);
    observationsb.col(i) = keypoints2.col(matches[i].trainIdx);
  }

  std::pair<size_t, size_t> image_size(camera_params.GetUndistortedSize()[0],
                                       camera_params.GetUndistortedSize()[1]);
  Eigen::Matrix3d k = camera_params.GetIntrinsicMatrix<camera::UNDISTORTED_C>();

  Eigen::Matrix3d e;
  // Calculate the essential matrix
  std::vector<size_t> vec_inliers;
  double error_max = std::numeric_limits<double>::max();
  double max_expected_error = 2.5;

  if (!interest_point::RobustEssential(k, k, observationsa, observationsb,
                                       &e, &vec_inliers,
                                       image_size, image_size,
                                       &error_max,
                                       max_expected_error)) {
    VLOG(2) << cam_a_idx << " " << cam_b_idx
            << " | Estimation of essential matrix failed!\n";
    return;
  }

  if (vec_inliers.size() < static_cast<size_t>(FLAGS_min_valid)) {
    VLOG(2) << cam_a_idx << " " << cam_b_idx
            << " | Failed to get enough inliers " << vec_inliers.size();
    return;
  }

  if (compute_inliers_only) {
    // We only need to know which interest points are inliers and not the
    // R and T matrices.
    int num_inliers = vec_inliers.size();
    inlier_matches->clear();
    inlier_matches->reserve(num_inliers);
    std::vector<Eigen::Matrix2Xd> observations2(2, Eigen::Matrix2Xd(2, num_inliers));
    for (int i = 0; i < num_inliers; i++) {
      inlier_matches->push_back(matches[vec_inliers[i]]);
    }
    return;
  }

  // Estimate the best possible R & T from the found Essential Matrix
  Eigen::Matrix3d r;
  Eigen::Vector3d t;
  if (!interest_point::EstimateRTFromE(k, k, observationsa, observationsb,
                                       e, vec_inliers,
                                       &r, &t)) {
    VLOG(2) << cam_a_idx << " " << cam_b_idx
            << " | Failed to extract RT from E";
    return;
  }

  VLOG(2) << cam_a_idx << " " << cam_b_idx << " | Inliers from E: "
          << vec_inliers.size() << " / " << observationsa.cols();

  // Get the observations corresponding to inliers
  // TODO(ZACK): We could reuse everything.
  int num_inliers = vec_inliers.size();
  std::vector<Eigen::Matrix2Xd> observations2(2, Eigen::Matrix2Xd(2, num_inliers));
  for (int i = 0; i < num_inliers; i++) {
    observations2[0].col(i) = observationsa.col(vec_inliers[i]);
    observations2[1].col(i) = observationsb.col(vec_inliers[i]);
  }

  // Refine the found T and R via bundle adjustment
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.max_num_iterations = 200;
  options.logging_type = ceres::SILENT;
  options.num_threads = FLAGS_num_threads;
  ceres::Solver::Summary summary;
  std::vector<Eigen::Affine3d> cameras(2);
  cameras[0].setIdentity();
  cameras[1].linear() = r;
  cameras[1].translation() = t;
  Eigen::Matrix3Xd pid_to_xyz(3, observations2[0].cols());
  double error;
  int num_pts_behind_camera = 0;
  for (ptrdiff_t i = 0; i < observations2[0].cols(); i++) {
    pid_to_xyz.col(i) =
      sparse_mapping::TriangulatePoint
      (Eigen::Vector3d(observations2[0](0, i), observations2[0](1, i),
                       camera_params.GetFocalLength()),
       Eigen::Vector3d(observations2[1](0, i), observations2[1](1, i),
                       camera_params.GetFocalLength()),
       r, t, &error);
    Eigen::Vector3d P = pid_to_xyz.col(i);
    Eigen::Vector3d Q = r*P + t;
    if (P[2] <= 0 || Q[2] <= 0) {
      num_pts_behind_camera++;
    }
  }
  VLOG(2) << "Pair " << cam_a_idx  << ' ' << cam_b_idx
          << ": number of points behind cameras: "
          << num_pts_behind_camera << "/" <<  observations2[0].cols()
          << " (" << round((100.0*num_pts_behind_camera) / observations2[0].cols())
          << "%)";

  sparse_mapping::BundleAdjustSmallSet(observations2, camera_params.GetFocalLength(), &cameras,
                                       &pid_to_xyz, new ceres::CauchyLoss(0.5), options,
                                       &summary);

  if (!summary.IsSolutionUsable()) {
    LOG(ERROR) << cam_a_idx << " " << cam_b_idx << " | Failed to refine RT with bundle adjustment";
    return;
  }
  VLOG(2) << summary.BriefReport();

  if (compute_rays_angle) {
    // Compute the median angle between rays.
    std::vector<double> angles;
    Eigen::Vector3d ctr0 = cameras[0].inverse().translation();
    Eigen::Vector3d ctr1 = cameras[1].inverse().translation();

    for (ptrdiff_t i = 0; i < observations2[0].cols(); i++) {
      Eigen::Vector3d P =
        sparse_mapping::TriangulatePoint
        (Eigen::Vector3d(observations2[0](0, i), observations2[0](1, i),
                         camera_params.GetFocalLength()),
         Eigen::Vector3d(observations2[1](0, i), observations2[1](1, i),
                         camera_params.GetFocalLength()),
         cameras[1].linear(), cameras[1].translation(), &error);

      Eigen::Vector3d X0 = ctr0 - P;
      Eigen::Vector3d X1 = ctr1 - P;
      double l0 = X0.norm(), l1 = X1.norm();
      double angle;
      // TODO(oalexan1): Integrate this code with the other angle computation
      // code.
      if (l0 == 0 || l1 == 0) {
        angle = 0.0;
      } else {
        double dot = X0.dot(X1)/l0/l1;
        dot = std::min(dot, 1.0);
        dot = std::max(-1.0, dot);
        angle = (180.0/M_PI)*acos(dot);
      }
      angles.push_back(angle);
    }
    // Median rays angle
    if (angles.size() >= static_cast<size_t>(2*FLAGS_min_valid))
      *rays_angle = angles[angles.size()/2];
  }

  // Give the solution
  Eigen::Affine3d result = cameras[1] * cameras[0].inverse();
  result.translation().normalize();

  // Must use a lock to protect this map shared among the threads
  CHECK(match_mutex) << "Forgot to provide the mutex lock.";
  CHECK(relative_affines) << "Forgot to provide relative_affines argument.";
  match_mutex->lock();
  relative_affines->insert(std::make_pair(std::make_pair(cam_a_idx, cam_b_idx),
                                        result));
  match_mutex->unlock();

  cv::Mat valid = cv::Mat::zeros(pt_count, 1, CV_8UC1);
  for (size_t i = 0; i < vec_inliers.size(); i++) {
    valid.at<uint8_t>(vec_inliers[i], 0) = 1;
  }

  // Count the number of inliers
  int32_t num_of_inliers =
    std::accumulate(valid.begin<uint8_t>(), valid.end<uint8_t>(), 0);

  // Keep about FLAGS_max_pairwise_matches inliers. This is to speed
  // up map generation so that we don't have to bother with a 1000
  // matches between consecutive images.
  if (FLAGS_max_pairwise_matches < num_of_inliers) {
    std::vector<double> dist;
    for (size_t query_index = 0; query_index < matches.size(); query_index++) {
      if (valid.at<uint8_t>(query_index, 0) > 0)
        dist.push_back(matches[query_index].distance);
    }
    std::sort(dist.begin(), dist.end());
    double max_dist = dist[FLAGS_max_pairwise_matches - 1];
    for (size_t query_index = 0; query_index < matches.size(); query_index++) {
      if (valid.at<uint8_t>(query_index, 0) > 0 &&
          matches[query_index].distance > max_dist) {
        valid.at<uint8_t>(query_index, 0) = 0;
      }
    }
    num_of_inliers
      = std::accumulate(valid.begin<uint8_t>(), valid.end<uint8_t>(), 0);
  }

  // Copy the inliers only
  inlier_matches->clear();
  inlier_matches->reserve(num_of_inliers);
  for (size_t m = 0; m < matches.size(); m++) {
    if (valid.at<uint8_t>(m, 0) > 0) {
      inlier_matches->push_back(matches[m]);
    }
  }
}

void BuildMapPerformMatching(openMVG::matching::PairWiseMatches * match_map,
                             std::vector<Eigen::Matrix2Xd > const& cid_to_keypoint_map,
                             std::vector<cv::Mat> const& cid_to_descriptor_map,
                             camera::CameraParameters const& camera_params,
                             CIDPairAffineMap * relative_affines,
                             std::mutex * match_mutex,
                             int i /*query cid index*/, int j /*train cid index*/,
                             bool compute_rays_angle, double * rays_angle) {
  Eigen::Matrix2Xd const& keypoints1 = cid_to_keypoint_map[i];
  Eigen::Matrix2Xd const& keypoints2 = cid_to_keypoint_map[j];

  std::vector<cv::DMatch> matches, inlier_matches;
  interest_point::FindMatches(cid_to_descriptor_map[i],
                              cid_to_descriptor_map[j],
                              &matches);

  // Do a check and verify that we meet our minimum before the
  // essential matrix fitting.
  if (static_cast<int32_t>(matches.size()) < FLAGS_min_valid) {
    if (!FLAGS_silent_matching) LOG(INFO) << i << " " << j << " | Failed to find enough matches " << matches.size();
    return;
  }

  bool compute_inliers_only = false;
  BuildMapFindEssentialAndInliers(keypoints1, keypoints2, matches,
                                  camera_params, compute_inliers_only,
                                  i, j,
                                  match_mutex,
                                  relative_affines,
                                  &inlier_matches,
                                  compute_rays_angle, rays_angle);

  if (static_cast<int32_t>(inlier_matches.size()) < FLAGS_min_valid) {
    if (!FLAGS_silent_matching)
      LOG(INFO) << i << " " << j << " | Failed to find enough inlier matches "
                << inlier_matches.size();
    return;
  }

  if (!FLAGS_silent_matching) LOG(INFO) << i << " " << j << " success " << inlier_matches.size();

  std::vector<openMVG::matching::IndMatch> mvg_matches;
  for (std::vector<cv::DMatch>::value_type const& match : inlier_matches)
    mvg_matches.push_back(openMVG::matching::IndMatch(match.queryIdx, match.trainIdx));
  match_mutex->lock();
  (*match_map)[ std::make_pair(i, j) ] = mvg_matches;
  match_mutex->unlock();
}

// Create the initial map by feature matching and essential affine computation.
// TODO(oalexan1): Remove the logic which requires saving intermediate results to disk.
// Keep it all in memory.
void MatchFeatures(const std::string & essential_file,
                   const std::string & matches_file,
                   sparse_mapping::SparseMap * s) {
  sparse_mapping::CIDPairAffineMap relative_affines;

  // Iterate through the cid pairings
  dense_map::ThreadPool thread_pool;
  std::mutex match_mutex;

  openMVG::matching::PairWiseMatches match_map;
  for (size_t cid = 0; cid < s->cid_to_keypoint_map_.size(); cid++) {
    std::vector<int> indices, queried_indices;
    // No matches in the db, or no db was provided.
    if ( s->cid_to_cid_.find(cid) != s->cid_to_cid_.end() ) {
      // See if perhaps we know which images to match to from a
      // previous map
      std::set<int> & matches = s->cid_to_cid_.find(cid)->second;
      for (auto it = matches.begin(); it != matches.end() ; it++) {
        indices.push_back(*it);
      }
    } else {
      // No way out, try matching brute force to subsequent images
      int subsequent = FLAGS_num_subsequent_images;
      if (FLAGS_match_all_rate > 0 && cid % FLAGS_match_all_rate == 0)
        subsequent = static_cast<int>(s->cid_to_keypoint_map_.size());
      int end = std::min(static_cast<int>(cid) + subsequent + 1,
                         static_cast<int>(s->cid_to_keypoint_map_.size()));
      for (int j = cid + 1; j < end; j++) {
        // Use subsequent images
        indices.push_back(j);
      }
    }

    bool compute_rays_angle = false;
    double rays_angle;
    for (size_t j = 0; j < indices.size(); j++) {
      // Need the check below for loop closing to pass in unit tests
      if (s->cid_to_filename_[cid] != s->cid_to_filename_[indices[j]]) {
        thread_pool.AddTask(&sparse_mapping::BuildMapPerformMatching,
                            &match_map,
                            s->cid_to_keypoint_map_,
                            s->cid_to_descriptor_map_,
                            std::cref(s->camera_params_),
                            &relative_affines,
                            &match_mutex,
                            cid, indices[j],
                            compute_rays_angle, &rays_angle);
      }
    }
  }
  thread_pool.Join();

  LOG(INFO) << "Number of affines found: " << relative_affines.size() << "\n";

  // Write the solution
  sparse_mapping::WriteAffineCSV(relative_affines, essential_file);

  WriteMatches(match_map, matches_file);

  // Initial cameras based on the affines (won't be used later,
  // just for visualization purposes).
  int num_images = s->cid_to_filename_.size();
  (s->cid_to_cam_t_global_).resize(num_images);
  (s->cid_to_cam_t_global_)[0].setIdentity();
  for (int cid = 1; cid < num_images; cid++) {
    std::pair<int, int> P(cid-1, cid);
    if (relative_affines.find(P) != relative_affines.end())
      (s->cid_to_cam_t_global_)[cid] = relative_affines[P]*(s->cid_to_cam_t_global_)[cid-1];
    else
      (s->cid_to_cam_t_global_)[cid] = (s->cid_to_cam_t_global_)[cid-1];  // no choice
  }
}

void BuildTracks(bool rm_invalid_xyz,
                 const std::string & matches_file,
                 sparse_mapping::SparseMap * s) {
  openMVG::matching::PairWiseMatches match_map;
  ReadMatches(matches_file, &match_map);

  // Build tracks using the interface tracksbuilder
  openMVG::tracks::TracksBuilder trackBuilder;
  trackBuilder.Build(match_map);  // Build:  Efficient fusion of correspondences
  trackBuilder.Filter();          // Filter: Remove tracks that have conflict
  // trackBuilder.ExportToStream(std::cout);
  openMVG::tracks::STLMAPTracks map_tracks;
  // Export tracks as a map (each entry is a sequence of imageId and featureIndex):
  //  {TrackIndex => {(imageIndex, featureIndex), ... ,(imageIndex, featureIndex)}
  trackBuilder.ExportToSTL(map_tracks);

  // TODO(oalexan1): Print how many pairwise matches were there before
  // and after filtering tracks.

  if (map_tracks.empty())
    LOG(FATAL) << "No tracks left after filtering. Perhaps images are too dis-similar?\n";

  size_t num_elems = map_tracks.size();
  // Populate back the filtered tracks.
  (s->pid_to_cid_fid_).clear();
  (s->pid_to_cid_fid_).resize(num_elems);
  size_t curr_id = 0;
  for (auto itr = map_tracks.begin(); itr != map_tracks.end(); itr++) {
    for (auto itr2 = (itr->second).begin(); itr2 != (itr->second).end(); itr2++) {
      (s->pid_to_cid_fid_)[curr_id][itr2->first] = itr2->second;
    }
    curr_id++;
  }

  // Triangulate. The results should be quite inaccurate, we'll redo this
  // later. This step is mostly for consistency.
  sparse_mapping::Triangulate(rm_invalid_xyz,
                              s->camera_params_.GetFocalLength(),
                              s->cid_to_cam_t_global_,
                              s->cid_to_keypoint_map_,
                              &(s->pid_to_cid_fid_),
                              &(s->pid_to_xyz_),
                              &(s->cid_fid_to_pid_));


  // Wipe file that is no longer needed
  try {
    std::remove(matches_file.c_str());
  }catch(...) {}

  // PrintTrackStats(s->pid_to_cid_fid_, "track building");
}

// TODO(oalexan1): This very naive code can use serious performance
// improvements.  Each time we add a new camera we triangulate all
// points. We bundle-adjust the last several cameras, but while seeing
// (and keeping fixed) all the earlier cameras. It is sufficient to
// both triangulate and see during bundle adjustment only the several
// most similar cameras. Fixing these would need careful testing for
// both map quality and run-time before and after the fix.
void IncrementalBA(std::string const& essential_file,
                   sparse_mapping::SparseMap * s) {
  // Do incremental bundle adjustment.

  // Optimize only the last several cameras, their number varies
  // between min_num_cams and max_num_cams.

  // TODO(oalexan1): Need to research how many previous cameras we
  // need for loop closure.
  int min_num_cams = 4;
  int max_num_cams = 128;

  // Read in all the affine R|t combinations between cameras
  sparse_mapping::CIDPairAffineMap relative_affines;
  sparse_mapping::ReadAffineCSV(essential_file,
                                &relative_affines);

  int num_images = s->cid_to_filename_.size();

  // Track and camera info up to the current cid
  std::vector<std::map<int, int>> pid_to_cid_fid_local;
  std::vector<Eigen::Affine3d> cid_to_cam_t_local;
  std::vector<Eigen::Vector3d> pid_to_xyz_local;
  std::vector<std::map<int, int>> cid_fid_to_pid_local;

  bool rm_invalid_xyz = true;

  for (int cid = 1; cid < num_images; cid++) {
    // The array of cameras so far including this one
    cid_to_cam_t_local.resize(cid + 1);
    for (int c = 0; c < cid; c++)
      cid_to_cam_t_local[c] = s->cid_to_cam_t_global_[c];

    // Add a new camera. Obtain it based on relative affines. Here we assume
    // the current camera is similar to the previous one.
    std::pair<int, int> P(cid-1, cid);
    if (relative_affines.find(P) != relative_affines.end())
      cid_to_cam_t_local[cid] = relative_affines[P]*cid_to_cam_t_local[cid-1];
    else
      cid_to_cam_t_local[cid] = cid_to_cam_t_local[cid-1];  // no choice

    // Restrict tracks to images up to cid.
    pid_to_cid_fid_local.clear();
    for (size_t p = 0; p < s->pid_to_cid_fid_.size(); p++) {
      std::map<int, int> & long_track = s->pid_to_cid_fid_[p];
      std::map<int, int> track;
      for (std::map<int, int>::iterator it = long_track.begin();
           it != long_track.end() ; it++) {
        if (it->first <= cid)
          track[it->first] = it->second;
      }

      // This is absolutely essential, using tracks of length >= 3
      // only greatly increases the reliability.
      if ( (cid == 1 && track.size() > 1) || track.size() > 2 )
        pid_to_cid_fid_local.push_back(track);
    }

    // Perform triangulation of all points. Multiview triangulation is
    // used.
    pid_to_xyz_local.clear();
    std::vector<std::map<int, int>> cid_fid_to_pid_local;
    sparse_mapping::Triangulate(rm_invalid_xyz,
                                s->camera_params_.GetFocalLength(),
                                cid_to_cam_t_local,
                                s->cid_to_keypoint_map_,
                                &pid_to_cid_fid_local,
                                &pid_to_xyz_local,
                                &cid_fid_to_pid_local);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.max_num_iterations = 500;
    options.logging_type = ceres::SILENT;
    options.num_threads = FLAGS_num_threads;
    ceres::Solver::Summary summary;
    ceres::LossFunction* loss = new ceres::CauchyLoss(0.5);

    // If cid+1 is divisible by 2^k, do at least 2^k cameras, ending
    // with camera cid.  E.g., if current camera index is 23 = 3*8-1, do at
    // least 8 cameras, so cameras 16, ..., 23. This way, we will try
    // to occasionally do more than just several close cameras.
    int val = cid+1;
    int offset = 1;
    while (val % 2 == 0) {
      val /= 2;
      offset *= 2;
    }
    offset = std::min(offset, max_num_cams);

    int start = cid-offset+1;
    start = std::min(cid-min_num_cams+1, start);
    if (start < 0) start = 0;

    LOG(INFO) << "Optimizing cameras from " << start << " to " << cid << " (total: "
        << cid-start+1 << ")";

    sparse_mapping::BundleAdjust(pid_to_cid_fid_local, s->cid_to_keypoint_map_,
                                 s->camera_params_.GetFocalLength(),
                                 &cid_to_cam_t_local, &pid_to_xyz_local,
                                 s->user_pid_to_cid_fid_,
                                 s->user_cid_to_keypoint_map_,
                                 &(s->user_pid_to_xyz_),
                                 loss, options, &summary,
                                 start, cid);

    // Copy back
    for (int c = 0; c <= cid; c++)
      s->cid_to_cam_t_global_[c] = cid_to_cam_t_local[c];
  }

  // Triangulate all points
  sparse_mapping::Triangulate(rm_invalid_xyz,
                              s->camera_params_.GetFocalLength(),
                              s->cid_to_cam_t_global_,
                              s->cid_to_keypoint_map_,
                              &(s->pid_to_cid_fid_),
                              &(s->pid_to_xyz_),
                              &(s->cid_fid_to_pid_));

  // Wipe file that is no longer needed
  try {
    std::remove(essential_file.c_str());
  }catch(...) {}
}

// Close loop after incremental BA
void CloseLoop(sparse_mapping::SparseMap * s) {
  // Consider a map with n images, where the first and the last image
  // are very similar. We would like to create a closed loop sequence
  // of cameras. To do that, after the last image we append the first several
  // images (parameter num_repeat_images), and do incremental BA.
  // We get a loop which ideally would perfectly overlap with itself at
  // the ends, but in practice does not. If the drift is small however,
  // in this function we identify repeated images and pids, and blend
  // the repeated camera positions and orientations. After this, the
  // user is meant to do another bundle adjustment step.

  // This process won't perform well if the drift is huge. In that case
  // the user is expected to redo the SFM calculations, perhaps
  // using better-positioned interest points, or or more of them.

  int num_images = s->cid_to_filename_.size();
  std::vector<std::string> & images = s->cid_to_filename_;  // alias
  std::map<std::string, int> image_map;

  // The first image to repeat is images[s1], it will also show
  // at position s2. The last image to repeat is image[e1],
  // it will also show at position e2.
  int s1 = -1, e1 = -1, s2 = -1, e2 = -1;
  bool has_repeat = false;
  for (int cid = 0; cid < num_images; cid++) {
    std::string image = images[cid];

    std::map<std::string, int>::iterator it = image_map.find(image);
    if (it == image_map.end()) {
      if (has_repeat) {
        // No more repeat images.
        break;
      }
      image_map[image] = cid;
      continue;
    }

    if (!has_repeat) {
      has_repeat = true;
      s1 = it->second;
      s2 = cid;
    }
    e1 = it->second;
    e2 = cid;
  }

  if (!has_repeat) {
    LOG(INFO) << "Could not find a loop to close";
    return;
  }

  // Sanity checks
  if (s1 < 0 || s2 < 0)
    LOG(FATAL) << "Could not find repeat images, failure in CloseLoop()";
  if (s2 - s1 != e2 - e1 || s1 >= e1)
    LOG(FATAL) << "Book-keeping failure in CloseLoop()";
  if (s1 != 0)
    LOG(FATAL) << "Situation not implemented in CloseLoop()";
  if (images[s1] != images[s2] || images[e1] != images[e2])
    LOG(FATAL) << "Expecting repeat images in CloseLoop().";

  // Blend the cameras. Start by giving full weight to the repeated
  // cameras, and gradually shifting the weight to the original
  // cameras.
  int pad = 0.1*(e1-s1);  // to make the blending a bit gentler
  double den = e1 - s1 - 2*pad;
  for (int cid = s1; cid <= e1; cid++) {
    int cid2 = cid - s1 + s2;
    double wt1 = (cid - s1 - pad)/den;
    if (wt1 < 0.0) wt1 = 0.0;
    if (wt1 > 1.0) wt1 = 1.0;
    double wt2 = 1.0 - wt1;

    // Blend rotations
    Eigen::Quaternion<double> q1(s->cid_to_cam_t_global_[cid].linear());
    Eigen::Quaternion<double> q2(s->cid_to_cam_t_global_[cid2].linear());
    Eigen::Quaternion<double> q = q2.slerp(wt1, q1);
    Eigen::Matrix3d R = q.toRotationMatrix();

    // Blend translations
    s->cid_to_cam_t_global_[cid].translation()
      = wt1*s->cid_to_cam_t_global_[cid].translation()
      + wt2*s->cid_to_cam_t_global_[cid2].translation();
    s->cid_to_cam_t_global_[cid].linear() = R;
  }

  // Merge the pids after identifying repeat images
  sparse_mapping::MergePids(e1, s2, &(s->pid_to_cid_fid_));

  // Wipe the now redundant info
  s->cid_to_filename_.resize(s2);
  s->cid_to_keypoint_map_.resize(s2);
  s->cid_to_cam_t_global_.resize(s2);
  s->cid_to_descriptor_map_.resize(s2);

  // sparse_mapping::PrintPidStats(s->pid_to_cid_fid_);
  bool rm_invalid_xyz = true;
  sparse_mapping::Triangulate(rm_invalid_xyz,
                              s->camera_params_.GetFocalLength(),
                              s->cid_to_cam_t_global_,
                              s->cid_to_keypoint_map_,
                              &(s->pid_to_cid_fid_),
                              &(s->pid_to_xyz_),
                              &(s->cid_fid_to_pid_));
}

void BundleAdjust(bool fix_all_cameras, sparse_mapping::SparseMap * map,
                  std::set<int> const& fixed_cameras) {
  for (int i = 0; i < FLAGS_num_ba_passes; i++) {
    LOG(INFO) << "Beginning bundle adjustment, pass: " << i << ".\n";

    // perform bundle adjustment
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::SPARSE_SCHUR; // Need to be building SuiteSparse
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    // What should the preconditioner be?
    options.num_threads = FLAGS_num_threads;
    options.max_num_iterations = FLAGS_max_num_iterations;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::LossFunction* loss = sparse_mapping::GetLossFunction(FLAGS_cost_function,
                                                                FLAGS_cost_function_threshold);
    sparse_mapping::BundleAdjustment(map, loss, options, &summary,
                                     FLAGS_first_ba_index, FLAGS_last_ba_index,
                                     fix_all_cameras, fixed_cameras);

    LOG(INFO) << summary.FullReport() << "\n";
    LOG(INFO) << "Starting average reprojection error: "
              << summary.initial_cost / map->GetNumObservations();
    LOG(INFO) << "Final average reprojection error:    "
              << summary.final_cost / map->GetNumObservations();
  }
}

void BundleAdjustment(sparse_mapping::SparseMap * s,
                      ceres::LossFunction* loss,
                      const ceres::Solver::Options & options,
                      ceres::Solver::Summary* summary,
                      int first, int last, bool fix_all_cameras,
                      std::set<int> const& fixed_cameras) {
  sparse_mapping::BundleAdjust(s->pid_to_cid_fid_, s->cid_to_keypoint_map_,
                               s->camera_params_.GetFocalLength(), &(s->cid_to_cam_t_global_),
                               &(s->pid_to_xyz_),
                               s->user_pid_to_cid_fid_, s->user_cid_to_keypoint_map_,
                               &(s->user_pid_to_xyz_),
                               loss, options, summary, first, last, fix_all_cameras,
                               fixed_cameras);

  // First do BA, and only afterwards remove outliers.
  if (!FLAGS_skip_filtering) {
    FilterPID(FLAGS_reproj_thresh,  s->camera_params_, s->cid_to_cam_t_global_,
              s->cid_to_keypoint_map_, &(s->pid_to_cid_fid_), &(s->pid_to_xyz_));
    s->InitializeCidFidToPid();
  }

  // sparse_mapping::PrintPidStats(s->pid_to_cid_fid_);
  // PrintTrackStats(s->pid_to_cid_fid_, "bundle adjustment and filtering");
}
  
// Extract a submap in-place.
void ExtractSubmap(std::vector<std::string> const& images_to_keep,
                   dense_map::nvmData & nvm) {

  // Sanity check. The images to keep must exist in the original map.
  std::map<std::string, int> image2cid;
  for (size_t cid = 0; cid < nvm.cid_to_filename.size(); cid++)
    image2cid[nvm.cid_to_filename[cid]] = cid;
  for (size_t cid = 0; cid < images_to_keep.size(); cid++) {
    if (image2cid.find(images_to_keep[cid]) == image2cid.end())
      std::cout << "Warning: Could not find in the input map the image: "
                << images_to_keep[cid] << "\n";
  }

  // To extract the submap-in place, it is simpler to reorder the images
  // to extract to be in the same order as in the map. Keep those in
  // local vector 'keep'.
  std::vector<std::string> keep;
  {
    std::set<std::string> keep_set;
    for (size_t cid = 0; cid < images_to_keep.size(); cid++)
      keep_set.insert(images_to_keep[cid]);
    for (size_t cid = 0; cid < nvm.cid_to_filename.size(); cid++) {
      if (keep_set.find(nvm.cid_to_filename[cid]) != keep_set.end())
        keep.push_back(nvm.cid_to_filename[cid]);
    }
  }

  // Map each image we keep to its index
  std::map<std::string, int> keep2cid;
  for (size_t cid = 0; cid < keep.size(); cid++)
    keep2cid[keep[cid]] = cid;

  // The map from the old cid to the new cid
  std::map<int, int> cid2cid;
  for (size_t cid = 0; cid < nvm.cid_to_filename.size(); cid++) {
    auto it = keep2cid.find(nvm.cid_to_filename[cid]);
    if (it == keep2cid.end()) continue;  // current image is not in the final submap
    cid2cid[cid] = it->second;
  }

  // Sanity checks. All the kept images must be represented in cid2cid,
  // and the values in cid2cid must be consecutive.
  if (cid2cid.size() != keep.size() || cid2cid.empty())
    LOG(FATAL) << "Cannot extract a submap. Check your inputs. Maybe some images "
               << "are duplicated or none are in the map.";
  for (auto it = cid2cid.begin(); it != cid2cid.end(); it++) {
    auto it2 = it; it2++;
    if (it2 == cid2cid.end()) continue;
    if (it->second + 1 != it2->second || cid2cid.begin()->second != 0 )
      LOG(FATAL) << "Cannot extract a submap. Check if the images "
                 << "you want to keep are in the same order as in the original map.";
  }

  // Over-write the data in-place. Should be safe with the checks done above.
  int num_cid = keep.size();
  for (size_t cid = 0; cid < nvm.cid_to_filename.size(); cid++) {
    if (cid2cid.find(cid) == cid2cid.end()) continue;
    size_t new_cid = cid2cid[cid];
    nvm.cid_to_filename[new_cid]     = nvm.cid_to_filename[cid];
    nvm.cid_to_keypoint_map[new_cid] = nvm.cid_to_keypoint_map[cid];
    nvm.cid_to_cam_t_global[new_cid] = nvm.cid_to_cam_t_global[cid];
  }
  nvm.cid_to_filename.resize(num_cid);
  nvm.cid_to_keypoint_map.resize(num_cid);
  nvm.cid_to_cam_t_global.resize(num_cid);

  // Create new pid_to_cid_fid and pid_to_xyz.
  std::vector<std::map<int, int>> pid_to_cid_fid;
  std::vector<Eigen::Vector3d> pid_to_xyz;
  for (size_t pid = 0; pid < nvm.pid_to_cid_fid.size(); pid++) {
    auto const& cid_fid = nvm.pid_to_cid_fid[pid];  // alias
    std::map<int, int> cid_fid2;
    for (auto it = cid_fid.begin(); it != cid_fid.end(); it++) {
      int cid = it->first;
      if (cid2cid.find(cid) == cid2cid.end()) continue;  // not an image we want to keep
      cid_fid2[cid2cid[cid]] = it->second; // fid does not change
    }
    if (cid_fid2.size() <= 1) continue;  // tracks must have size at least 2
    pid_to_cid_fid.push_back(cid_fid2);
    pid_to_xyz.push_back(nvm.pid_to_xyz[pid]);
  }
  nvm.pid_to_cid_fid = pid_to_cid_fid;
  nvm.pid_to_xyz = pid_to_xyz;

  std::cout << "Number of images in the extracted map: " << nvm.cid_to_filename.size() << "\n";
  std::cout << "Number of tracks in the extracted map: " << nvm.pid_to_cid_fid.size() << "\n";

  return;
}

// This fitting functor attempts to find a rotation + translation + scale transformation
// between two vectors of points.
struct TranslationRotationScaleFittingFunctor {
  typedef Eigen::Affine3d result_type;

  /// A transformation requires 3 inputs and 3 outputs to make a fit.
  size_t min_elements_needed_for_fit() const { return 3; }

  result_type operator() (std::vector<Eigen::Vector3d> const& in_vec,
                          std::vector<Eigen::Vector3d> const& out_vec) const {
    // check consistency
    if (in_vec.size() != out_vec.size())
      LOG(FATAL) << "There must be as many inputs as outputs to be "
                 << "able to compute a transform between them.\n";
    if (in_vec.size() < min_elements_needed_for_fit())
      LOG(FATAL) << "Cannot compute a transformation. Insufficient data.\n";

    Eigen::Matrix3Xd in_mat  = Eigen::MatrixXd(3, in_vec.size());
    Eigen::Matrix3Xd out_mat = Eigen::MatrixXd(3, in_vec.size());
    for (size_t it = 0; it < in_vec.size(); it++) {
      in_mat.col(it)  = in_vec[it];
      out_mat.col(it) = out_vec[it];
    }
    result_type out_trans;
    dense_map::Find3DAffineTransform(in_mat, out_mat, &out_trans);
    return out_trans;
  }
};

// How well does the given transform do to map p1 to p2.
struct TransformError {
  double operator() (Eigen::Affine3d const& T, Eigen::Vector3d const& p1,
                     Eigen::Vector3d const& p2) const {
    return (T*p1 - p2).norm();
  }
};

// Given a set of points in 3D, heuristically estimate what it means
// for two points to be "not far" from each other. The logic is to
// find a bounding box of an inner cluster and multiply that by 0.1.
double estimateCloseDistance(std::vector<Eigen::Vector3d> const& vec) {
  Eigen::Vector3d range;
  int num_pts = vec.size();
  if (num_pts <= 0)
    LOG(FATAL) << "Empty set of points.\n";  // to avoid a segfault

  std::vector<double> vals(num_pts);
  for (int it = 0; it < range.size(); it++) {  // iterate in each coordinate
    // Sort all values in given coordinate
    for (int p = 0; p < num_pts; p++)
      vals[p] = vec[p][it];
    std::sort(vals.begin(), vals.end());

    // Find some percentiles
    int min_p = round(num_pts*0.25);
    int max_p = round(num_pts*0.75);
    if (min_p >= num_pts) min_p = num_pts - 1;
    if (max_p >= num_pts) max_p = num_pts - 1;
    double min_val = vals[min_p], max_val = vals[max_p];
    range[it] = 0.1*(max_val - min_val);
  }

  // Find the average of all ranges
  double range_val = 0.0;
  for (int it = 0; it < range.size(); it++)
    range_val += range[it];
  range_val /= range.size();

  return range_val;
}

// Given a transform from cid2cid from some cid values to some others,
// apply the same transform to the tracks. This may make the tracks
// shorter if cid2cid maps different inputs to the same output.
// New tracks of length 1 can be excluded if desired.
void TransformTracks(std::map<int, int> const& cid2cid,
                     bool rm_tracks_of_len_one,
                     std::vector<std::map<int, int>> * pid_to_cid_fid) {
  std::vector<std::map<int, int>> pid_to_cid_fid2;
  for (size_t pid = 0; pid < (*pid_to_cid_fid).size(); pid++) {
    auto & cid_fid = (*pid_to_cid_fid)[pid];  // alias
    std::map<int, int> cid_fid2;
    for (auto it = cid_fid.begin(); it != cid_fid.end(); it++) {
      int cid = it->first;
      int fid = it->second;
      auto cid_it = cid2cid.find(cid);
      if (cid_it == cid2cid.end()) continue;
      int out_cid = cid_it->second;
      cid_fid2[out_cid] = fid;
    }

    bool will_skip = rm_tracks_of_len_one && (cid_fid2.size() <= 1);
    if (!will_skip)
      pid_to_cid_fid2.push_back(cid_fid2);
  }
  *pid_to_cid_fid = pid_to_cid_fid2;
}

void PrintTrackStats(std::vector<std::map<int, int>>const& pid_to_cid_fid,
                       std::string const& step) {
  LOG(INFO) << "Track statistics after: " << step;

  double track_len = 0.0;
  for (size_t pid = 0; pid < pid_to_cid_fid.size(); pid++)
    track_len += pid_to_cid_fid[pid].size();
  double avg_len = track_len / pid_to_cid_fid.size();

  LOG(INFO) << "Number of tracks (points in the control network): " << pid_to_cid_fid.size();
  LOG(INFO) << "Total length of all tracks: " << track_len;
  LOG(INFO) << "Average track length: " << avg_len;

  std::map<int, int> stats;
  for (size_t pid = 0; pid < pid_to_cid_fid.size(); pid++)
    stats[pid_to_cid_fid[pid].size()]++;
  for (std::map<int, int>::const_iterator it = stats.begin(); it != stats.end() ; it++) {
    LOG(INFO) << "Track length and their number: "
              << it->first << ' ' << it->second;
  }
}

// I/O Functions
template <class IterT>
void WriteCIDPairAffineIterator(IterT it,
                                IterT end,
                                std::ofstream* file) {
  Eigen::IOFormat fmt(Eigen::FullPrecision, 0, " ", "\n", "", "", "", "");
  while (it != end) {
    *file << it->first.first << " " << it->first.second << std::endl;
    *file << it->second.affine().format(fmt) << std::endl;
    it++;
  }
}

template <class IterT>
void ReadAffine(std::ifstream* file,
                IterT output_iter) {
  std::string line[4];
  std::getline(*file, line[0]);
  std::getline(*file, line[1]);
  std::getline(*file, line[2]);
  std::getline(*file, line[3]);
  if (line[0].empty())
    return;

  int i, j;
  Eigen::Matrix3d r;
  Eigen::Vector3d t;
  {
    std::stringstream ss(line[0]);
    ss >> i >> j;
  }

  for (int k = 0; k < 3; k++) {
    std::stringstream ss(line[k + 1]);
    ss >> r(k, 0) >> r(k, 1) >> r(k, 2) >> t[k];
  }

  Eigen::Affine3d affine;
  affine.linear() = r;
  affine.translation() = t;
  *output_iter = std::make_pair(std::make_pair(i, j),
                                affine);
}

// Use a back inserter with this if you haven't previously allocated enough space.
template <class IterT>
void PushBackCIDPairAffine(std::ifstream* file,
                           IterT output_iter,
                           IterT output_iter_end) {
  do {
    ReadAffine(file, output_iter);
    output_iter++;
  } while (file->good() && output_iter != output_iter_end);
}

template <class IterT>
void PushBackCIDPairAffine(std::ifstream* file,
                           IterT iter) {
  do {
    ReadAffine(file, iter);
    iter++;
  } while (file->good());
}

void WriteAffineCSV(CIDPairAffineMap const& relative_affines,
                    std::string const& output_filename) {
  LOG(INFO) << "Writing: " << output_filename;
  std::ofstream f(output_filename, std::ofstream::out);
  WriteCIDPairAffineIterator(relative_affines.begin(),
                             relative_affines.end(),
                             &f);
  f.close();
}
void WriteAffineCSV(CIDAffineTupleVec const& relative_affines,
                    std::string const& output_filename) {
  LOG(INFO) << "Writing: " << output_filename;
  std::ofstream f(output_filename, std::ofstream::out);
  for (CIDAffineTupleVec::value_type const& tuple : relative_affines) {
    f << "Tuple:" << std::endl;
    WriteCIDPairAffineIterator(tuple.begin(), tuple.end(), &f);
  }
  f.close();
}
void ReadAffineCSV(std::string const& input_filename,
                   CIDPairAffineMap* relative_affines) {
  LOG(INFO) << "Reading: " << input_filename;
  std::ifstream f(input_filename, std::ifstream::in);
  if (!f.good())
    LOG(FATAL) << "Could no read: " << input_filename << ". Must redo the matching step.";
  relative_affines->clear();
  PushBackCIDPairAffine(&f, std::inserter(*relative_affines, relative_affines->begin()));
  f.close();
}
void ReadAffineCSV(std::string const& input_filename,
                   CIDAffineTupleVec* relative_affines) {
  std::ifstream f(input_filename, std::ifstream::in);
  if (!f.good())
    LOG(FATAL) << "Could no read: " << input_filename << ". Must redo the matching step.";
  relative_affines->clear();
  std::string line;
  std::getline(f, line);
  while (!line.empty()) {
    relative_affines->push_back({});
    PushBackCIDPairAffine(&f, relative_affines->back().begin(),
                          relative_affines->back().end());
    std::getline(f, line);
  }
  f.close();
}

void Triangulate(bool rm_invalid_xyz, double focal_length,
                 std::vector<Eigen::Affine3d> const& cid_to_cam_t_global,
                 std::vector<Eigen::Matrix2Xd> const& cid_to_keypoint_map,
                 std::vector<std::map<int, int>> * pid_to_cid_fid,
                 std::vector<Eigen::Vector3d> * pid_to_xyz,
                 std::vector<std::map<int, int>> * cid_fid_to_pid) {
  Eigen::Matrix3d k;
  k << focal_length, 0, 0,
    0, focal_length, 0,
    0, 0, 1;

  // Build p matrices for all of the cameras. openMVG::Triangulation
  // will be holding pointers to all of the cameras.
  std::vector<openMVG::Mat34> cid_to_p(cid_to_cam_t_global.size());
  for (size_t cid = 0; cid < cid_to_p.size(); cid++) {
    openMVG::P_From_KRt(k, cid_to_cam_t_global[cid].linear(),
                        cid_to_cam_t_global[cid].translation(), &cid_to_p[cid]);
  }

  pid_to_xyz->resize(pid_to_cid_fid->size());
  for (int pid = pid_to_cid_fid->size() - 1; pid >= 0; pid--) {
    openMVG::Triangulation tri;
    for (std::pair<int, int> const& cid_fid : pid_to_cid_fid->at(pid)) {
      tri.add(cid_to_p[cid_fid.first],  // they're holding a pointer to this
              cid_to_keypoint_map[cid_fid.first].col(cid_fid.second));
    }
    Eigen::Vector3d solution = tri.compute();
    if ( rm_invalid_xyz && (std::isnan(solution[0]) || tri.minDepth() < 0) ) {
      pid_to_xyz->erase(pid_to_xyz->begin() + pid);
      pid_to_cid_fid->erase(pid_to_cid_fid->begin() + pid);
    } else {
      pid_to_xyz->at(pid) = solution;
    }
  }

  // Must always keep the book-keeping correct
  sparse_mapping::InitializeCidFidToPid(cid_to_cam_t_global.size(),
                                        *pid_to_cid_fid,
                                        cid_fid_to_pid);
}
  
// As result of matching some images in A to some images in B, we must
// now merge some tracks in A with some tracks in B, as those tracks
// correspond physically to the same point in space. A track in
// C.pid_to_cid_fid_ tells us which track in A.pid_to_cid_fid_ is tied
// with which track in B.pid_to_cid_fid_. If it turns out one track in
// A should be merged with multiple tracks in B or vice-versa, select
// just one candidate from each map, based on who got most votes. Note
// that here it is easier to work with A.cid_fid_to_pid_ rather than
// A.pid_to_cid_fid_.
void FindPidCorrespondences(std::vector<std::map<int, int>> const& A_cid_fid_to_pid,
                            std::vector<std::map<int, int>> const& B_cid_fid_to_pid,
                            std::vector<std::map<int, int>> const& C_pid_to_cid_fid,
                            int num_acid,  // How many images are in A
                            std::map<int, int> * A2B, std::map<int, int> * B2A) {
  A2B->clear();
  B2A->clear();

  std::map<int, std::map<int, int>> VoteMap;
  for (int pid = 0; pid < static_cast<int>(C_pid_to_cid_fid.size()); pid++) {
    // This track has some cid indices from A (those < num_acid)
    // and some from B (those >= num_acid). Ignore all other combinations.
    auto const& cid_fid_c = C_pid_to_cid_fid[pid];  // alias
    for (auto it_a = cid_fid_c.begin(); it_a != cid_fid_c.end(); it_a++) {
      for (auto it_b = it_a; it_b != cid_fid_c.end(); it_b++) {
        int cid_a = it_a->first, fid_a = it_a->second;
        int cid_b = it_b->first, fid_b = it_b->second;
        if (cid_a >= num_acid) continue;
        if (cid_b <  num_acid) continue;

        // Subtract num_acid from cid_b so it becomes a cid in B.
        cid_b -= num_acid;

        auto it_fida = A_cid_fid_to_pid[cid_a].find(fid_a);
        if (it_fida == A_cid_fid_to_pid[cid_a].end()) continue;

        auto it_fidb = B_cid_fid_to_pid[cid_b].find(fid_b);
        if (it_fidb == B_cid_fid_to_pid[cid_b].end()) continue;

        int pid_a = it_fida->second;
        int pid_b = it_fidb->second;

        VoteMap[pid_a][pid_b]++;
      }
    }
  }

  // For each pid in A, keep the pid in B with most votes
  std::map<int, std::map<int, int>> B2A_Version0;  // still not fully one-to-one
  for (auto it_a = VoteMap.begin(); it_a != VoteMap.end(); it_a++) {
    auto & M = it_a->second;  // all pid_b corresp to given pid_a with their votes
    int pid_a = it_a->first;
    int best_pid_b = -1;
    int max_vote = -1;
    for (auto it_b = M.begin(); it_b != M.end(); it_b++) {
      int pid_b = it_b->first;
      int vote = it_b->second;
      if (vote > max_vote) {
        best_pid_b = pid_b;
        max_vote = vote;
      }
    }
    B2A_Version0[best_pid_b][pid_a] = max_vote;
  }

  // And vice-versa
  for (auto it_b = B2A_Version0.begin(); it_b != B2A_Version0.end(); it_b++) {
    int pid_b = it_b->first;
    auto & M = it_b->second;
    int best_pid_a = -1;
    int max_vote = -1;
    for (auto it_a = M.begin(); it_a != M.end(); it_a++) {
      int pid_a = it_a->first;
      int vote = it_a->second;
      if (vote > max_vote) {
        best_pid_a = pid_a;
        max_vote = vote;
      }
    }

    (*A2B)[best_pid_a] = pid_b;  // track from A and track from B
    (*B2A)[pid_b] = best_pid_a;  // track from B and track from A
  }
}

// TODO(oalexan1): Move this to basic_algs.h.
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

// Merge the camera poses from both maps using the cid2cid map of
// indices. By now the poses are in the same coordinate system but
// some show up in both maps.
void MergePoses(std::map<int, int> & cid2cid,
                std::vector<Eigen::Affine3d> & cid_to_cam_t_global) {
    
  // The total number of output cameras (last new cid value + 1)
  int num_out_cams = dense_map::maxMapVal(cid2cid) + 1;
  
  // Each blob will be original cids that end up being a single cid
  // after identifying repeat images.
  std::vector<std::set<int>> blobs(num_out_cams);
  for (size_t cid = 0; cid < cid_to_cam_t_global.size(); cid++)
    blobs[cid2cid[cid]].insert(cid);

  // To merge cid_to_cam_t_global, find the average rotation and translation
  // from the two maps.
  std::vector<Eigen::Affine3d> cid_to_cam_t_global2(num_out_cams);
  for (size_t c = 0; c < blobs.size(); c++) {
    if (blobs[c].size() == 1) {
      cid_to_cam_t_global2[c] = cid_to_cam_t_global[*blobs[c].begin()];
    } else {
      int num = blobs[c].size();

      // All cams to merge get equal weight
      std::vector<double> W(num, 1.0/num);

      std::vector<Eigen::Quaternion<double>> Q(num);
      cid_to_cam_t_global2[c].translation() << 0.0, 0.0, 0.0;
      int pos = -1;
      for (auto it = blobs[c].begin(); it != blobs[c].end(); it++) {
        pos++;
        int cid = *it;
        Q[pos] = Eigen::Quaternion<double>(cid_to_cam_t_global[cid].linear());

        cid_to_cam_t_global2[c].translation()
          += W[pos]*cid_to_cam_t_global[cid].translation();
      }
      Eigen::Quaternion<double> S = sparse_mapping::slerp_n(W, Q);
      cid_to_cam_t_global2[c].linear() = S.toRotationMatrix();
    }
  }

  // Return the updated poses
  cid_to_cam_t_global = cid_to_cam_t_global2;
  
  return;
}

// If two maps share images, can match tracks between the maps
// just based on that, which is fast.
void findTracksForSharedImages(sparse_mapping::SparseMap * A_in,
                               sparse_mapping::SparseMap * B_in,
                               // Outputs
                               std::map<int, int> & A2B,
                               std::map<int, int> & B2A) {
  // Wipe the outputs
  A2B.clear();
  B2A.clear();

  // Create aliases to not use pointers all the time.
  sparse_mapping::SparseMap & A = *A_in;
  sparse_mapping::SparseMap & B = *B_in;

  size_t num_acid = A.cid_to_filename_.size();
  size_t num_bcid = B.cid_to_filename_.size();

  // Map from file name to cid
  std::map<std::string, int> A_file_to_cid, B_file_to_cid;
  for (size_t cid = 0; cid < num_acid; cid++)
    A_file_to_cid[A.cid_to_filename_[cid]] = cid;
  for (size_t cid = 0; cid < num_bcid; cid++)
    B_file_to_cid[B.cid_to_filename_[cid]] = cid;

  // Iterate through A's cid_fid_to_pid_ and find matches in B.
  int num_shared_cid = 0;
  for (size_t cid_a = 0; cid_a < A.cid_fid_to_pid_.size(); cid_a++) {
    std::string filename = A.cid_to_filename_[cid_a];
    auto it = B_file_to_cid.find(filename);
    if (it == B_file_to_cid.end())
      continue;

    num_shared_cid++;

    // The corresponding camera id in the second map
    size_t cid_b = it->second;

    if (A.cid_to_keypoint_map_[cid_a] != B.cid_to_keypoint_map_[cid_b])
      LOG(FATAL) << "The input maps don't have the same features. "
                 << "They need to be rebuilt.";

    auto a_fid_to_pid = A.cid_fid_to_pid_[cid_a];
    auto b_fid_to_pid = B.cid_fid_to_pid_[cid_b];

    // Find tracks corresponding to same cid_fid
    for (auto it_a = a_fid_to_pid.begin(); it_a != a_fid_to_pid.end(); it_a++) {
      int pid_a = it_a->second;
      int fid = it_a->first;  // shared fid
      auto it_b = b_fid_to_pid.find(fid);
      if (it_b == b_fid_to_pid.end()) {
        // This fid is not in second image. This is fine. A feature in a current image
        // may match to features in one image but not in another.
        continue;
      }

      int pid_b = it_b->second;

      A2B[pid_a] = pid_b;
    }
  }

  // Now create B2A
  for (auto it = A2B.begin(); it != A2B.end(); it++) {
    B2A[it->second] = it->first;
  }

  // Just in case, recreate A2B, to avoid issues when the original
  // A2B mapped multiple A pids to same B pid.
  A2B.clear();
  for (auto it = B2A.begin(); it != B2A.end(); it++) {
    A2B[it->second] = it->first;
  }


  LOG(INFO) << "Number of shared images in the two maps: "
            << num_shared_cid << std::endl;
  LOG(INFO) << "Number of shared tracks: " << A2B.size() << std::endl;

  // Sanity check
  if (num_shared_cid <= 0 || A2B.size() <= 5)
    LOG(FATAL) << "Not enough shared images or features among the two maps. "
               << "Run without the --fast option.";
}

// Choose the images to match and load them. It is assumed that in image_files
// we have the images from the first and then he second maps to merge.
void setupLoadMatchingImages(std::vector<std::string> const& image_files,
                             dense_map::RigSet const& R,
                             int map1_len, int map2_len,
                             int num_image_overlaps_at_endpoints,
                             // Outputs
                             std::vector<std::pair<int, int>> & image_pairs,
                             std::vector<dense_map::cameraImage> & cams) {

  // sanity check
  if (map1_len + map2_len != image_files.size()) 
    LOG(FATAL) << "Book-keeping error, total number of images is not right.\n";
  
  // Initialize the outputs
  image_pairs.clear();
  cams.resize(image_files.size());
  
  std::set<int> map1_search, map2_search;  // use sets to avoid duplicates
  int num = num_image_overlaps_at_endpoints;

  // Images in map1 to search for matches in map2
  for (int cid = 0; cid < num; cid++)
    if (cid < map1_len) map1_search.insert(cid);
  for (int cid = map1_len-num; cid < map1_len; cid++)
    if (cid >= 0) map1_search.insert(cid);

  // Images in map2 to search for matches in map1. Add map1_len since we will
  // match map1 and map2 inside of the merged map.
  for (int cid = 0; cid < num; cid++)
    if (cid < map2_len) map2_search.insert(map1_len + cid);
  for (int cid = map2_len-num; cid < map2_len; cid++)
    if (cid >= 0) map2_search.insert(map1_len + cid);

  // The indices in the merged map between which we need matches. Do not match
  // an image with itself. That can happen if the maps to merge
  // have shared images.
  for (auto it1 = map1_search.begin(); it1 != map1_search.end() ; it1++) {
    for (auto it2 = map2_search.begin(); it2 != map2_search.end(); it2++) {
      if (*it1 == *it2)
        LOG(FATAL) << "Book-keeping failure in map merging.";
      if (image_files[*it1] == image_files[*it2])
        continue;
      image_pairs.push_back(std::make_pair(*it1, *it2));
    }
  }

  // Allocate a structure having an entry for all images, but load
  // only those for which we need to find matches.
  if (!map1_search.empty() && !map2_search.empty()) 
    std::cout << "Loading images to match." << std::endl;
  
  for (size_t cid = 0; cid < image_files.size(); cid++) {
    auto & c = cams[cid]; // alias
    // Populate most fields. All we need is the image data and camera type.
    c.image_name = image_files[cid];
    dense_map::findCamTypeAndTimestamp(c.image_name,  
                                       R.cam_names,  
                                       // Outputs 
                                       c.camera_type, c.timestamp);
    if (map1_search.find(cid) != map1_search.end() ||
        map2_search.find(cid) != map2_search.end()) {
      std::cout << "Loading image: " << c.image_name << std::endl;
      c.image = cv::imread(c.image_name, cv::IMREAD_GRAYSCALE);
    }
  }
}

// Compute the transform from the B map to the A map by finding the median
// transform based on the shared images
Eigen::Affine3d computeTransformFromBToA(const dense_map::nvmData& A,
                                         const dense_map::nvmData& B) {
  // Calc all transforms from B poses to A poses
  std::vector<Eigen::MatrixXd> B2A_vec;
  
  // Put the B poses in a map
  std::map<std::string, Eigen::Affine3d> B_world_to_cam;
  for (size_t cid = 0; cid < B.cid_to_cam_t_global.size(); cid++)
    B_world_to_cam[B.cid_to_filename[cid]] = B.cid_to_cam_t_global[cid];
  
  // Find the transform from B to A based on shared poses
  for (size_t cid = 0; cid < A.cid_to_filename.size(); cid++) {
    auto b_it = B_world_to_cam.find(A.cid_to_filename[cid]);
    if (b_it == B_world_to_cam.end()) 
      continue;
    
    auto const& A_world_to_cam = A.cid_to_cam_t_global[cid];
    auto const& B_world_to_cam = b_it->second;
    
    // Go from world of B to world of A
    B2A_vec.push_back( ((A_world_to_cam.inverse()) * B_world_to_cam).matrix() );
  }

  // Find the median transform, for robustness
  Eigen::Affine3d B2A_trans;
  B2A_trans.matrix() = dense_map::median_matrix(B2A_vec);
  
  return B2A_trans;
}
  
// Merge two maps. See sfm_merge.cc. Approach: Find matches among
// several images in map A and several in map B, based on
// num_image_overlaps_at_endpoints. Then build tracks (so merging the
// pairwise matches into tracks). If a track is partially in A and
// partially in B, (with at least two features in each), that makes it
// possible to find a triangulated point in A and one in B for that
// track. Doing RANSAC between them will find the transform from B to
// A. Then merge the transformed poses, remove the repeated images,
// and concatenate and remove repetitions from tracks.  If the
// -fast_merge flag is used, find the transform between the maps using
// shared poses.  It is assumed features in keypoint maps are not
// shifted relative to the optical center. The caller is responsible
// to ensure that.
// TODO(oalexan1): Modularize and move to some new tracks.cc file,
// together with other logic from interest_point.cc.
void MergeMaps(dense_map::nvmData const& A,
               dense_map::nvmData const& B,
               dense_map::RigSet const& R,
               int num_image_overlaps_at_endpoints,
               bool fast_merge,
               double close_dist,
               dense_map::nvmData & C) { // output merged map

  // Wipe the output
  C = dense_map::nvmData();

  if (fast_merge && num_image_overlaps_at_endpoints > 0) {
    std::cout << "Setting number of image overlaps at end points to zero, "
              << "as fast merging is used.\n";
    num_image_overlaps_at_endpoints = 0;
  }
  
  // Merge things that make sense to merge and are easy to do. Later
  // some of these will be shrunk if the input maps have shared data.
  int num_acid = A.cid_to_filename.size();
  int num_bcid = B.cid_to_filename.size();
  int num_ccid = num_acid + num_bcid;

  // Concatenate the images from A and B into C.
  C.cid_to_filename.clear();
  C.cid_to_filename.insert(C.cid_to_filename.end(),
                           A.cid_to_filename.begin(), A.cid_to_filename.end());
  C.cid_to_filename.insert(C.cid_to_filename.end(),
                           B.cid_to_filename.begin(), B.cid_to_filename.end());

  std::vector<dense_map::cameraImage> cams;
  std::vector<std::pair<int, int>> image_pairs;
  setupLoadMatchingImages(C.cid_to_filename, R,  
                          num_acid, num_bcid,  
                          num_image_overlaps_at_endpoints,  
                          image_pairs, cams); // Outputs
  
  Eigen::Affine3d B2A_trans;
  if (fast_merge) {

    // This will be empty, as we add no new features, during merging,
    // but ensure it has the right size.
    C.cid_to_keypoint_map.clear();
    C.cid_to_keypoint_map.resize(C.cid_to_filename.size());

    // Compute the transform from the B map to the A map by finding the median
    // transform based on the shared images
    B2A_trans = computeTransformFromBToA(A, B);
  } else {
    // TODO(oalexan1): Modularize all this block
  
    // Find features among matching images
    std::string out_dir = "";
    bool save_matches = false;
    int num_overlaps = 0; // will use image_pairs
    std::vector<std::vector<std::pair<float, float>>> C_keypoint_vec;
    int initial_max_reprojection_error = -1; // won't be used
    bool verbose = false;
    bool filter_matches_using_cams = false; // do not have a single camera set yet
    bool read_nvm_no_shift = true; // not used, part of the api
    bool no_nvm_matches = true; // not used, part of the api
    dense_map::nvmData empty_nvm; // not used, part of the api
    C.cid_to_cam_t_global.resize(C.cid_to_filename.size()); // won't be used
    std::cout << "Number of image pairs to match: " << image_pairs.size() << std::endl;
    dense_map::detectMatchFeatures(// Inputs
                                   cams, R.cam_params,  out_dir, save_matches,  
                                   filter_matches_using_cams,  
                                   C.cid_to_cam_t_global,
                                   num_overlaps, image_pairs,
                                   initial_max_reprojection_error, FLAGS_num_threads,  
                                   read_nvm_no_shift, no_nvm_matches, verbose,  
                                   // Outputs
                                   C_keypoint_vec, C.pid_to_cid_fid, empty_nvm);

#if 1
    // TODO(oalexan1): Make this a function called splitMap().
    // Inputs: C.pid_to_cid_fid, C_keypoint_vec, cams, numa_cid, num_total_cid.
    // Find the tracks in both images
    std::vector<std::map<int, int>> A_pid_to_cid_fid, B_pid_to_cid_fid;
    std::vector<std::vector<std::pair<float, float>>> A_keypoint_vec, B_keypoint_vec;
    std::vector<dense_map::cameraImage> A_cams, B_cams;
    for (size_t pid = 0; pid < C.pid_to_cid_fid.size(); pid++) {

      auto & cid_fid = C.pid_to_cid_fid[pid];
      std::map<int, int> A_cid_fid, B_cid_fid;
    
      for (auto map_it = cid_fid.begin(); map_it != cid_fid.end(); map_it++) {
        int cid = map_it->first;
        int fid = map_it->second;
        if (cid < num_acid) 
          A_cid_fid[cid] = fid; // belongs to A
        else
          B_cid_fid[cid - num_acid] = fid; // belongs to B
      }
    
      if (A_cid_fid.size() > 1 && B_cid_fid.size() > 1) {
        // This is a shared track, that we break in two. Each obtained track
        // must have at least two images.
        A_pid_to_cid_fid.push_back(A_cid_fid);
        B_pid_to_cid_fid.push_back(B_cid_fid);
      }
    }

    // Break up the keypoint vec and the images
    if (C.cid_to_filename.size() != C_keypoint_vec.size()) 
      LOG(FATAL) << "There must be one keypoint set for each image.\n";
    for (size_t cid = 0; cid < C_keypoint_vec.size(); cid++) {
      if (cid < num_acid) {
        A_keypoint_vec.push_back(C_keypoint_vec[cid]);
        A_cams.push_back(cams[cid]);
      } else {
        B_keypoint_vec.push_back(C_keypoint_vec[cid]);
        B_cams.push_back(cams[cid]);
      }
    }
#endif

#if 1
    // TODO(oalexan1): This should be a function called findMatchingTriPoints().
    // Flag as outliers features outside of the distorted crop box
    std::vector<std::map<int, std::map<int, int>>> A_pid_cid_fid_inlier,
      B_pid_cid_fid_inlier;
    dense_map::flagOutlierByExclusionDist(// Inputs
                                          R.cam_params, A_cams, A_pid_to_cid_fid,
                                          A_keypoint_vec,
                                          // Outputs
                                          A_pid_cid_fid_inlier);
    dense_map::flagOutlierByExclusionDist(// Inputs
                                          R.cam_params, B_cams, B_pid_to_cid_fid,
                                          B_keypoint_vec,
                                          // Outputs
                                          B_pid_cid_fid_inlier);

    // Find triangulated points
    std::vector<Eigen::Vector3d> A_xyz_vec, B_xyz_vec; // triangulated points go here
    dense_map::multiViewTriangulation(// Inputs
                                      R.cam_params, A_cams, A.cid_to_cam_t_global,
                                      A_pid_to_cid_fid,
                                      A_keypoint_vec,
                                      // Outputs
                                      A_pid_cid_fid_inlier, A_xyz_vec);
    dense_map::multiViewTriangulation(// Inputs
                                      R.cam_params, B_cams,
                                      B.cid_to_cam_t_global,
                                      B_pid_to_cid_fid,
                                      B_keypoint_vec,
                                      // Outputs
                                      B_pid_cid_fid_inlier, B_xyz_vec);
  
    // Keep only the good points
    int count = 0;
    for (size_t pid = 0; pid < A_xyz_vec.size(); pid++) {
      if (!dense_map::isGoodTri(A_xyz_vec[pid]) || !dense_map::isGoodTri(B_xyz_vec[pid]))
        continue;
      A_xyz_vec[count] = A_xyz_vec[pid];
      B_xyz_vec[count] = B_xyz_vec[pid];
      count++;
    }
    A_xyz_vec.resize(count);
    B_xyz_vec.resize(count);
#endif

#if 1
    // TODO(oalexan1): Make this into a function named findMapToMapTransform().
    double inlier_threshold = estimateCloseDistance(A_xyz_vec);
    if (close_dist > 0.0) 
      inlier_threshold = close_dist;
  
    std::cout << "3D points are declared to be rather close if their distance is " 
              << inlier_threshold << " meters (option --close_dist). "
              << "Using this as inlier threshold.\n";
  
    // Estimate the transform from B_xyz_vec to A_xyz_vec using RANSAC.
    // A lot of outliers are possible.
    int  num_iterations = 1000;
    int  min_num_output_inliers = A_xyz_vec.size()/2;
    bool reduce_min_num_output_inliers_if_no_fit = true;  // If too many outliers
    bool increase_threshold_if_no_fit = false;  // better fail than give bad results
    std::vector<size_t> inlier_indices;
    try {
      RandomSampleConsensus<TranslationRotationScaleFittingFunctor, TransformError>
        ransac(TranslationRotationScaleFittingFunctor(),
               TransformError(), num_iterations,
               inlier_threshold, min_num_output_inliers,
               reduce_min_num_output_inliers_if_no_fit, increase_threshold_if_no_fit);
      B2A_trans = ransac(B_xyz_vec, A_xyz_vec);
      inlier_indices
        = ransac.inlier_indices(B2A_trans, B_xyz_vec, A_xyz_vec);
    } catch(std::exception const& e) {
      LOG(FATAL) << e.what() << "\n" << "Consider adjusting --close_dist or "
                 << "taking a closer look at your maps. They should have "
                 << "mostly images with non-small baseline.";
    }

    std::cout << "Number of RANSAC inliers: " << inlier_indices.size() << " ("
              << (100.0 * inlier_indices.size()) / A_xyz_vec.size() << " %)\n";
#endif

    // Convert keypoints to nvm format, updating C.cid_to_keypoint_map.
    C.cid_to_keypoint_map.resize(C.cid_to_filename.size());
    for (size_t cid = 0; cid < C.cid_to_filename.size(); cid++)
      vec2eigen(C_keypoint_vec[cid], C.cid_to_keypoint_map[cid]);
    C_keypoint_vec = std::vector<std::vector<std::pair<float, float>>> (); // wipe this
    
  } // end finding the transform using matches

  // LOG(INFO) does not do well with Eigen.
  std::cout << "Affine transform from second map to first map:\n";
  std::cout << "Rotation + scale:\n" << B2A_trans.linear()  << "\n";
  std::cout << "Translation: " << B2A_trans.translation().transpose() << "\n";
  
  // Bring the B map cameras in the A map coordinate system. Do not modify
  // B.cid_to_cam_t_global, but make a copy of it in B_trans_world_to_cam.
  std::vector<Eigen::Affine3d> B_trans_world_to_cam = B.cid_to_cam_t_global;
  dense_map::TransformCameras(B2A_trans, B_trans_world_to_cam);

  // Append all to the C map. Note how we use the transformed B.
  C.cid_to_cam_t_global.clear();
  C.cid_to_cam_t_global.insert(C.cid_to_cam_t_global.end(),
                           A.cid_to_cam_t_global.begin(), A.cid_to_cam_t_global.end());
  C.cid_to_cam_t_global.insert(C.cid_to_cam_t_global.end(),
                               B_trans_world_to_cam.begin(), B_trans_world_to_cam.end());

#if 1
  // TODO(oalexan1): Make this into a function called findCidMap.
  // If a few images show up in both and in B, so far they show up in C twice,
  // with different cid value. Fix that.
  // Also keep the images sorted.
  std::vector<std::string> sorted = C.cid_to_filename; // make a copy
  std::sort(sorted.begin(), sorted.end());
  int num_out_cams = 0;
  
  // The new index of each image after rm repetitions
  std::map<std::string, int> image2cid;  
  for (size_t cid = 0; cid < sorted.size(); cid++) {
    std::string img = sorted[cid];
    if (image2cid.find(img) == image2cid.end()) {
      image2cid[img] = num_out_cams;
      num_out_cams++;
    }
  }

  // The index of the cid after removing the repetitions
  std::map<int, int> cid2cid;
  for (size_t cid = 0; cid < C.cid_to_filename.size(); cid++)
    cid2cid[cid] = image2cid[C.cid_to_filename[cid]];
  if (num_out_cams != dense_map::maxMapVal(cid2cid) + 1) // sanity check
    LOG(FATAL) << "Book-keeping error in merging maps.\n";
  
#endif

  // Update C.cid_to_cam_t_global by merging poses for same images in the two maps
  MergePoses(cid2cid, C.cid_to_cam_t_global);

  // TODO(oalexan1): Make this a function too
  // Merge camera names
  {
    std::vector<std::string> cid_to_filename2(num_out_cams);
    for (size_t cid = 0; cid < C.cid_to_filename.size(); cid++)
      cid_to_filename2[cid2cid[cid]] = C.cid_to_filename[cid];

    C.cid_to_filename = cid_to_filename2;
  }

#if 1
  // By now we have 3 maps: A, B, and the new one in C having shared
  // tracks. Each of these has its own images and indices, and C
  // has repeated indices too, and need to merge them all into
  // a single set of tracks. We will not merge tracks, just remove
  // duplicates.
  // Factor this block out and call it mergeTracks().
  // Note that keypoint_offsets are applied before the cid2cid transform gets used!
  // There must be enough for all the input cameras.
  // This is very error-prone!
  std::vector<Eigen::Vector2d> keypoint_offsets(num_acid + num_bcid, Eigen::Vector2d(0, 0));
  std::vector<std::map<std::pair<float, float>, int>> merged_keypoint_map(num_out_cams);
  std::vector<int> find_count(num_out_cams, 0); // how many keypoints so far
  std::vector<std::map<int, int>> merged_pid_to_cid_fid;
  // Add A
  int cid_shift = 0; // A and C start with same images, so no shift
  dense_map::transformAppendNvm(A.pid_to_cid_fid, A.cid_to_keypoint_map,  
                                cid2cid, keypoint_offsets, cid_shift, num_out_cams,
                                // Append below
                                find_count, merged_keypoint_map,
                                merged_pid_to_cid_fid);
  // Add B
  cid_shift = num_acid; // the B map starts later
  dense_map::transformAppendNvm(B.pid_to_cid_fid, B.cid_to_keypoint_map,  
                                cid2cid, keypoint_offsets, cid_shift, num_out_cams,  
                                // Append below
                                find_count, merged_keypoint_map,
                                merged_pid_to_cid_fid);
  // Add C
  cid_shift = 0; // no shift, C is consistent with itself
  dense_map::transformAppendNvm(C.pid_to_cid_fid, C.cid_to_keypoint_map,  
                                cid2cid, keypoint_offsets, cid_shift, num_out_cams,  
                                // Append below
                                find_count, merged_keypoint_map,
                                merged_pid_to_cid_fid);

  // Overwrite C.pid_to_cid_fid after the merge
  C.pid_to_cid_fid = merged_pid_to_cid_fid;
  merged_pid_to_cid_fid = std::vector<std::map<int, int>>();

  // Remove duplicate tracks
  dense_map::rmDuplicateTracks(C.pid_to_cid_fid);

  // Update C.cid_to_keypoint_map. This has the same data as
  // merged_keypoint_map but need to reverse key and value and use
  // an Eigen::Matrix.
  C.cid_to_keypoint_map.clear();
  C.cid_to_keypoint_map.resize(num_out_cams);
  for (int cid = 0; cid < num_out_cams; cid++) {
    auto const& map = merged_keypoint_map[cid]; // alias
    C.cid_to_keypoint_map[cid] = Eigen::MatrixXd(2, map.size());
    for (auto map_it = map.begin(); map_it != map.end(); map_it++) {
      std::pair<float, float> const& K = map_it->first; 
      int fid = map_it->second;
      C.cid_to_keypoint_map.at(cid).col(fid) = Eigen::Vector2d(K.first, K.second);
    }
  }
#endif
  
  // Merge the camera vector
  // TODO(oalexan1): Make this a function
  {
    std::vector<dense_map::cameraImage> merged_cams(num_out_cams);
    for (size_t cid = 0; cid < cams.size(); cid++) 
      merged_cams[cid2cid[cid]] = cams[cid];

    cams = merged_cams;
  }
  
  // Create C_keypoint_vec. Same info as C.cid_to_keypoint_map but different structure.
  std::vector<std::vector<std::pair<float, float>>> C_keypoint_vec;
  C_keypoint_vec.resize(num_out_cams);
  for (int cid = 0; cid < num_out_cams; cid++)
    eigen2vec(C.cid_to_keypoint_map[cid], C_keypoint_vec[cid]);

  // Flag outliers
  std::vector<std::map<int, std::map<int, int>>> C_pid_cid_fid_inlier;
  dense_map::flagOutlierByExclusionDist(// Inputs
                                        R.cam_params, cams, C.pid_to_cid_fid,
                                        C_keypoint_vec,
                                        // Outputs
                                        C_pid_cid_fid_inlier);

  // Triangulate the merged tracks with merged cameras
  dense_map::multiViewTriangulation(// Inputs
                                    R.cam_params, cams, C.cid_to_cam_t_global,
                                    C.pid_to_cid_fid,
                                    C_keypoint_vec,
                                    // Outputs
                                    C_pid_cid_fid_inlier, C.pid_to_xyz);

  // TODO(oalexan1): Should one remove outliers from tri points
  // and C.pid_to_cid_fid?
}

}  // namespace sparse_mapping
