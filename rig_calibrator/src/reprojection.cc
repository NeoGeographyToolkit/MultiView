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

#include <rig_calibrator/bundle_adjust.h>
#include <rig_calibrator/matching.h>
#include <rig_calibrator/essential.h>

#include <camera_model/camera_model.h>

#include <ceres/rotation.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <gflags/gflags.h>

#include <random>
#include <thread>
#include <unordered_map>

DEFINE_uint64(num_min_localization_inliers, 10,
              "If fewer than this many number of inliers, localization has failed.");

// TODO(oalexan1): This may not work well with haz cameras
DEFINE_int32(min_pairwise_matches, 10,
             "Minimum number of valid inlier matches required to keep matches for given "
             "image pair.");

DEFINE_int32(max_pairwise_matches, 2000,
             "Maximum number of pairwise matches in an image pair to keep.");

DEFINE_bool(silent_matching, false,
            "Do not print a lot of verbose info when matching.");

namespace sparse_mapping {

ceres::LossFunction* GetLossFunction(std::string cost_fun, double th) {
  // Convert to lower-case
  std::transform(cost_fun.begin(), cost_fun.end(), cost_fun.begin(), ::tolower);

  ceres::LossFunction* loss_function = NULL;
  if (cost_fun == "l2")
    loss_function = NULL;
  else if ( cost_fun == "huber"  )
    loss_function = new ceres::HuberLoss(th);
  else if ( cost_fun == "cauchy" )
    loss_function = new ceres::CauchyLoss(th);
  else if ( cost_fun == "l1"     )
    loss_function = new ceres::SoftLOneLoss(th);
  else
    LOG(FATAL) << "Unknown cost function: " + cost_fun;

  return loss_function;
}

// if parms is null, don't worry about converting to pixels
struct ReprojectionError {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit ReprojectionError(const Eigen::Vector2d & observed)
    : observed(observed) {}

  template <typename T>
  bool operator()(const T* const camera_p_global,
                  const T* const camera_aa_global,
                  const T* const point_global,
                  const T* const focal_length,
                  T* residuals) const {
    // Project the point into the camera's coordinate frame
    T p[3];
    ceres::AngleAxisRotatePoint(camera_aa_global, point_global, p);
    p[0] += camera_p_global[0];
    p[1] += camera_p_global[1];
    p[2] += camera_p_global[2];

    T xp = (p[0] / p[2]) * focal_length[0];
    T yp = (p[1] / p[2]) * focal_length[0];

    // The error is the difference between the prediction and observed
    residuals[0] = xp - T(observed.x());
    residuals[1] = yp - T(observed.y());

    return true;
  }

  // Helper function ... make the code look nice
  static ceres::CostFunction* Create(const Eigen::Vector2d & observed) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3, 3, 1>
            (new ReprojectionError(observed)));
  }

  Eigen::Vector2d observed;
};

Eigen::Vector3d TriangulatePoint(Eigen::Vector3d const& unnormalized_pt1,
                                 Eigen::Vector3d const& unnormalized_pt2,
                                 Eigen::Matrix3d const& cam2_r_cam1,
                                 Eigen::Vector3d const& cam2_t_cam1,
                                 double * error) {
  // The second camera's center in the coordinate system of the first
  // camera.
  Eigen::Vector3d p2 = -cam2_r_cam1.transpose() * cam2_t_cam1;

  // Calculate the two unit pointing vectors in the domain of cam1
  Eigen::Vector3d unit1 = unnormalized_pt1.normalized();
  Eigen::Vector3d unit2 = cam2_r_cam1.transpose() * unnormalized_pt2.normalized();

  Eigen::Vector3d v12 = unit1.cross(unit2);
  Eigen::Vector3d v1 = v12.cross(unit1);
  Eigen::Vector3d v2 = v12.cross(unit2);

  Eigen::Vector3d closestPoint1 = v2.dot(p2) / v2.dot(unit1) * unit1;
  Eigen::Vector3d closestPoint2 = p2 + v1.dot(-p2) / v1.dot(unit2) * unit2;
  *error = (closestPoint2 - closestPoint1).norm();

  return 0.5 * (closestPoint2 + closestPoint1);
}
  
void BundleAdjust(std::vector<std::map<int, int> > const& pid_to_cid_fid,
                  std::vector<Eigen::Matrix2Xd> const& cid_to_keypoint_map, double focal_length,
                  std::vector<Eigen::Affine3d>* cid_to_cam_t_global,
                  std::vector<Eigen::Vector3d>* pid_to_xyz,
                  std::vector<std::map<int, int> > const& user_pid_to_cid_fid,
                  std::vector<Eigen::Matrix2Xd> const& user_cid_to_keypoint_map,
                  std::vector<Eigen::Vector3d>* user_pid_to_xyz, ceres::LossFunction* loss,
                  ceres::Solver::Options const& options, ceres::Solver::Summary* summary,
                  int first, int last,
                  bool fix_all_cameras, std::set<int> const& fixed_cameras) {
  // Perform bundle adjustment. Keep fixed all cameras with cid
  // not within [first, last] and all xyz points which project only
  // onto fixed cameras.

  // If provided, use user-set registration points in the second pass.

  // Allocate space for the angle axis representation of rotation
  std::vector<double> camera_aa_storage(3 * cid_to_cam_t_global->size());
  for (size_t cid = 0; cid < cid_to_cam_t_global->size(); cid++) {
    Eigen::Map<Eigen::Vector3d> aa_storage(camera_aa_storage.data() + 3 * cid);
    Eigen::Vector3d vec;
    camera::RotationToRodrigues(cid_to_cam_t_global->at(cid).linear(),
                               &vec);
    aa_storage = vec;
  }

  // Build problem
  ceres::Problem problem;

  // Ideally the block inside of the loop below must be a function call,
  // but the compiler does not handle that correctly with ceres.
  // So do this by changing where things are pointing.

  int num_passes = 1;
  if (!user_pid_to_xyz->empty()) num_passes = 2;  // A second pass using control points

  for (int pass = 0; pass < num_passes; pass++) {
    std::vector<std::map<int, int> > const * p_pid_to_cid_fid;
    std::vector<Eigen::Matrix2Xd >   const * p_cid_to_keypoint_map;
    std::vector<Eigen::Vector3d>           * p_pid_to_xyz;
    ceres::LossFunction * local_loss;
    if (pass == 0) {
      local_loss            = loss;  // outside-supplied loss
      p_pid_to_cid_fid      = &pid_to_cid_fid;
      p_cid_to_keypoint_map = &cid_to_keypoint_map;
      p_pid_to_xyz          = pid_to_xyz;
    } else {
      local_loss            = NULL;  // l2, as user-supplied data is reliable
      p_pid_to_cid_fid      = &user_pid_to_cid_fid;
      p_cid_to_keypoint_map = &user_cid_to_keypoint_map;
      p_pid_to_xyz          = user_pid_to_xyz;
    }

    for (size_t pid = 0; pid < p_pid_to_xyz->size(); pid++) {
      if ((*p_pid_to_cid_fid)[pid].size() < 2)
        LOG(FATAL) << "Found a track of size < 2.";

      // Don't vary points which project only into cameras which we don't vary.
      bool fix_pid = true;
      for (std::map<int, int>::value_type const& cid_fid : (*p_pid_to_cid_fid)[pid]) {
        if (cid_fid.first >= first && cid_fid.first <= last)
          fix_pid = false;
      }

      for (std::map<int, int>::value_type const& cid_fid : (*p_pid_to_cid_fid)[pid]) {
        ceres::CostFunction* cost_function =
          ReprojectionError::Create((*p_cid_to_keypoint_map)[cid_fid.first].col(cid_fid.second));

        problem.AddResidualBlock(cost_function,
                                 local_loss,
                                 &cid_to_cam_t_global->at(cid_fid.first).translation()[0],
                                 &camera_aa_storage[3 * cid_fid.first],
                                 &p_pid_to_xyz->at(pid)[0],
                                 &focal_length);

        if (fix_all_cameras || (cid_fid.first < first || cid_fid.first > last) ||
            fixed_cameras.find(cid_fid.first) != fixed_cameras.end()) {
          problem.SetParameterBlockConstant(&cid_to_cam_t_global->at(cid_fid.first).translation()[0]);
          problem.SetParameterBlockConstant(&camera_aa_storage[3 * cid_fid.first]);
        }
      }
      if (fix_pid || pass == 1) {
        // Fix pids which don't project in cameras that are floated.
        // Also, must not float points given by the user, those are measurements
        // we are supposed to reference ourselves against, and floating
        // them can make us lose the real world scale.
        problem.SetParameterBlockConstant(&p_pid_to_xyz->at(pid)[0]);
      }
    }
    problem.SetParameterBlockConstant(&focal_length);
  }

  // Solve the problem
  ceres::Solve(options, &problem, summary);

  // Write the rotations back to the transform
  for (size_t cid = 0; cid < cid_to_cam_t_global->size(); cid++) {
    Eigen::Map<Eigen::Vector3d> aa_storage
      (camera_aa_storage.data() + 3 * cid);
    Eigen::Matrix3d r;
    camera::RodriguesToRotation(aa_storage, &r);
    cid_to_cam_t_global->at(cid).linear() = r;
  }
}

// This is a very specialized function
void BundleAdjustSmallSet(std::vector<Eigen::Matrix2Xd> const& features_n,
                          double focal_length,
                          std::vector<Eigen::Affine3d> * cam_t_global_n,
                          Eigen::Matrix3Xd * pid_to_xyz,
                          ceres::LossFunction * loss,
                          ceres::Solver::Options const& options,
                          ceres::Solver::Summary * summary) {
  CHECK(cam_t_global_n) << "Variable cam_t_global_n needs to be defined";
  CHECK(cam_t_global_n->size() == features_n.size())
    << "Variables features_n and cam_t_global_n need to agree on the number of cameras";
  CHECK(cam_t_global_n->size() > 1) << "Bundle adjust needs at least 2 or more cameras";
  CHECK(pid_to_xyz->cols() == features_n[0].cols())
    << "There should be an equal amount of XYZ points as there are feature observations";
  for (size_t i = 1; i < features_n.size(); i++) {
    CHECK(features_n[0].cols() == features_n[i].cols())
      << "The same amount of features should be seen in all cameras";
  }

  const size_t n_cameras = features_n.size();

  // Allocate space for the angle axis representation of rotation
  std::vector<Eigen::Vector3d> aa(n_cameras);
  for (size_t cid = 0; cid < n_cameras; cid++) {
    camera::RotationToRodrigues(cam_t_global_n->at(cid).linear(), &aa[cid]);
  }

  // Build the problem
  ceres::Problem problem;
  for (ptrdiff_t pid = 0; pid < pid_to_xyz->cols(); pid++) {
    for (size_t cid = 0; cid < n_cameras; cid++) {
      ceres::CostFunction* cost_function = ReprojectionError::Create(features_n[cid].col(pid));
      problem.AddResidualBlock(cost_function, loss,
                               &cam_t_global_n->at(cid).translation()[0],
                               &aa.at(cid)[0],
                               &pid_to_xyz->col(pid)[0],
                               &focal_length);
    }
  }
  problem.SetParameterBlockConstant(&focal_length);

  // Solve the problem
  ceres::Solve(options, &problem, summary);

  // Write the rotations back to the transform
  Eigen::Matrix3d r;
  for (size_t cid = 0; cid < n_cameras; cid++) {
    camera::RodriguesToRotation(aa[cid], &r);
    cam_t_global_n->at(cid).linear() = r;
  }
}

void EstimateCamera(camera::CameraModel* camera_estimate,
                    std::vector<Eigen::Vector3d>* landmarks,
                    const std::vector<Eigen::Vector2d> & observations,
                    const ceres::Solver::Options & options,
                    ceres::Solver::Summary* summary) {
  Eigen::Affine3d guess = camera_estimate->GetTransform();
  camera::CameraParameters params = camera_estimate->GetParameters();

  // Initialize the angle axis representation of rotation
  Eigen::Vector3d aa;
  camera::RotationToRodrigues(guess.linear(), &aa);

  double focal_length = params.GetFocalLength();

  // Build problem
  ceres::Problem problem;
  for (size_t pid = 0; pid < landmarks->size(); pid++) {
    ceres::CostFunction* cost_function = ReprojectionError::Create(
                Eigen::Vector2d(observations[pid].x(), observations[pid].y()));
    problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(1.0),
                             &guess.translation()[0],
                             &aa[0],
                             &landmarks->at(pid)[0],
                             &focal_length);
    problem.SetParameterBlockConstant(&landmarks->at(pid)[0]);
  }
  problem.SetParameterBlockConstant(&focal_length);

  // Solve the problem
  ceres::Solve(options, &problem, summary);

  // Write the rotations back to the transform
  Eigen::Matrix3d r;
  camera::RodriguesToRotation(aa, &r);
  guess.linear() = r;
  camera_estimate->SetTransform(guess);
}

// random intger in [min, max)
int RandomInt(int min, int max) {
  static std::mt19937 generator;  // should be thread_local for thread safe, gcc 4.6 doesn't support
  std::uniform_int_distribution<int> random_item(min, max - 1);
  return random_item(generator);
}

void SelectRandomObservations(const std::vector<Eigen::Vector3d> & all_landmarks,
        const std::vector<Eigen::Vector2d> & all_observations, size_t num_selected,
        std::vector<cv::Point3d> * landmarks, std::vector<cv::Point2d> * observations) {
  std::unordered_map<int, int> used;
  // not enough observations
  if (all_observations.size() < num_selected)
    return;
  // Reserve space in the output so we don't have to keep reallocating on
  // push_back().
  landmarks->reserve(num_selected);
  observations->reserve(num_selected);
  while (observations->size() < num_selected) {
    int id = RandomInt(0, all_observations.size());
    if (used.count(id) > 0)
      continue;
    Eigen::Vector3d p = all_landmarks[id];
    landmarks->push_back(cv::Point3d(p[0], p[1], p[2]));
    observations->push_back(cv::Point2d(all_observations[id][0], all_observations[id][1]));
    used[id] = 1;
  }
}

bool P3P(const std::vector<cv::Point3d> & landmarks, const std::vector<cv::Point2d> & observations,
         const camera::CameraParameters & params, Eigen::Vector3d * pos,
         Eigen::Matrix3d * rotation) {

  cv::Mat camera_matrix(3, 3, cv::DataType<double>::type);
  cv::eigen2cv(params.GetIntrinsicMatrix<camera::UNDISTORTED_C>(), camera_matrix);
  cv::Mat rvec(3, 1, cv::DataType<double>::type, cv::Scalar(0));
  cv::Mat tvec(3, 1, cv::DataType<double>::type, cv::Scalar(0));
  cv::Mat distortion(4, 1, cv::DataType<double>::type, cv::Scalar(0));
  bool result = cv::solvePnP(landmarks, observations, camera_matrix, distortion, rvec, tvec,
                             false, cv::SOLVEPNP_P3P);
  if (!result)
    return false;
  cv::cv2eigen(tvec, *pos);
  camera::RodriguesToRotation(Eigen::Vector3d(rvec.at<double>(0), rvec.at<double>(1),
                                              rvec.at<double>(2)), rotation);
  
  return true;
}

size_t CountInliers(const std::vector<Eigen::Vector3d> & landmarks,
                    const std::vector<Eigen::Vector2d> & observations, 
                    const camera::CameraModel & camera, int tolerance,
                    std::vector<size_t>* inliers) {
  int num_inliers = 0;
  if (inliers) {
    // To save ourselves some allocation time. We'll prealloc for a 50% inlier
    // success rate
    inliers->reserve(observations.size()/2);
  }

  double tolerance_sq = tolerance * tolerance;

  for (size_t i = 0; i < landmarks.size(); i++) {
    Eigen::Vector2d pos = camera.ImageCoordinates(landmarks[i]);
    if ((observations[i] - pos).squaredNorm() <= tolerance_sq) {
      num_inliers++;
      if (inliers)
        inliers->push_back(i);
    }
  }
  return num_inliers;
}

int RansacEstimateCamera(const std::vector<Eigen::Vector3d> & landmarks,
                         const std::vector<Eigen::Vector2d> & observations,
                         int num_tries, int inlier_tolerance, camera::CameraModel * camera_estimate,
                         std::vector<Eigen::Vector3d> * inlier_landmarks_out,
                         std::vector<Eigen::Vector2d> * inlier_observations_out,
                         bool verbose) {
  size_t best_inliers = 0;
  camera::CameraParameters params = camera_estimate->GetParameters();

  // Need the minimum number of observations
  if (observations.size() < 4)
    return 1;

  // RANSAC to find the best camera with P3P
  std::vector<cv::Point3d> subset_landmarks;
  std::vector<cv::Point2d> subset_observations;
  // TODO(oalexan1): Use multiple threads here?
  for (int i = 0; i < num_tries; i++) {
    subset_landmarks.clear();
    subset_observations.clear();
    SelectRandomObservations(landmarks, observations, 4, &subset_landmarks, &subset_observations);

    Eigen::Vector3d pos;
    Eigen::Matrix3d rotation;
    bool result = P3P(subset_landmarks, subset_observations, params, &pos, &rotation);
    if (!result)
      continue;
    Eigen::Affine3d cam_t_global;
    cam_t_global.setIdentity();
    cam_t_global.translate(pos);
    cam_t_global.rotate(rotation);
    camera::CameraModel guess(cam_t_global, camera_estimate->GetParameters());

    size_t inliers = CountInliers(landmarks, observations, guess, inlier_tolerance, NULL);
    if (inliers > best_inliers) {
      best_inliers = inliers;
      *camera_estimate = guess;
    }
  }

  if (verbose)
    std::cout << observations.size() << " Ransac observations "
              << best_inliers << " inliers\n";

  // TODO(bcoltin): Return some sort of confidence?
  if (best_inliers < FLAGS_num_min_localization_inliers)
    return 2;

  std::vector<size_t> inliers;
  CountInliers(landmarks, observations, *camera_estimate, inlier_tolerance, &inliers);
  std::vector<Eigen::Vector3d> inlier_landmarks;
  std::vector<Eigen::Vector2d> inlier_observations;
  inlier_landmarks.reserve(inliers.size());
  inlier_observations.reserve(inliers.size());
  for (size_t idx : inliers) {
    inlier_landmarks.push_back(landmarks[idx]);
    inlier_observations.push_back(observations[idx]);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.num_threads = 1;  // it is no slower with only one thread
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  // improve estimate with CERES solver
  EstimateCamera(camera_estimate, &inlier_landmarks, inlier_observations, options, &summary);

  // find inliers again with refined estimate
  inliers.clear();
  best_inliers = CountInliers(landmarks, observations, *camera_estimate, inlier_tolerance, &inliers);

  if (verbose)
    std::cout << "Number of inliers with refined camera: " << best_inliers << "\n";

  if (best_inliers < FLAGS_num_min_localization_inliers)
    return 2;

  inlier_landmarks.clear();
  inlier_observations.clear();
  inlier_landmarks.reserve(inliers.size());
  inlier_observations.reserve(inliers.size());
  for (size_t idx : inliers) {
    inlier_landmarks.push_back(landmarks[idx]);
    inlier_observations.push_back(observations[idx]);
  }
  if (inlier_landmarks_out) {
    inlier_landmarks_out->reserve(inliers.size());
    std::copy(inlier_landmarks.begin(), inlier_landmarks.end(),
        std::back_inserter(*inlier_landmarks_out));
  }
  if (inlier_observations_out) {
    inlier_observations_out->reserve(inliers.size());
    std::copy(inlier_observations.begin(), inlier_observations.end(),
        std::back_inserter(*inlier_observations_out));
  }

  return 0;
}

// Filter the matches by a geometric constraint. Compute the essential matrix.
// TODO(oalexan1): This assumes both cameras have the same intrinsics. Fix this.
// TODO(oalexan1): This likely assumes undistorted keypoints.
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

  if (vec_inliers.size() < static_cast<size_t>(FLAGS_min_pairwise_matches)) {
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
  // TODO(oalexan1): 200 iterations may be too much. Likely 20 is enough.
  options.max_num_iterations = 200;
  options.logging_type = ceres::SILENT;
  options.num_threads = 1; // this function will be called with multiple threads
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
    if (angles.size() >= static_cast<size_t>(2*FLAGS_min_pairwise_matches))
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
  if (static_cast<int32_t>(matches.size()) < FLAGS_min_pairwise_matches) {
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

  if (static_cast<int32_t>(inlier_matches.size()) < FLAGS_min_pairwise_matches) {
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
  
}
