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

#include <camera_model/rpc_distortion.h>
#include <camera_model/camera_params.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <ceres/cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/dynamic_numeric_diff_cost_function.h>
#include <ceres/numeric_diff_cost_function.h>
#include <ceres/autodiff_cost_function.h>

#include <Eigen/Dense>

#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <iomanip>

namespace dense_map {
const int PIXEL_SIZE = 2;

int rpc_degree(int num_dist_params) {
  return static_cast<int>(round(sqrt(2.0 * num_dist_params + 5.0) / 2.0 - 1.5));
}

int num_dist_params(int rpc_degree) {
  return 2*(rpc_degree+1)*(rpc_degree+2)-2;
}

// See if the current set of parameters have the right size to be usable
// with some RPC model
void validate_distortion_params(int num_params) {
  int deg = rpc_degree(num_params);
  if (num_dist_params(deg) != num_params || deg <= 0 || std::isnan(deg))
    throw "Incorrect number of RPC coefficients.";
}

// A little function to append zeros to a Vector.
void append_zeros_to_vector(Eigen::VectorXd & vec, int num) {
  int len = vec.size();

  // Create a vector big enough to store the output
  Eigen::VectorXd out_vec;
  out_vec.resize(len + num);
  // Copy current elements
  for (int it = 0; it < len; it++) out_vec[it] = vec[it];
  // Fill the rest with zeros
  for (int it = len; it < len + num; it++) out_vec[it] = 0.0;

  vec = out_vec;
}

// Prepend a 1 to a vector
void prepend_1(Eigen::VectorXd & vec) {
  int old_len = vec.size();
  Eigen::VectorXd old_vec = vec;

  vec.resize(old_len + 1);
  vec[0] = 1.0;

  for (int it = 0; it < old_len; it++) vec[it + 1] = old_vec[it];

}

// Remove 1 from first position in a vector
void remove_1(Eigen::VectorXd & vec) {
  int old_len = vec.size();
  if (old_len <= 0)
    throw "Found an unexpected empty vector.";
  Eigen::VectorXd old_vec = vec;

  vec.resize(old_len - 1);
  for (int it = 0; it < old_len - 1; it++) vec[it] = old_vec[it + 1];
}

Eigen::VectorXd subvector(Eigen::VectorXd const& vec, int start, int len) {
  if (start + len > vec.size()) throw "Out of range in subvector().";

  Eigen::VectorXd subvec(len);
  for (int it = 0; it < len; it++) subvec[it] = vec[start + it];

  return subvec;
}

void set_subvector(Eigen::VectorXd & vec, int start, int len, Eigen::VectorXd const& subvec) {
  if (start + len > vec.size()) throw "Out of range in set_subvector().\n";

  if (len != subvec.size()) throw "Size mismatch in set_subvector().";

  for (int it = 0; it < len; it++) vec[start + it] = subvec[it];
}

// Compute the RPC model with given coefficients at the given point.
// Recall that RPC is ratio of two polynomials in x and y.
Eigen::Vector2d compute_rpc(Eigen::Vector2d const& p, Eigen::VectorXd const& coeffs)  {
  validate_distortion_params(coeffs.size());

  int rpc_deg = rpc_degree(coeffs.size());
  double x = p[0];
  double y = p[1];

  // Precompute x^n and y^m values
  std::vector<double> powx(rpc_deg + 1), powy(rpc_deg + 1);
  double valx = 1.0, valy = 1.0;
  for (int deg = 0; deg <= rpc_deg; deg++) {
    powx[deg] = valx;
    valx *= x;
    powy[deg] = valy;
    valy *= y;
  }

  // Evaluate the RPC expression. The denominator always has a 1 as
  // the 0th coefficient.
  int coeff_index = 0;

  // Loop four times, for output first coordinate numerator and
  // denominator, then for output second coordinate numerator and
  // denominator.

  double vals[] = {0.0, 1.0, 0.0, 1.0};

  for (int count = 0; count < 4; count++) {
    int start = 0;                            // starting degree for numerator
    if (count == 1 || count == 3) start = 1;  // starting degree for denominator

    for (int deg = start; deg <= rpc_deg; deg++) {
      for (int i = 0; i <= deg; i++) {
        // Add coeff * x^(deg-i) * y^i
        vals[count] += coeffs[coeff_index] * powx[deg - i] * powy[i];
        coeff_index++;
      }
    }
  }

  if (coeff_index != static_cast<int>(coeffs.size()))
    throw "Book-keeping failure in RPCLensDistortion.";

  return Eigen::Vector2d(vals[0]/vals[1], vals[2]/vals[3]);
}

// Put the vectors of numerator and denominator coefficients for the x and y
// coordinates into a single vector.
void pack_params(Eigen::VectorXd& params, Eigen::VectorXd const& num_x,
                 Eigen::VectorXd const& den_x, Eigen::VectorXd const& num_y,
                 Eigen::VectorXd const& den_y) {
  int num_len = num_x.size();
  int den_len = den_x.size();

  if (num_len != den_len + 1 ||
      num_len != static_cast<int>(num_y.size()) ||
      den_len != static_cast<int>(den_y.size()))
    throw "Book-keeping failure in RPCLensDistortion.";

  params.resize(2*num_len + 2*den_len);

  set_subvector(params, 0, num_len, num_x);
  set_subvector(params, num_len,                     den_len, den_x);
  set_subvector(params, num_len + den_len,           num_len, num_y);
  set_subvector(params, num_len + den_len + num_len, den_len, den_y);
  validate_distortion_params(params.size());
}

void unpack_params(Eigen::VectorXd const& params, Eigen::VectorXd& num_x, Eigen::VectorXd& den_x,
                   Eigen::VectorXd& num_y, Eigen::VectorXd& den_y) {
  validate_distortion_params(params.size());

  int num_params = params.size();
  int num_len = (num_params + 2)/4;
  int den_len = num_len - 1;  // because the denominator always starts with 1.
  num_x = subvector(params, 0,                           num_len);
  den_x = subvector(params, num_len,                     den_len);
  num_y = subvector(params, num_len + den_len,           num_len);
  den_y = subvector(params, num_len + den_len + num_len, den_len);
}

// RPC lens distortion of arbitrary degree.
// For a given undistorted centered pixel (x, y), compute
// (P1num(x, y)/P1den(x, y), P2num(x, y)/P2den(x, y))
// where these polynomials are of at most given degree,
// and P1den(0, 0) = P2den(0, 0) = 1.

// Undistortion is done analogously using a second set of
// coefficients.

// TODO(oalexan1): Make undistortion computation a member of this class.

// ======== RPCLensDistortion ========
// This class is not fully formed until both distortion and
// undistortion parameters are computed.
// One must always call set_undistortion_parameters()
// only after set_distortion_parameters().

RPCLensDistortion::RPCLensDistortion() {
  m_rpc_degree = 0;
  m_can_undistort = false;
}

RPCLensDistortion::RPCLensDistortion(Eigen::VectorXd const& params): m_distortion(params) {
  validate_distortion_params(params.size());
  m_rpc_degree = rpc_degree(params.size());
  m_can_undistort = false;
}

Eigen::VectorXd RPCLensDistortion::distortion_parameters() const { return m_distortion; }

Eigen::VectorXd RPCLensDistortion::undistortion_parameters() const { return m_undistortion; }

void RPCLensDistortion::set_distortion_parameters(Eigen::VectorXd const& params) {
  validate_distortion_params(params.size());

  // If the distortion parameters changed, one cannot undistort until the undistortion
  // coefficients are computed.
  if (params.size() != m_distortion.size()) {
    m_can_undistort = false;
  } else {
    for (size_t it = 0; it < params.size(); it++) {
      if (m_distortion[it] != params[it]) {
        m_can_undistort = false;
        break;
      }
    }
  }
  m_distortion = params;
  m_rpc_degree = rpc_degree(params.size());
}

void RPCLensDistortion::set_undistortion_parameters(Eigen::VectorXd const& params) {
  if (params.size() != num_dist_params())
    throw "The number of distortion and undistortion parameters must agree.";
  m_undistortion = params;
  m_can_undistort = true;
}

Eigen::VectorXd RPCLensDistortion::dist_undist_params() {
  int num_dist = m_distortion.size();
  int num_undist = m_undistortion.size();
  if (num_dist != num_undist) 
    throw "There must be as many distortion as undistortion params.";
  
  Eigen::VectorXd dist_undist_params(num_dist + num_undist);

  for (int it = 0; it < num_dist; it++)
    dist_undist_params[it] = m_distortion[it];

  for (int it = 0; it < num_undist; it++)
    dist_undist_params[it + num_dist] = m_undistortion[it];
  
  return dist_undist_params;
}

void RPCLensDistortion::set_dist_undist_params(Eigen::VectorXd const& dist_undist) {

  int num = dist_undist.size();
  if (num % 2 != 0) 
    throw "The total number of distortion and undistortion params must be even.";

  int num_dist = num / 2;
  int num_undist = num / 2;
  
  Eigen::VectorXd dist_params(num_dist);
  Eigen::VectorXd undist_params(num_undist);

  for (int it = 0; it < num_dist; it++)
    dist_params[it] = dist_undist[it];

  for (int it = 0; it < num_undist; it++)
    undist_params[it] = dist_undist[it + num_dist];

  // These two functions will do further sanity checks
  set_distortion_parameters(dist_params);
  set_undistortion_parameters(undist_params);
}
  
void RPCLensDistortion::scale(double scale) {
  m_distortion *= scale;
  m_undistortion *= scale;
}


// Make RPC coefficients so that the RPC transform is the identity.
// The vector params must already have the right size.
void RPCLensDistortion::init_as_identity(Eigen::VectorXd& params) {
  validate_distortion_params(params.size());

  for (size_t it = 0; it < params.size(); it++)
    params[it] = 0.0;

  Eigen::VectorXd num_x, den_x, num_y, den_y;
  unpack_params(params, num_x, den_x, num_y, den_y);

  // Initialize the transform (x, y) -> (x, y), which is
  // ( (0 + 1*x + 0*y)/(1 + 0*x + 0*y), (0 + 0*x + 1*y)/(1 + 0*x + 0*y) )
  // hence set num_x and num_y accordingly. As always, we do not
  // store the 1 values in the denominator.
  num_x[1] = 1; num_y[2] = 1;
  pack_params(params, num_x, den_x, num_y, den_y);
}

// Form the identity transform
void RPCLensDistortion::reset(int rpc_degree) {
  if (rpc_degree <= 0) throw "The RPC degree must be positive.";

  m_rpc_degree = rpc_degree;
  int num_params = dense_map::num_dist_params(rpc_degree);
  m_distortion.resize(num_params);
  m_undistortion.resize(num_params);
  init_as_identity(m_distortion);
  init_as_identity(m_undistortion);
  m_can_undistort = true;
}

// Given the RPC coefficients corresponding to the four polynomials,
// increase the degree of each polynomial by 1 and set the new
// coefficients to 0.
void RPCLensDistortion::increment_degree(Eigen::VectorXd& params) {
  validate_distortion_params(params.size());

  Eigen::VectorXd num_x, den_x, num_y, den_y;
  unpack_params(params, num_x, den_x, num_y, den_y);

  int r = rpc_degree(params.size());

  // The next monomials to add will be
  // x^(r+1), x^r*y, ..., x*y^r, y^(r+1)
  // and there are r + 2 of them.
  // Set their coefficients to zero.
  int num = r + 2;

  append_zeros_to_vector(num_x, num);
  append_zeros_to_vector(den_x, num);
  append_zeros_to_vector(num_y, num);
  append_zeros_to_vector(den_y, num);

  pack_params(params, num_x, den_x, num_y, den_y);
}

Eigen::Vector2d RPCLensDistortion::distort_centered(Eigen::Vector2d const& p) const {
  return compute_rpc(p, m_distortion);
}

Eigen::Vector2d RPCLensDistortion::undistort_centered(Eigen::Vector2d const& p) const {
  if (!m_can_undistort)
    throw "The RPC model is not ready for undistortion";
  return compute_rpc(p, m_undistortion);
}

// An error function minimizing the fit of an RPC model, that is,
// minimizing norm of dist_pix - RPC_model(undist_pix).
struct RpcFitError {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  RpcFitError(Eigen::Vector2d const& undist_pix, Eigen::Vector2d const& dist_pix,
              std::vector<int> const& block_sizes):
    m_undist_pix(undist_pix), m_dist_pix(dist_pix), m_block_sizes(block_sizes) {
    // Sanity check
    if (block_sizes.size() != 1)
      throw "RpcFitError: The block sizes were not set up properly.\n";
    validate_distortion_params(block_sizes[0]);
  }

  // Call to work with ceres::DynamicNumericDiffCostFunction.
  bool operator()(double const* const* parameters, double* residuals) const {
    int num_coeffs = m_block_sizes[0];
    Eigen::VectorXd coeffs(num_coeffs);
    for (int it = 0; it < num_coeffs; it++) coeffs[it] = parameters[0][it];

    // distort
    Eigen::Vector2d rpc_dist = compute_rpc(m_undist_pix, coeffs);

    for (int it = 0; it < PIXEL_SIZE; it++) residuals[it] = rpc_dist[it] - m_dist_pix[it];

    return true;
  }

  // Factory to hide the construction of the CostFunction object from the client code.
  // TODO(oalexan1): Use analytical diff cost function
  static ceres::CostFunction*
  Create(Eigen::Vector2d const& undist_pix, Eigen::Vector2d const& dist_pix,
         std::vector<int> const& block_sizes) {
    ceres::DynamicNumericDiffCostFunction<RpcFitError>* cost_function =
      new ceres::DynamicNumericDiffCostFunction<RpcFitError>
      (new RpcFitError(undist_pix, dist_pix, block_sizes));

    cost_function->SetNumResiduals(PIXEL_SIZE);

    for (size_t i = 0; i < block_sizes.size(); i++)
      cost_function->AddParameterBlock(block_sizes[i]);

    return cost_function;
  }

 private:
  Eigen::Vector2d m_undist_pix, m_dist_pix;
  std::vector<int> m_block_sizes;
};  // End class RpcFitError

// TODO(oalexan1): Move this to utils and factor out of camera_refiner.cc as well.
// Calculate the rmse residual for each residual type.
void calc_residuals_stats(std::vector<double> const& residuals,
                           std::vector<std::string> const& residual_names,
                           std::string const& tag) {
  size_t num = residuals.size();

  if (num != residual_names.size())
    throw "There must be as many residuals as residual names.";

  std::map<std::string, std::vector<double>> stats;
  for (size_t it = 0; it < residuals.size(); it++)
    stats[residual_names[it]] = std::vector<double>();  // initialize

  for (size_t it = 0; it < residuals.size(); it++)
    stats[residual_names[it]].push_back(std::abs(residuals[it]));

  std::cout << "The 25, 50, 75, and 100th percentile residual stats " << tag << std::endl;
  for (auto it = stats.begin(); it != stats.end(); it++) {
    std::string const& name = it->first;
    std::vector<double> vals = stats[name];  // make a copy
    std::sort(vals.begin(), vals.end());

    int len = vals.size();

    int it1 = static_cast<int>(0.25 * len);
    int it2 = static_cast<int>(0.50 * len);
    int it3 = static_cast<int>(0.75 * len);
    int it4 = static_cast<int>(len - 1);

    if (len == 0)
      std::cout << name << ": " << "none";
    else
      std::cout << std::setprecision(5)
                << name << ": " << vals[it1] << ' ' << vals[it2] << ' '
                << vals[it3] << ' ' << vals[it4];
    std::cout << " (" << len << " residuals)" << std::endl;
  }
}

// Evaluate the residuals before and after optimization
void evalResiduals(  // Inputs
  std::string const& tag, std::vector<std::string> const& residual_names,
  // Outputs
  ceres::Problem& problem, std::vector<double>& residuals) {
  double total_cost = 0.0;
  ceres::Problem::EvaluateOptions eval_options;
  eval_options.num_threads = 1;
  eval_options.apply_loss_function = false;  // want raw residuals
  problem.Evaluate(eval_options, &total_cost, &residuals, NULL, NULL);

  // Sanity checks, after the residuals are created
  if (residuals.size() != residual_names.size())
    throw "There must be as many residual names as residual values.";

  calc_residuals_stats(residuals, residual_names, tag);

  return;
}

struct BBox {
    
  Eigen::Vector2d m_min, m_max;
  BBox() {
    double big = std::numeric_limits<double>::max();
    for (int t = 0; t < 2; t++) {
      m_min[t] = big;
      m_max[t] = -big;
    }
  }
  void grow(Eigen::Vector2d const& p) {
    for (int t = 0; t < 2; t++) {
      m_min[t] = std::min(m_min[t], p[t]);
      m_max[t] = std::max(m_max[t], p[t]);
    }
  }
};
  
// Collect a set of pairs of centered distorted and undistorted pixels. Keep
// only the distorted pixels within image domain, and as far from image
// boundary as desired.
// Note that: dist_pix - dist_half_size = distortion_function(undist_pix - undist_size/2)
void genUndistDistPairs(int num_samples, int num_exclude_boundary_pixels,
                        camera::CameraParameters const& cam_params,
                        std::vector<Eigen::Vector2d>& undist_centered_pixels,
                        std::vector<Eigen::Vector2d>& dist_centered_pixels) {
  undist_centered_pixels.clear();
  dist_centered_pixels.clear();

  std::cout << "--temporary box!" << std::endl;
  
  Eigen::Vector2i dist_size        = cam_params.GetDistortedSize();
  Eigen::Vector2i undist_size      = cam_params.GetUndistortedSize();
  Eigen::Vector2d dist_half_size   = cam_params.GetDistortedHalfSize();
  Eigen::Vector2d undist_half_size = cam_params.GetUndistortedHalfSize();

  for (int ix = 0; ix < num_samples; ix++) {
    // Sample uniformly the undistorted width. Ensure values
    // are converted to double before division.
    double x = (undist_size[0] - 1.0) * ix / (num_samples - 1.0);
    for (int iy = 0; iy < num_samples; iy++) {
      // Sample uniformly the undistorted height
      double y = (undist_size[1] - 1.0) * iy / (num_samples - 1.0);

      // Generate an undistorted/distorted point pair using the input model.
      Eigen::Vector2d undist_pix(x, y);

      if (std::abs(undist_pix[0] - undist_half_size[0]) > 430  || 
          std::abs(undist_pix[1] - undist_half_size[1]) > 270)
        continue; 
      
      Eigen::Vector2d dist_pix;
      cam_params.Convert<camera::UNDISTORTED,  camera::DISTORTED>
        (undist_pix, &dist_pix);

      int excl = num_exclude_boundary_pixels;

      if (std::abs(dist_pix[0] - dist_half_size[0]) > 340  || 
          std::abs(dist_pix[1] - dist_half_size[1]) > 160)
          continue; 
      
//       if (dist_pix[0] < excl || dist_pix[0] > dist_size[0] - 1 - excl ||
//           dist_pix[1] < excl || dist_pix[1] > dist_size[1] - 1 - excl)
//         continue;

      // Ensure that these pixels are centered. Here we use the same
      // convention as in camera_params.cc.
      undist_pix -= undist_half_size;
      dist_pix   -= dist_half_size;

      undist_centered_pixels.push_back(undist_pix);
      dist_centered_pixels.push_back(dist_pix);
    }
  }

  BBox undist, dist;
  for (size_t it = 0; it < undist_centered_pixels.size(); it++) {
    undist.grow(undist_centered_pixels[it]);
    dist.grow(dist_centered_pixels[it]);
  }

  std::cout << "--orig undist " << undist.m_min[0] << ' ' << undist.m_min[1] << ' ' << undist.m_max[0] << ' ' << undist.m_max[1] << std::endl;
  std::cout << "--orig dist " << dist.m_min[0] << ' ' << dist.m_min[1] << ' ' << dist.m_max[0] << ' ' << dist.m_max[1] << std::endl;
  return;
}

void fitCurrDegRPC(std::vector<Eigen::Vector2d> const& undist_centered_pixels,
                   std::vector<Eigen::Vector2d> const& dist_centered_pixels,
                   int num_opt_threads, int num_iterations, double parameter_tolerance,
                   bool verbose, Eigen::VectorXd & rpc_coeffs) {
  std::vector<int> block_sizes;
  block_sizes.push_back(rpc_coeffs.size());

  // Form the problem
  ceres::Problem problem;
  std::vector<std::string> residual_names;
  for (size_t it = 0; it < undist_centered_pixels.size(); it++) {
    ceres::CostFunction* rpc_cost_fun =
      RpcFitError::Create(undist_centered_pixels[it], dist_centered_pixels[it],
                          block_sizes);
    ceres::LossFunction* rpc_loss_fun = NULL;

    residual_names.push_back("pix_x");
    residual_names.push_back("pix_y");
    problem.AddResidualBlock(rpc_cost_fun, rpc_loss_fun, &rpc_coeffs[0]);
  }

  if (verbose) {
    Eigen::VectorXd num_x, den_x, num_y, den_y;
    unpack_params(rpc_coeffs, num_x, den_x, num_y, den_y);
    std::cout << "input num_x " << num_x.transpose() << std::endl;
    std::cout << "input den_x " << den_x.transpose() << std::endl;
    std::cout << "input num_y " << num_y.transpose() << std::endl;
    std::cout << "input den_y " << den_y.transpose() << std::endl;
  }

  std::vector<double> residuals;
  evalResiduals("before opt", residual_names, problem, residuals);

  // Solve the problem
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.num_threads = num_opt_threads;  // The result is more predictable with one thread
  options.max_num_iterations = num_iterations;
  options.minimizer_progress_to_stdout = true;
  options.gradient_tolerance = 1e-16;
  options.function_tolerance = 1e-16;
  options.parameter_tolerance = parameter_tolerance;

  if (!verbose)
    options.logging_type = ceres::SILENT;

  ceres::Solve(options, &problem, &summary);

  if (verbose) {
    Eigen::VectorXd num_x, den_x, num_y, den_y;
    unpack_params(rpc_coeffs, num_x, den_x, num_y, den_y);
    std::cout << "output num_x " << num_x.transpose() << std::endl;
    std::cout << "output den_x " << den_x.transpose() << std::endl;
    std::cout << "output num_y " << num_y.transpose() << std::endl;
    std::cout << "output den_y " << den_y.transpose() << std::endl;
  }

  evalResiduals("after opt", residual_names, problem, residuals);
}

void fitRpcDist(int rpc_degree, int num_samples, int num_exclude_boundary_pixels,
                camera::CameraParameters const& cam_params,
                int num_opt_threads, int num_iterations, double parameter_tolerance,
                bool verbose,
                // Output
                Eigen::VectorXd & rpc_dist_coeffs) {

  std::vector<Eigen::Vector2d> undist_centered_pixels, dist_centered_pixels;
  genUndistDistPairs(num_samples, num_exclude_boundary_pixels, cam_params,
                     // Outputs
                     undist_centered_pixels, dist_centered_pixels);

  std::cout << "Found " << dist_centered_pixels.size() << " pixel correspondences "
            << "between undistorted and distorted images within set bounds." << std::endl;

  // First fit RPC of degree 1. Then refine to degree 2, etc, till desired rpc_degree.
  // That is more likely to be successful than aiming right way for the full solution.
  int initial_rpc_degree = 1;
  int init_num_params = num_dist_params(initial_rpc_degree);

  // Set up the initial guess for the variable of optimization
  rpc_dist_coeffs = Eigen::VectorXd::Zero(init_num_params);
  RPCLensDistortion::init_as_identity(rpc_dist_coeffs);  // this changes rpc_dist_coeffs

  std::cout << "\nComputing RPC distortion." << std::endl;
  for (int deg = 1; deg <= rpc_degree; deg++) {
    if (deg >= 2) {
      // Use the previously solved model as an initial guess. Increment its degree by adding
      // to the polynomials new powers of given degree with zero coefficient in front.
      RPCLensDistortion::increment_degree(rpc_dist_coeffs);
    }
    std::cout << "Fitting RPC distortion of degree " << deg << std::endl;
    fitCurrDegRPC(undist_centered_pixels, dist_centered_pixels, num_opt_threads, num_iterations,
                  parameter_tolerance, verbose, rpc_dist_coeffs);
  }
}

void fitRpcUndist(Eigen::VectorXd const & rpc_dist_coeffs,
                  int num_samples, int num_exclude_boundary_pixels,
                  camera::CameraParameters const& cam_params,
                  int num_opt_threads, int num_iterations, double parameter_tolerance,
                  bool verbose,
                  // output
                  Eigen::VectorXd & rpc_undist_coeffs) {

  int rpc_deg = rpc_degree(rpc_dist_coeffs.size());

  // TODO(oalexan1): This is fishy, since it still uses the old
  // distortion models.

  // Also, samples are from all over the undist
  // region. Need to find a good value for that one.
  
  // Also need to ensure we never use points outside the restricted
  // undist region in the calibrator!
  
  // Now that we can model distortion with RPC, find another RPC model
  // which can do undistortion. Ideally,
  // undistort_rpc(distort_rpc(pix)) = pix.
  std::vector<Eigen::Vector2d> undist_centered_pixels, dist_centered_pixels;
  genUndistDistPairs(num_samples, num_exclude_boundary_pixels, cam_params,
                     // Outputs
                     undist_centered_pixels, dist_centered_pixels);

  // Create correspondences. Note that we overwrite
  // dist_centered_pixels which are no longer needed.
  for (size_t it = 0; it < undist_centered_pixels.size(); it++)
    dist_centered_pixels[it] = compute_rpc(undist_centered_pixels[it], rpc_dist_coeffs);

  BBox undist, dist;
  for (size_t it = 0; it < undist_centered_pixels.size(); it++) {
    undist.grow(undist_centered_pixels[it]);
    dist.grow(dist_centered_pixels[it]);
  }
  
  std::cout << "--final undist " << undist.m_min[0] << ' ' << undist.m_min[1] << ' ' << undist.m_max[0] << ' ' << undist.m_max[1] << std::endl;
  std::cout << "--final dist " << dist.m_min[0] << ' ' << dist.m_min[1] << ' ' << dist.m_max[0] << ' ' << dist.m_max[1] << std::endl;
  
  int initial_rpc_degree = 1;
  int init_num_params = num_dist_params(initial_rpc_degree);

  // Set up the initial guess for the variable of optimization
  rpc_undist_coeffs = Eigen::VectorXd::Zero(init_num_params);
  RPCLensDistortion::init_as_identity(rpc_undist_coeffs);  // this changes rpc_undist_coeffs

  // We repeat the code above, but with the order being in reverse:
  // dist pixels being mapped to undist pixels.
  std::cout << "\nComputing RPC distortion." << std::endl;
  for (int deg = 1; deg <= rpc_deg; deg++) {
    if (deg >= 2) {
      // Use the previously solved model as an initial guess. Increment its degree by adding
      // to the polynomials new powers of given degree with zero coefficient in front.
      RPCLensDistortion::increment_degree(rpc_undist_coeffs);
    }
    std::cout << "Fitting RPC undistortion of degree " << deg << std::endl;
    // Note how dist_centered_pixels and undist_centered_pixels are swapped
    fitCurrDegRPC(dist_centered_pixels, undist_centered_pixels, num_opt_threads, num_iterations,
                  parameter_tolerance, verbose, rpc_undist_coeffs);
  }

}

void evalRpcDistUndist(int num_samples, int num_exclude_boundary_pixels,
                       camera::CameraParameters const& cam_params,
                       RPCLensDistortion const& rpc) {

  std::vector<Eigen::Vector2d> undist_centered_pixels, dist_centered_pixels;
  genUndistDistPairs(num_samples, num_exclude_boundary_pixels, cam_params,
                     // Outputs
                     undist_centered_pixels, dist_centered_pixels);

  double max_err = 0.0;
  for (size_t it = 0; it < undist_centered_pixels.size(); it++) {
    Eigen::Vector2d pix = rpc.distort_centered(undist_centered_pixels[it]);
    Eigen::Vector2d pix2 = rpc.undistort_centered(pix);
    max_err = std::max(max_err, (undist_centered_pixels[it] - pix2).norm());
  }

  std::cout << "Max distort_undistort error: " << max_err << std::endl;
}
  
}  // end namespace camera
