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

#ifndef DENSE_MAP_UTILS_H_
#define DENSE_MAP_UTILS_H_

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

//#include <config_reader/config_reader.h>
#include <camera_model/camera_params.h>

#include <boost/shared_ptr.hpp>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace dense_map {

const int NUM_SCALAR_PARAMS  = 1;  // Used to float single-value params // NOLINT
const int NUM_OPT_CTR_PARAMS = 2;  // optical center in x and y         // NOLINT
const int NUM_PIX_PARAMS     = 2;                                       // NOLINT
const int NUM_XYZ_PARAMS     = 3;                                       // NOLINT
const int NUM_RIGID_PARAMS   = 7;  // quaternion (4) + translation (3)  // NOLINT
const int NUM_AFFINE_PARAMS  = 12; // 3x3 matrix (9) + translation (3)  // NOLINT

const std::string NO_DEPTH_FILE      = "no_depth_file";
const std::string FISHEYE_DISTORTION = "fisheye";
const std::string RADTAN_DISTORTION  = "radtan";
const std::string RPC_DISTORTION     = "rpc";
const std::string NO_DISTORION       = "no_distortion";

// A function to parse a string like
// 'cam1:focal_length,optical_center,distortion cam2:focal_length' and
// extract the intrinsics to float. Separators can be space, comma,
// colon.
void parse_intrinsics_to_float(std::string const& intrinsics_to_float_str,
                               std::vector<std::string> const& cam_names,
                               std::vector<std::set<std::string>>& intrinsics_to_float);

// A  function to split a string like 'haz_cam sci_cam' into
// its two constituents and validate against the list of known cameras.
void parse_camera_names(std::vector<std::string> const& cam_names,
                        std::string const&
                        depth_to_image_transforms_to_float_str,
                        std::set<std::string>&
                        depth_to_image_transforms_to_float);
  
// A  function to split a string like 'haz_cam sci_cam' into
// its two constituents and validate against the list of known cameras.
// Do not allow to float the transform from ref cam to itself, as that
// is the identity.
void parse_rig_transforms_to_float(std::vector<std::string> const& cam_names,
                                   int ref_cam_type,
                                   std::string const& rig_transforms_to_float_str,
                                   std::set<std::string>& rig_transforms_to_float);

// Extract a rigid transform to an array of length NUM_RIGID_PARAMS
void rigid_transform_to_array(Eigen::Affine3d const& aff, double* arr);

// Convert an array of length NUM_RIGID_PARAMS to a rigid
// transform. Normalize the quaternion to make it into a rotation.
void array_to_rigid_transform(Eigen::Affine3d& aff, const double* arr);

void affine_transform_to_array(Eigen::Affine3d const& aff, double* arr);
void array_to_affine_transform(Eigen::Affine3d& aff, const double* arr);

// Convert a string of values separated by spaces to a vector of doubles.
std::vector<double> string_to_vector(std::string const& str);

// Read a 4x4 pose matrix of doubles from disk
void readPoseMatrix(cv::Mat& pose, std::string const& filename);

// Read an affine matrix with double values
bool readAffine(Eigen::Affine3d& T, std::string const& filename);

// Write a matrix with double values
void writeMatrix(Eigen::MatrixXd const& M, std::string const& filename);

void writeCloud(std::vector<float> const& points, size_t point_size, std::string const& filename);

// Two minor and local utility functions
std::string print_vec(double a);
std::string print_vec(Eigen::Vector3d a);

// Return the type of an opencv matrix
std::string matType(cv::Mat const& mat);

// Given two poses aff0 and aff1, and 0 <= alpha <= 1, do linear interpolation.
Eigen::Affine3d linearInterp(double alpha, Eigen::Affine3d const& aff0,
                               Eigen::Affine3d const& aff1);

// Given a set of poses indexed by timestamp in an std::map, find the
// interpolated pose at desired timestamp. This is efficient
// only for very small maps. Else use the StampedPoseStorage class.
bool findInterpPose(double desired_time, std::map<double, Eigen::Affine3d> const& poses,
                    Eigen::Affine3d& interp_pose);

// Implement some heuristic to find the maximum rotation angle that can result
// from applying the given transform. It is assumed that the transform is not
// too different from the identity.
double maxRotationAngle(Eigen::Affine3d const& T);

// A class to store timestamped poses, implementing O(log(n)) linear
// interpolation at a desired timestamp. For fast access, keep the
// poses in bins obtained by flooring the timestamp, which is measured
// in seconds. It is assumed that there are a handful of poses
// measured every second, so in each bin. When bins get large, or too
// many bins are empty, the efficiency of this algorithm goes down.
class StampedPoseStorage {
 public:
  void addPose(Eigen::Affine3d const& pose, double timestamp);

  // Find the interpolated pose by looking up the two poses with
  // closest timestamps that are below and above input_timestamp. If
  // the gap between those timestamps is more than max_gap, return
  // failure, as then likely the interpolation result is not accurate.
  bool interpPose(double input_timestamp, double max_gap, Eigen::Affine3d& out_pose) const;

  void clear();

  bool empty() const;

 private:
  std::map<int, std::map<double, Eigen::Affine3d> > m_poses;
};

// Compute the azimuth and elevation for a (normal) vector
void normalToAzimuthAndElevation(Eigen::Vector3d const& normal, double& azimuth, double& elevation);

// Compute a normal vector based on the azimuth and elevation angles
void azimuthAndElevationToNormal(Eigen::Vector3d& normal, double azimuth, double elevation);

// Snap the normal to the plane (and the plane itself) to make
// all angles multiple of 45 degrees with the coordinate axes.
void snapPlaneNormal(Eigen::Vector3d& plane_normal);

// Find the best fitting plane to a set of points
void bestFitPlane(const std::vector<Eigen::Vector3d>& points, Eigen::Vector3d& centroid, Eigen::Vector3d& plane_normal);

// Extract from a string of the form someDir/1234.5678.jpg the number 123.456.
double fileNameToTimestamp(std::string const& file_name);

// A little holding structure for nav, sci, and haz poses
struct CameraPoses {
  std::map<double, double> haz_depth_to_image_timestamps;
  std::map<std::string, std::map<double, Eigen::Affine3d> > world_to_cam_poses;
};

// Some small utilities for writing a file having poses for nav, sci, and haz cam,
// and also the depth timestamp corresponding to given haz intensity timestamp
void writeCameraPoses(std::string const& filename, std::map<double, double> const& haz_depth_to_image_timestamps,
                      std::map<std::string, std::map<double, Eigen::Affine3d> > const& world_to_cam_poses);

void readCameraPoses(std::string const& filename, std::map<double, double>& haz_depth_to_image_timestamps,
                     std::map<std::string, std::map<double, Eigen::Affine3d> >& world_to_cam_poses);

// Gamma and inverse gamma functions
// https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
double gamma(double x);
double inv_gamma(double x);

// Apply the inverse gamma transform to images, multiply them by
// max_iso_times_exposure/ISO/exposure_time to adjust for
// lightning differences, then apply the gamma transform back.
void exposureCorrection(double max_iso_times_exposure, double iso,
                        double exposure, cv::Mat const& input_image,
                        cv::Mat& output_image);

// Scale an image to correct for lightning variations by taking into
// account that JPEG images have gamma correction applied to them.
// See https://en.wikipedia.org/wiki/Gamma_correction.
void scaleImage(double max_iso_times_exposure, double iso, double exposure,
                cv::Mat const& input_image,
                cv::Mat& output_image);

// Given two bounds, pick two timestamps within these bounds, the one
// closest to the left bound and the one to the right bound. Take into
// account that the timestamps may need to have an offset added to
// them. Assume that the input timestamps are sorted in increasing order.
// TODO(oalexan1): May have to add a constraint to only pick
// a timestamp if not further from the bound than a given value.
void pickTimestampsInBounds(std::vector<double> const& timestamps, double left_bound,
                            double right_bound, double offset,
                            std::vector<double>& out_timestamps);

// Must always have NUM_EXIF the last.
enum ExifData { TIMESTAMP = 0, EXPOSURE_TIME, ISO, APERTURE, FOCAL_LENGTH, NUM_EXIF };

// A utility for saving a camera in a format ASP understands.
// TODO(oalexan1): Expose the sci cam intrinsics rather than having
// them hard-coded.
void saveTsaiCamera(Eigen::MatrixXd const& desired_cam_to_world_trans,
                      std::string const& output_dir,
                      double curr_time, std::string const& suffix);

// Write an image with 3 floats per pixel. OpenCV's imwrite() cannot do that.
void saveXyzImage(std::string const& filename, cv::Mat const& img);

// Read an image with 3 floats per pixel. OpenCV's imread() cannot do that.
void readXyzImage(std::string const& filename, cv::Mat & img);

// Forward declaration
struct cameraImage;

// Create the image and depth cloud file names
void genImageAndDepthFileNames(  // Inputs
  std::vector<cameraImage> const& cams, std::vector<std::string> const& cam_names,
  std::string const& out_dir,
  // Outputs
  std::vector<std::string>& image_files, std::vector<std::string>& depth_files);

// Save images and depth clouds to disk
void saveImagesAndDepthClouds(std::vector<cameraImage> const& cams);

// A struct to collect together some attributes of an image or depth cloud
// (stored as an image with 3 channels)
struct ImageMessage {
  cv::Mat image;
  double timestamp;
  std::string name;
  Eigen::Affine3d world_to_cam;
};

// Find an image at the given timestamp or right after it. We assume
// that during repeated calls to this function we always travel
// forward in time, and we keep track of where we are in the vector using
// the variable beg_pos that we update as we go.
bool lookupImage(double desired_time, std::vector<ImageMessage> const& msgs,
                 cv::Mat& image, std::string & image_name, int& beg_pos, double& found_time);

// Convert a string of space-separated numbers to a vector
void strToVec(std::string const& str, std::vector<double> & vec);
  
// Read the images, depth clouds, and their metadata
// Save the properties of images. Use space as separator.
void writeImageList(std::string const& out_dir, std::vector<dense_map::cameraImage> const& cams,
                    std::vector<Eigen::Affine3d> const& world_to_cam);

// Save the optimized rig configuration
void writeRigConfig(std::string const& out_dir, bool model_rig, int ref_cam_type,
                    std::vector<std::string> const& cam_names,
                    std::vector<camera::CameraParameters> const& cam_params,
                    std::vector<Eigen::Affine3d> const& ref_to_cam_trans,
                    std::vector<Eigen::Affine3d> const& depth_to_image,
                    std::vector<double> const& ref_to_cam_timestamp_offsets);
  
// Read a rig configuration. Check if the transforms among the sensors
// on the rig is not 0, in that case will use it.
void readRigConfig(std::string const& rig_config, bool have_rig_transforms, int & ref_cam_type,
                   std::vector<std::string> & cam_names,
                   std::vector<camera::CameraParameters> & cam_params,
                   std::vector<Eigen::Affine3d> & ref_to_cam_trans,
                   std::vector<Eigen::Affine3d> & depth_to_image,
                   std::vector<double> & ref_to_cam_timestamp_offsets);
  
void ReadNVM(std::string const& input_filename,
             std::vector<Eigen::Matrix2Xd> * cid_to_keypoint_map,
             std::vector<std::string> * cid_to_filename,
             std::vector<std::map<int, int> > * pid_to_cid_fid,
             std::vector<Eigen::Vector3d> * pid_to_xyz,
             std::vector<Eigen::Affine3d> *
             cid_to_cam_t_global);

// Save the depth clouds and optimized transforms needed to create a mesh with voxblox
// (if depth clouds exist).
void exportToVoxblox(std::vector<std::string> const& cam_names,
                     std::vector<dense_map::cameraImage> const& cam_images,
                     std::vector<Eigen::Affine3d> const& depth_to_image,
                     std::vector<Eigen::Affine3d> const& world_to_cam,
                     std::string const& out_dir);

void saveTransformedDepthClouds(std::vector<std::string> const& cam_names,
                                std::vector<dense_map::cameraImage> const& cam_images,
                                std::vector<Eigen::Affine3d> const& depth_to_image,
                                std::vector<Eigen::Affine3d> const& world_to_cam,
                                std::string const& out_dir);
  
}  // namespace dense_map

#endif  // DENSE_MAP_UTILS_H_
