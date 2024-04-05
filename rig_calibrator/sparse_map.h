#ifndef RIG_CALIBRATOR_SPARSE_MAP_H_
#define RIG_CALIBRATOR_SPARSE_MAP_H_

#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>

#include <camera_model/camera_model.h>
#include <rig_calibrator/matching.h>
#include <rig_calibrator/interest_point.h>

#include <numeric>
#include <set>

namespace cv {
  class Mat;
  class DMatch;
}

// TODO(oalexan1): Wipe this class

namespace sparse_mapping {

/**
 * A class representing a sparse map, which consists of a collection
 * of keyframes and detected features. To localize, an image's features
 * are matched to the keyframes in the map. They keyframe features have known
 * positions and the camera pose can be estimated with ransac.
 **/
struct SparseMap {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Create an empty map. It is strongly recommended to not use this function,
  // as it requires carefully initializing many members. Hence this is made
  // private. Consider using the other constructors.
  SparseMap() {}

  /**
   * Constructs a new sparse map from a list of image files and their
   * associate keypoint and descriptor files. If use_cached_features
   * is set to false, it reads the image files and performs feature
   * detection instead. Does not perform bundle adjustment.
   **/
  SparseMap(const std::vector<std::string> & filenames,
            const std::string & detector,
            const camera::CameraParameters & params);

  /**
   * Constructs a new sparse map from a protobuf file, with specified
   * vocabulary tree and optional parameters.
   **/
  SparseMap(const std::string & protobuf_file,
            bool localization = false);

  /**
     Form a sparse map with given cameras/images, and no features
  **/
  SparseMap(const std::vector<Eigen::Affine3d>& cid_to_cam_t,
            const std::vector<std::string> & filenames,
            const std::string & detector,
            const camera::CameraParameters & params);


  SparseMap(bool bundler_format, std::string const& filename,
            std::vector<std::string> const& files);

  void SetDetectorParams(int min_features, int max_features, int retries,
                         double min_thresh, double default_thresh, double max_thresh);

  /**
   * Detect features in given images
   **/
  void DetectFeatures();

  // access map frames
  /**
   * Get the number of keyframes in the map.
   **/
  size_t GetNumFrames(void) const {return cid_to_filename_.size();}
  /**
   * Get the filename of a keyframe in the map.
   **/
  const std::string & GetFrameFilename(int frame) const {return cid_to_filename_[frame];}
  /**
   * Get the global camera transform for a keyframe in the map.
   **/
  const Eigen::Affine3d & GetFrameGlobalTransform(int frame) const
        {return cid_to_cam_t_global_[frame];}

  void SetFrameGlobalTransform(int frame, const Eigen::Affine3d & transform) {
    cid_to_cam_t_global_[frame] = transform;
  }
  /**
   * Get the keypoint coordinates in the specified frame.
   **/
  const Eigen::Matrix2Xd & GetFrameKeypoints(int frame) const {return cid_to_keypoint_map_[frame];}
  /**
   * Get the descriptor for a frame and feature.
   **/
  cv::Mat GetDescriptor(int frame, int fid) const { return cid_to_descriptor_map_[frame].row(fid);}
  /**
   * Returns map of feature ids to landmark ids for the specified frame.
   **/
  const std::map<int, int> & GetFrameFidToPidMap(int frame) const {return cid_fid_to_pid_[frame];}

  // access map landmarks
  /**
   * Get the number of landmark points in the map.
   **/
  size_t GetNumLandmarks(void) const {return pid_to_xyz_.size();}
  /**
   * Get the global position of the specified landmark.
   **/
  Eigen::Vector3d GetLandmarkPosition(int landmark) const {return pid_to_xyz_[landmark];}
  /**
   * Return a map for a specified landmark, matching the ids of all the keyframes that landmark
   * was seen in to the feature id within that frame.
   **/
  const std::map<int, int> & GetLandmarkCidToFidMap(int landmark) const {return pid_to_cid_fid_[landmark];}

  // access and modify parameters
  /**
   * Set the number of RANSAC iterations.
   **/
  void SetRansacIterations(int iterations) {num_ransac_iterations_ = iterations;}
  /**
   * Return the number of RANSAC iterations.
   **/
  int GetRansacIterations(void) const {return num_ransac_iterations_;}
  /**
   * Set the RANSAC inlier tolerance, the number of pixels an inlier
   * feature is allowed to be off by.
   **/
  void SetRansacInlierTolerance(int tolerance) {ransac_inlier_tolerance_ = tolerance;}
  /**
   * Get the RANSAC inlier tolerance, the number of pixels an inlier
   * feature is allowed to be off by.
   **/
  int GetRansacInlierTolerance(void) const {return ransac_inlier_tolerance_;}
  /**
   * Set the number of early break landmarks, when to stop in adding landmarks when localizing.
   **/
  void SetEarlyBreakLandmarks(int early_break_landmarks) {early_break_landmarks_ = early_break_landmarks;}
  void SetHistogramEqualization(int histogram_equalization) {histogram_equalization_ = histogram_equalization;}
  int GetHistogramEqualization() {return histogram_equalization_;}
  /**
   * Return the parameters of the camera used to construct the map.
   **/
  camera::CameraParameters GetCameraParameters(void) const {return camera_params_;}
  void SetCameraParameters(camera::CameraParameters camera_params) {camera_params_ = camera_params;}
  /**
   * Return the number of observations. Use this number to divide the final error to find the average pixel error.
   **/
  size_t GetNumObservations(void) const {return std::accumulate(pid_to_cid_fid_.begin(),
                                                                pid_to_cid_fid_.end(),
                                                                0, [](size_t v,
                                                                      std::map<int, int> const& map)
                                                                { return v + map.size(); }); }
  /**
   * Return the transform to real world coordinates.
   **/
  Eigen::Affine3d GetWorldTransform() const {return world_transform_;}

  /**
   * Apply given transform to camera positions and 3D points
   **/
  void ApplyTransform(Eigen::Affine3d const& T) {
    dense_map::TransformCamerasAndPoints(T, &cid_to_cam_t_global_, &pid_to_xyz_);
  }

  // Load map. If localization is true, load only the parts of the map
  // needed for localization.
  void Load(const std::string & protobuf_file, bool localization = false);

  // construct from pid_to_cid_fid
  void InitializeCidFidToPid();

  // detect features with opencv
  void DetectFeaturesFromFile(std::string const& filename,
                              bool multithreaded,
                              cv::Mat* descriptors,
                              Eigen::Matrix2Xd* keypoints);
  void DetectFeatures(cv::Mat const& image,
                      bool multithreaded,
                      cv::Mat* descriptors,
                      Eigen::Matrix2Xd* keypoints);
  // delete feature descriptors with no matching landmark
  void PruneMap(void);

  std::string GetDetectorName() { return detector_.GetDetectorName(); }

  // stored in map file
  std::vector<std::string> cid_to_filename_;
  // TODO(bcoltin) replace Eigen2Xd everywhere with one keypoint class
  std::vector<Eigen::Matrix2Xd > cid_to_keypoint_map_;
  std::vector<std::map<int, int> > pid_to_cid_fid_;
  std::vector<Eigen::Vector3d> pid_to_xyz_;
  std::vector<Eigen::Affine3d > cid_to_cam_t_global_;
  std::vector<cv::Mat> cid_to_descriptor_map_;
  // generated on load
  std::vector<std::map<int, int> > cid_fid_to_pid_;

  interest_point::FeatureDetector detector_;
  camera::CameraParameters camera_params_;
  int num_similar_;
  int num_ransac_iterations_;
  int ransac_inlier_tolerance_;
  int early_break_landmarks_;
  int histogram_equalization_;

  // e.g, 10th db image is 3rd image in cid_to_filename_
  std::map<int, int> db_to_cid_map_;

  // If datastructure is available, match only pairs of cids
  // that are present in it (this info can come from example from
  // a map that was built previously with the same images but
  // a different descriptor.
  std::map<int, std::set<int>> cid_to_cid_;  // TODO(oalexan1): Need not be a member

  // These are used to register the map to real world coordinates
  // with information provided by the user.
  // TODO(oalexan1): These need not be members
  Eigen::Affine3d world_transform_;
  std::vector<Eigen::Matrix2Xd> user_cid_to_keypoint_map_;
  std::vector<std::map<int, int> > user_pid_to_cid_fid_;
  std::vector<Eigen::Vector3d> user_pid_to_xyz_;
  std::mutex mutex_detector_;

 private:

  // I found out the hard way that sparse maps cannot be copied
  // correctly, hence prohibit this. The only good way seems to be to
  // load a copy from disk. (oalexan1)
  SparseMap(SparseMap &);
  SparseMap& operator=(const SparseMap&);

  // Reorder the images in the map and the rest of the data accordingly
  void reorderMap(std::map<int, int> const& old_cid_to_new_cid);
};

}  // namespace sparse_mapping

#endif  // RIG_CALIBRATOR_SPARSE_MAP_H_
