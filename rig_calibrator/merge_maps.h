#ifndef RIG_CALIBRATOR_MERGE_MAPS_H_
#define RIG_CALIBRATOR_MERGE_MAPS_H_

namespace camera {
  class CameraParameters;
}

namespace dense_map {

class nvmData;
  
// Merge two maps
void MergeMaps(dense_map::nvmData const& A_in,
               dense_map::nvmData const& B_in,
               dense_map::RigSet const& R,
               int num_image_overlaps_at_endpoints,
               bool fast_merge,
               bool no_transform,
               double close_dist,
               std::string const& image_sensor_list, 
               dense_map::nvmData & C_out);

}  // namespace dense_map

#endif  // RIG_CALIBRATOR_MERGE_MAPS_H_
