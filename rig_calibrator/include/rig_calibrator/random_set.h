#ifndef RIG_CALIBRATOR_RANDOM_SET_H
#define RIG_CALIBRATOR_RANDOM_SET_H

#include <vector>

namespace dense_map {

  // Given a vector of integers, pick a random subset. Must have
  // output_len <= input_len. The output elements are not sorted.
  // Note: The complexity of this is O(input_len * log(input_len)), which
  // may be not be good enough for some applications.
  void pick_random_subset(int input_len, int output_len,
                          std::vector<int> const& input, std::vector<int> & output);
  
  // Pick unsorted random indices in [0, ..., input_len - 1]
  void pick_random_indices_in_range(int input_len, int output_len,
                                    std::vector<int> & output);

} // namespace dense_map

#endif // RIG_CALIBRATOR_RANDOM_SET_H
