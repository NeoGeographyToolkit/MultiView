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

#ifndef RIG_CALIBRATOR_BASIC_ALGS_H
#define RIG_CALIBRATOR_BASIC_ALGS_H

// Some basic algorithms on STL data structures

#include <glog/logging.h>
#include <map>
#include <algorithm>

namespace dense_map {

// Look up a map value and throw an error when not found
template<class A, class B>
B mapVal(std::map<A, B> const& map, A const& a) {
  auto it = map.find(a);
  if (it == map.end())
    LOG(FATAL) << "Cannot look up expected map value.\n";

  return it->second;
}

// Get a map value while being const-correct and also checking that the value exists.
template <class T>
T getMapValue(std::vector<std::map<int, std::map<int, T>>> const& pid_cid_fid,
              size_t pid, int cid, int fid) {
  if (pid_cid_fid.size() <= pid)
    LOG(FATAL) << "Current pid is out of range.\n";

  auto& cid_fid_map = pid_cid_fid[pid];  // alias
  auto cid_it = cid_fid_map.find(cid);
  if (cid_it == cid_fid_map.end()) LOG(FATAL) << "Current cid it out of range.\n";

  auto& fid_map = cid_it->second;  // alias
  auto fid_it = fid_map.find(fid);
  if (fid_it == fid_map.end()) LOG(FATAL) << "Current fid is out of range.\n";

  return fid_it->second;
}

// Set a map value while taking care that the place for it exists.
template <class T>
void setMapValue(std::vector<std::map<int, std::map<int, T>>> & pid_cid_fid,
                 size_t pid, int cid, int fid, int val) {
  if (pid_cid_fid.size() <= pid)
    LOG(FATAL) << "Current pid is out of range.\n";

  auto& cid_fid_map = pid_cid_fid[pid];  // alias
  auto cid_it = cid_fid_map.find(cid);
  if (cid_it == cid_fid_map.end()) LOG(FATAL) << "Current cid it out of range.\n";

  auto& fid_map = cid_it->second;  // alias
  auto fid_it = fid_map.find(fid);
  if (fid_it == fid_map.end()) LOG(FATAL) << "Current fid is out of range.\n";

  fid_it->second = val;
}

std::string file_extension(std::string const& file);

// Maximum value in a map
template<typename K, typename V>
V maxMapVal(const std::map<K,V> &map) {
  if (map.empty()) 
    LOG(FATAL) << "Cannot find the maximum value in an empty map.\n";
  
  auto key = std::max_element(map.begin(), map.end(),
                              [](std::pair<K,V> const &x, std::pair<K,V> const &y) {
                                return x.second < y.second;
                              });
  return key->second;
}
  
}  // end namespace dense_map

#endif  // RIG_CALIBRATOR_BASIC_ALGS_H
