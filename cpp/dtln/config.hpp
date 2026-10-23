#pragma once 
#include <complex>6
#include <array>


namespace dtln {

inline constexpr int block_len = 512; 
inline constexpr int hop_len = 128; 
inline constexpr int k_bins = 257;
inline constexpr int k_state = 512;

inline constexpr size_t bytes_mag   = k_bins    * sizeof(float);
inline constexpr size_t bytes_block = block_len * sizeof(float);
inline constexpr size_t bytes_state = k_state   * sizeof(float);


}
