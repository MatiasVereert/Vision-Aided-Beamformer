#pragma once 
#include <complex>
#include <array>
#include <cstddef>


namespace dtln {

inline constexpr int block_len = 512; 
inline constexpr int hop_len = 128; 
inline constexpr int k_bins = 257;
inline constexpr int k_state = 512;   // 2 capas * 128 unidades * (h,c). Que de 512
                                      // igual que block_len es CASUALIDAD.

// Tamanos en BYTES para TfLiteTensorCopyFrom/ToBuffer, que no trabaja en
// elementos. Con nombres distintos no se pueden confundir k_state y block_len.
inline constexpr std::size_t bytes_mag   = k_bins    * sizeof(float);   //  1028
inline constexpr std::size_t bytes_block = block_len * sizeof(float);   //  2048
inline constexpr std::size_t bytes_state = k_state   * sizeof(float);   //  2048

}
