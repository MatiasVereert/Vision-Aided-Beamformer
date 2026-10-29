#pragma once 
#include <ofb/config.hpp>
#include <array>

namespace ofb {


// Custom Data Types 

// Single Bin
using Vec = std::array<cplx_t, M> ; // (M,1)
using Vec_real = std::array<real_t, M> ; // (M,1)
using Mat = std::array<Vec, M> ; // (M,M)

// Multi Bin 
using Frame = std::array<Vec, K> ; //(K, M)
using Reals = std::array<real_t, K> ; //(K,)



// Single mic 
using Frame_real= std::array<real_t, block_len>;
using Frame_complex = std::array<cplx_t, K>;
using Frame_mag = std::array<real_t, K>;

// Real multi mic
using Frame_real_multi =  std::array<Frame_real, M>; //(M, t)

// Un hop: lo que entrega un periodo del DMA
using Hop_real       = std::array<real_t, hop_len>;  //(hop,)
using Hop_real_multi = std::array<Hop_real, M>;      //(M, hop)


// Helpers 
constexpr Vec fill_with(real_t v){
  Vec a{};
  for (int m = 0; m < M; ++m) a[m] = v;
  return a;
}

// Version para (K,): la usa Den_SS/Den_NN. Nombre distinto porque el tipo de
// retorno no alcanza para sobrecargar (Vec y Reals son dos array<T,N> con N
// y T distintos, pero la resolucion de sobrecarga no mira el tipo de retorno).
constexpr Reals fill_reals(real_t v){
  Reals a{};
  for (int k = 0; k < K; ++k) a[k] = v;
  return a;
}


inline constexpr Mat Mat_scalar_div( Mat R, real_t scalar){
  real_t scalar_inv = 1 / scalar; 
  for (int i = 0; i<M; i++){
    for (int j = 0; j<M; j++){
      R[i][j] = scalar_inv *  R[i][j]; 
    }
  }
  return R;
}

}// ofb namespace