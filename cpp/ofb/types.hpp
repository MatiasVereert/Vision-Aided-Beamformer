#pragma once 
#include <ofb/config.hpp>
#include <array>

namespace ofb {


// Custom Data Types 

// Single Bin
using Vec = std::array<cplx_t, M> ; // (M,1)
using Mat = std::array<Vec, M> ; // (M,M)

// Multi Bin 
using Frame = std::array<Vec, K> ; //(K, M)
using Reals = std::array<real_t, K> ; //(K,)


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