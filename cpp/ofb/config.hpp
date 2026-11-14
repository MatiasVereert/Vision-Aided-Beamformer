#pragma once
#include <complex>

namespace ofb {

// M se fija al COMPILAR con -DOFB_M=<M>, como el P de abajo. 10 = el aro de 12
// con dos capsulas anuladas (mic-array-platform, bitacora 34): quien llama elige
// QUE canales entran. kRefMic = M/2 es un indice en ESA seleccion.
#ifndef OFB_M
#define OFB_M 12
#endif
inline constexpr int M = OFB_M;
inline constexpr int kFft  = 512;
inline constexpr int K = kFft / 2 + 1;    // 257
inline constexpr int kHop  = 128;
inline constexpr int kRefMic = M / 2;
inline constexpr int kFs   = 16000;           // fijo: no varia con el array

using real_t = float;
using cplx_t = std::complex<real_t>;

inline constexpr real_t kAlpha      = 0.99f;
inline constexpr real_t kMinLoading = 1e-5f;  // OJO: 1e-9 desaparece en float
inline constexpr real_t kSharpen    = 8.0f;

// P: cada bin se recalcula cada P frames (solve ciclico, K/P solves por frame).
// Se fija al COMPILAR con -DOFB_BLOCK_UPD=<P>, sin tocar este archivo: la Kria
// usa 2 (medido: hop de 6.9 -> 4.2 ms, mismo DNSMOS). El default es 1 porque es
// el unico valor en el que el C++ coincide con ofb.py -- con P > 1 Python
// recalcula TODOS los bins cada P frames -- y test_ofb lo exige.
#ifndef OFB_BLOCK_UPD
#define OFB_BLOCK_UPD 1
#endif
inline constexpr int    kBlockUpd   = OFB_BLOCK_UPD;

// --- ISIRTracker + agenda del post-filtro (Fase 2) -------------------------
// ISIR_BAND_HZ de ofb.py = 300-3400 Hz. Los limites de bin se resuelven en
// tiempo de COMPILACION con aritmetica ENTERA (no float) para no depender del
// redondeo de una division constexpr en punto flotante:
//   k esta en la banda  <=>  k*kFs/kFft in [lo_hz, hi_hz]
//                       <=>  k*kFs in [lo_hz*kFft, hi_hz*kFft]
// kIsirBinLo = ceil(lo_hz*kFft/kFs), kIsirBinHi = floor(hi_hz*kFft/kFs)
// (INCLUSIVE, como el `slice` de Python). Con kFft=512, kFs=16000 da
// [10, 108], 99 bins -- coincide con el numero que trae el docstring de
// ISIRTracker en ofb.py, que es la confirmacion de que fs=16000 es correcto.
inline constexpr int kIsirLoHz = 300;
inline constexpr int kIsirHiHz = 3400;
inline constexpr int kIsirBinLo = (kIsirLoHz * kFft + kFs - 1) / kFs;
inline constexpr int kIsirBinHi = (kIsirHiHz * kFft) / kFs;

inline constexpr real_t kSmooth      = 0.2f;
inline constexpr real_t kIsirAlpha   = 0.998f;
inline constexpr real_t kIsirCalibG  = 0.496f;
inline constexpr real_t kIsirCalibB  = 3.77f;
inline constexpr real_t kIsirCenter  = 2.2f;
inline constexpr real_t kIsirWidth   = 3.5f;

static_assert(kRefMic >= 0 && kRefMic < M, "ref_mic fuera de rango");
static_assert(kBlockUpd >= 1 && kBlockUpd <= K, "OFB_BLOCK_UPD tiene que estar en [1, K]");
static_assert(K == kFft / 2 + 1, "K tiene que salir de kFft");
static_assert(kIsirBinLo <= kIsirBinHi && kIsirBinHi < K,
              "la banda del ISIR tiene que caer dentro de los bins validos");
//dtln 
inline constexpr int block_len = 512; 
inline constexpr int hop_len = 128; 
inline constexpr int k_bins = 257;
inline constexpr int k_state = 512;

inline constexpr size_t bytes_mag   = k_bins    * sizeof(float);
inline constexpr size_t bytes_block = block_len * sizeof(float);
inline constexpr size_t bytes_state = k_state   * sizeof(float);



}  // namespace ofb