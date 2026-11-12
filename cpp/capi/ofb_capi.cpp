// ofb_capi: OfbMvdr detras de una API de C, para cargarlo desde Python con
// ctypes (mic-array-platform/tools/ofb_gui: la REPLICA del OFB de la Kria).
//
// Por que C + ctypes y no pybind11: son cinco funciones con buffers planos de
// float32, no hace falta ninguna dependencia de compilacion nueva, y ctypes
// SUELTA EL GIL durante la llamada -> la replica corre en su hilo sin frenar
// al resto del proceso (red, aplay, WebSocket).
//
// Se compila con el MISMO P que la Kria (OFB_CAPI_BLOCK_UPD, default 2): la
// replica tiene que hacer la misma cuenta que el server, no la de ofb.py.
//
// Disposicion de los buffers (todo float32, C-contiguo):
//   hop_in   (n_hops, M, kHop)
//   out      (n_hops, kHop)
//   diag     (n_hops, OFB_DIAG_LEN) o NULL; cada fila:
//              [0K, 2K)  x_ref  complejo (re, im intercalados)
//              [2K, 4K)  y_bf   complejo, w^H x antes del post-filtro
//              [4K, 6K)  y_pf   complejo, despues del post-filtro
//              [6K, 7K)  m_out   [7K, 8K) m_ref   [8K, 9K) m_pf
//              9K        c_pf    9K+1     isir_db
//   weights  (K, M) complejo                         = 2*K*M floats
//   phi_ss, phi_nn  (K, M, M) complejo, hermitiana COMPLETA, Num/Den sin carga
#include <ofb/ofb.hpp>
#include <cstring>
#include <exception>
#include <memory>

using namespace ofb;

namespace {

constexpr int kDiagLen = 9 * K + 2;

struct Handle {
  std::unique_ptr<OfbMvdr> bf;
  FrameDiag diag;
};

void copy_cplx(float* dst, const Frame_complex& src){
  static_assert(sizeof(cplx_t) == 2 * sizeof(float), "complex<float> = float[2]");
  std::memcpy(dst, src.data(), sizeof(cplx_t) * K);
}

void copy_real(float* dst, const Frame_mag& src){
  std::memcpy(dst, src.data(), sizeof(real_t) * K);
}

// Num/Den empaquetado -> (M, M) completa. La guarda de Den es la de solve_bin.
void unpack_full(const TriMat& num, real_t den, cplx_t* out){
  const real_t inv = 1 / (den + 1e-15f);
  for (int i = 0; i < M; i++)
    for (int j = 0; j <= i; j++){
      const cplx_t v = inv * num[tri(i, j)];
      out[i * M + j] = v;
      out[j * M + i] = std::conj(v);
    }
}

}  // namespace

extern "C" {

// Constantes de compilacion, para que Python no las asuma.
void ofb_dims(int* m, int* k, int* hop, int* fft, int* ref_mic, int* block_upd,
              int* diag_len){
  *m = M; *k = K; *hop = kHop; *fft = kFft; *ref_mic = kRefMic;
  *block_upd = kBlockUpd; *diag_len = kDiagLen;
}

// NULL si falla; el motivo queda en err (si err != NULL).
void* ofb_create(const char* model_path, char* err, int err_len){
  try {
    auto* h = new Handle;
    h->bf = std::make_unique<OfbMvdr>(model_path);
    h->bf->set_diag(&h->diag);
    return h;
  } catch (const std::exception& e){
    if (err && err_len > 0){ std::strncpy(err, e.what(), err_len - 1); err[err_len - 1] = 0; }
    return nullptr;
  }
}

void ofb_destroy(void* p){ delete static_cast<Handle*>(p); }

void ofb_process_hops(void* p, int n_hops, const float* hop_in, float* out, float* diag){
  auto* h = static_cast<Handle*>(p);
  Hop_real_multi in{};
  for (int n = 0; n < n_hops; n++){
    for (int m = 0; m < M; m++)
      std::memcpy(in[m].data(), hop_in + (size_t(n) * M + m) * kHop, sizeof(float) * kHop);
    const Hop_real y = h->bf->process_hop(in);
    std::memcpy(out + size_t(n) * kHop, y.data(), sizeof(float) * kHop);
    if (diag){
      float* d = diag + size_t(n) * kDiagLen;
      const FrameDiag& g = h->diag;
      copy_cplx(d,         g.x_ref);
      copy_cplx(d + 2 * K, g.y_bf);
      copy_cplx(d + 4 * K, g.y_pf);
      copy_real(d + 6 * K, g.m_out);
      copy_real(d + 7 * K, g.m_ref);
      copy_real(d + 8 * K, g.m_pf);
      d[9 * K]     = g.c_pf;
      d[9 * K + 1] = g.isir_db;
    }
  }
}

void ofb_get_weights(void* p, float* out){
  const Frame& w = static_cast<Handle*>(p)->bf->current_weights();
  for (int k = 0; k < K; k++)
    std::memcpy(out + size_t(k) * M * 2, w[k].data(), sizeof(cplx_t) * M);
}

void ofb_get_cov(void* p, float* phi_ss, float* phi_nn){
  const SoudenCore& c = static_cast<Handle*>(p)->bf->core();
  auto* ss = reinterpret_cast<cplx_t*>(phi_ss);
  auto* nn = reinterpret_cast<cplx_t*>(phi_nn);
  for (int k = 0; k < K; k++){
    if (ss) unpack_full(c.num_ss()[k], c.den_ss()[k], ss + size_t(k) * M * M);
    if (nn) unpack_full(c.num_nn()[k], c.den_nn()[k], nn + size_t(k) * M * M);
  }
}

// Den_SS / Den_NN (K,): la masa de mascara que acumulo cada rama. Phi * Den =
// Num es la energia ATRIBUIDA a cada rama, no el promedio condicional.
void ofb_get_den(void* p, float* den_ss, float* den_nn){
  const SoudenCore& c = static_cast<Handle*>(p)->bf->core();
  std::memcpy(den_ss, c.den_ss().data(), sizeof(real_t) * K);
  std::memcpy(den_nn, c.den_nn().data(), sizeof(real_t) * K);
}

}  // extern "C"
