#pragma once 
#include "kiss_fft.h"
#include "kiss_fftr.h"
#include <algorithm>
#include <array>
#include <cmath> 
#include <complex>
#include <stdexcept>
#include <ofb/config.hpp>
#include <ofb/isir_tracker.hpp>
#include <ofb/solve.hpp>
#include <ofb/types.hpp>
#include <ofb/dtln_mask.hpp>


namespace ofb{

// La STFT del beamformer (kFft/kHop) y los buffers (block_len/hop_len) son el
// mismo numero escrito dos veces en config.hpp; esto avisa si se separan.
static_assert(kFft == block_len && kHop == hop_len,
              "kFft/kHop y block_len/hop_len tienen que coincidir");
static_assert(kFft % kHop == 0, "el OLA asume kFft multiplo de kHop");

class OfbMvdr {
  private:
  DtlnModel shared_;      // 1ro declarado -> ULTIMO destruido
  DtlnMask  dtln_out_;    // 2do
  DtlnMask  dtln_ref_;    // 3ro declarado -> PRIMERO destruido

  SoudenCore   mvdr_solve_;
  isir_tracker isir_;

  //fft 
  kiss_fftr_cfg fftr_cfg_  = nullptr; 
  kiss_fftr_cfg ifftr_cfg_ = nullptr; 

  // Arranque: w = e_ref -> Y(0) = x_ref. Con el nucleo sobre estado nulo daria
  // w = 0 y la red se quedaria sin nada que mirar en el primer frame.
  Frame weights{};
  int   t_ = 0;

  // Diagnostico del ultimo frame (el lazo no los lee).
  real_t isir_db_ = 0.0f;
  real_t c_pf_    = 0.0f;

  // STFT de streaming (process_hop). Analisis RECTANGULAR: la ventana es el
  // buffer tal cual, que es lo que deja al frame y al bloque del DTLN siendo
  // las mismas muestras. Sintesis con taper de Hann, como ola_taper.
  Frame_real_multi in_buf_{};   // (M, kFft): los ultimos kFft de cada mic
  Frame_real       ola_{};      // acumulador de sintesis; ola_[0] = la mas vieja
  Frame_real       w_syn_{};    // Hann periodica / su suma OLA

  public:
    OfbMvdr(const char* model_path)
      : shared_(model_path), dtln_out_(shared_), dtln_ref_(shared_)
    { 
      // set fft config: DOS configuraciones distintas, directa e inversa
      fftr_cfg_  = kiss_fftr_alloc( block_len, 0, nullptr, nullptr);
      ifftr_cfg_ = kiss_fftr_alloc( block_len, 1, nullptr, nullptr);
      if (fftr_cfg_ == nullptr || ifftr_cfg_ == nullptr)
        throw std::runtime_error("kiss_fftr_alloc fallo");

      // init weights
      for( int k = 0; k < K; k++)
        weights[k][kRefMic] = 1.0f;

      // Ventana de sintesis: Hann PERIODICA (fftbins=True de scipy) dividida
      // por la media de su suma OLA, igual que ola_taper. Asi
      // sum_t w(n - t*kHop) = 1 y sin tocar Y la reconstruccion es exacta.
      // En double: con Hann y kFft/kHop = 4 la suma es 2, pero se calcula para
      // no depender de eso si cambia el hop.
      constexpr double kPi = 3.14159265358979323846;
      std::array<double, kFft> w{};
      for (int n = 0; n < kFft; ++n)
        w[n] = 0.5 - 0.5 * std::cos(2.0 * kPi * n / kFft);
      double cola = 0.0;
      for (int n = 0; n < kFft; ++n)
        for (int s = 0; s < kFft; s += kHop)
          cola += w[(n + s) % kFft];
      cola /= kFft;
      for (int n = 0; n < kFft; ++n)
        w_syn_[n] = static_cast<real_t>(w[n] / cola);

      shared_.finalize();            // en el CUERPO: los miembros ya existen
    }

    ~OfbMvdr(){
      kiss_fftr_free(fftr_cfg_);
      kiss_fftr_free(ifftr_cfg_);
    }

    OfbMvdr(const OfbMvdr&)            = delete;
    OfbMvdr& operator=(const OfbMvdr&) = delete;

    // Streaming: entra un hop de M canales, sale un hop de la salida. Es la
    // unidad que encaja con un periodo del DMA (128 frames = kHop).
    // La salida esta ATRASADA kFft - kHop muestras (384 = 24 ms) respecto de
    // la entrada: el hop que sale es el mas viejo del frame, el unico al que
    // ya le sumaron los kFft/kHop frames que lo cubren.
    Hop_real process_hop( const Hop_real_multi& hop_in );

    // Un frame de la STFT: (M, kFft) -> irfft de Y, SIN ventana de sintesis.
    // Publica para el test de paridad contra ofb.py; el streaming usa
    // process_hop.
    Frame_real procces_frame( const Frame_real_multi& input_frame );

    // diagnostico
    real_t isir_db() const { return isir_db_; }
    real_t c_pf()    const { return c_pf_;    }
  };


  inline Frame_real OfbMvdr::procces_frame( const Frame_real_multi& input_frame ){
    Frame_real    output_frame{};
    Frame         X{};
    Frame_complex Y{};
    Frame_complex frame_ref{};
    Frame_mag     m_out{}, m_ref{}, m_scm{}, m_pf{};
    Reals         m_s{}, m_n{}, px{};
    std::array<kiss_fft_cpx, k_bins> bins{};

    // ---- 1. FFT por canal: (M, 512) tiempo -> X (K, M) frecuencia -----------
    for( int m = 0; m < M; m++){
      kiss_fftr(fftr_cfg_, input_frame[m].data(), bins.data());
      for (int k = 0; k < K; ++k)
        X[k][m] = { bins[k].r, bins[k].i };      // transpuesta: (M,K) -> (K,M)
    }

    // ---- 2. CAMINO CRITICO: Y = w^H x, con los pesos del frame ANTERIOR -----
    for(int f = 0; f < K; f++){
      cplx_t acc{};
      for(int m = 0; m < M; m++)
        acc += std::conj(weights[f][m]) * X[f][m];
      Y[f]         = acc;
      frame_ref[f] = X[f][kRefMic];
      px[f]        = std::norm(frame_ref[f]);    // |x_ref|^2, para el tracker
    }

    // ---- 3. las dos mascaras, cada red con SU estado ------------------------
    m_out = dtln_out_.process_mask( Y );         // sobre la SALIDA del beamformer
    m_ref = dtln_ref_.process_mask( frame_ref ); // sobre el canal crudo

    // ---- 4. la agenda: que mascara describe mejor a Y en este iSIR ----------
    isir_db_ = isir_.update( m_ref, px );
    c_pf_    = 1.0f / (1.0f + std::exp( -(isir_db_ - kIsirCenter) / kIsirWidth ));

    for (int f = 0; f < K; f++)
      m_pf[f] = (1.0f - c_pf_) * m_out[f] + c_pf_ * m_ref[f];

    // ---- 5. POST-FILTRO sobre la salida -------------------------------------
    for (int f = 0; f < K; f++)
      Y[f] *= kSmooth + (1.0f - kSmooth) * m_pf[f];

    // ---- 6. FUERA DEL CAMINO CRITICO: los pesos del frame que viene ---------
    // El SCM come el PROMEDIO de las dos mascaras: necesita masa en las DOS
    // ramas, cosa que el post-filtro -- que solo escala la salida -- no.
    for (int f = 0; f < K; f++){
      m_scm[f] = 0.5f * ( m_out[f] + m_ref[f] );
      // ^kSharpen por elevaciones al cuadrado: 3 multiplicaciones en vez de un
      // std::pow, que son ~50 ciclos cada uno x 514 llamadas por frame.
      const real_t s2 = m_scm[f] * m_scm[f];
      const real_t n  = 1.0f - m_scm[f];
      const real_t n2 = n * n;
      m_s[f] = (s2 * s2) * (s2 * s2);
      m_n[f] = (n2 * n2) * (n2 * n2);
    }
    static_assert(kSharpen == 8.0f, "el desarrollo por cuadrados asume exponente 8");

    mvdr_solve_.update( X, m_s, m_n);

    // Solve CICLICO: en el frame t se recalculan los bins f = t mod P, asi
    // cada frame paga K/P solves y cada bin se refresca cada P frames. SIN el
    // `if (t % P == 0)` de ofb.py: con el, el `for` arrancaba siempre en 0 y los
    // bins f != 0 mod P no se recalculaban nunca. OJO: ofb.py con P > 1 hace
    // otra cosa (todos los bins cada P frames); con P = 1 coinciden.
    for (int f = t_ % kBlockUpd; f < K; f += kBlockUpd)
      weights[f] = mvdr_solve_.solve_bin(f);
    t_ = (t_ + 1) % kBlockUpd;   // solo importa t mod P: asi no desborda

    // ---- 7. IFFT de la salida ya post-filtrada ------------------------------
    //se copia a un buffer del tipo de kissfft en vez de castear.
    for (int k = 0; k < k_bins; ++k)
      bins[k] = { Y[k].real(), Y[k].imag() };

    kiss_fftri(ifftr_cfg_, bins.data(), output_frame.data());

    // kiss_fftri NO normaliza; np.fft.irfft SI divide por n. Medido: el ida y
    // vuelta da 512.0154x sin esto.
    constexpr real_t inv_n = 1.0f / block_len;
    for (int i = 0; i < block_len; ++i)
      output_frame[i] *= inv_n;

    return output_frame;
  }


  inline Hop_real OfbMvdr::process_hop( const Hop_real_multi& hop_in ){
    // ---- 1. ventana de analisis: corre kHop y el hop nuevo entra al final ---
    // Un corrimiento y no un anillo: kiss_fftr necesita el frame contiguo, y
    // son 24 kB de memmove por hop.
    for (int m = 0; m < M; ++m){
      std::copy(in_buf_[m].begin() + kHop, in_buf_[m].end(), in_buf_[m].begin());
      std::copy(hop_in[m].begin(), hop_in[m].end(), in_buf_[m].end() - kHop);
    }

    const Frame_real frame = procces_frame( in_buf_ );

    // ---- 2. overlap-add con el taper de sintesis ----------------------------
    for (int i = 0; i < kFft; ++i)
      ola_[i] += w_syn_[i] * frame[i];

    // ---- 3. sale el hop mas viejo, que ya esta completo ---------------------
    Hop_real out{};
    std::copy(ola_.begin(), ola_.begin() + kHop, out.begin());
    std::copy(ola_.begin() + kHop, ola_.end(), ola_.begin());
    std::fill(ola_.end() - kHop, ola_.end(), 0.0f);
    return out;
  }

}
