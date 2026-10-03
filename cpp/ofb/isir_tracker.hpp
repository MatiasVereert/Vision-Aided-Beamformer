#pragma once 

#include <ofb/config.hpp>
#include <ofb/types.hpp>
#include <cmath>

namespace ofb{

 class isir_tracker{
  public:
    // `m_ref`: la mascara CRUDA de la red sobre el canal de referencia -- sin
    // elevar a kSharpen y sin fundir con m_out. NO son las dos ramas del SCM.
    // `power`: |X_ref|^2. Cualquier factor de escala se cancela en el cociente.
    real_t update( const Reals& m_ref, const Reals& power);

  private: 

  //variables..

  real_t S{};
  real_t N{};

  real_t M_s{};
  real_t M_n{};

  // `ofb.py` asigna el PRIMER frame directo en vez de suavizarlo desde cero
  // (state = v if state is None). Sin esto el cociente diverge hasta 1.4 dB.
  bool first{true};
 };


 inline real_t isir_tracker::update(const Reals& m_ref, const Reals& power){

  // la banda es INCLUSIVA en los dos extremos, como el `slice` de Python
  constexpr int Kb = kIsirBinHi - kIsirBinLo + 1;

  // scratch state
  real_t acc_s = 0;
  real_t acc_M_s = 0;
  real_t acc_p = 0;

  // Calculate states. La rama de RUIDO no se acumula: sale por identidad,
  //   sum (1-a) p = sum p - sum a p        sum (1-a) = Kb - sum a
  // asi que son UN producto interno y DOS reducciones.
  for(int k = kIsirBinLo; k <= kIsirBinHi; k++ ){
      acc_s   += m_ref[k] * power[k];
      acc_M_s += m_ref[k];
      acc_p   += power[k];
    };

  const real_t acc_n   = acc_p - acc_s;
  const real_t acc_M_n = real_t(Kb) - acc_M_s;

  // Smothing
  if (first){
    S = acc_s;  N = acc_n;  M_s = acc_M_s;  M_n = acc_M_n;
    first = false;
  } else {
    S   = kIsirAlpha * S   + (1 - kIsirAlpha) * acc_s;
    N   = kIsirAlpha * N   + (1 - kIsirAlpha) * acc_n;
    M_s = kIsirAlpha * M_s + (1 - kIsirAlpha) * acc_M_s;
    M_n = kIsirAlpha * M_n + (1 - kIsirAlpha) * acc_M_n;
  }

  // Estimate. Los guardas son los de `ofb.py`: sin ellos, M_s = 0 da 0/0 = NaN
  // y el NaN se propaga al sigmoide y al post-filtro sin nada que lo saque.
  // std::fmax en vez de std::max: ya viene en <cmath> y en ARM es una sola
  // instruccion (fmaxnm), sin rama.
  real_t isir_db = 10.0f * std::log10( (S / std::fmax(M_s, 1e-20f) + 1e-20f)
                                     / (N / std::fmax(M_n, 1e-20f) + 1e-20f) );
  // Calibration: invierte hat = g * iSIR_real + b, asi que la salida esta en dB
  // de iSIR REAL y kIsirCenter/kIsirWidth son magnitudes fisicas.
  isir_db = (isir_db - kIsirCalibB) / kIsirCalibG;
  

    return isir_db;
  }

}
