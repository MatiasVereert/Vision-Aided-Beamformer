// Veredicto: OfbMvdr contra la referencia de ofb.py, frame a frame, y
// process_hop (buffer de analisis + OLA) contra ola_taper, en el tiempo.
// Genera la referencia con tests/cpp_parity/dump_ofb_reference.py
#include <ofb/ofb.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

static bool leer(const std::string& p, void* dst, size_t bytes){
  std::ifstream f(p, std::ios::binary);
  if (!f) { std::printf("no pude abrir %s\n", p.c_str()); return false; }
  f.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  return f.gcount() == static_cast<std::streamsize>(bytes);
}

int main(int argc, char** argv){
  if (argc != 3){ std::printf("uso: test_ofb <modelo.tflite> <dir_datos>\n"); return 2; }
  const std::string model = argv[1], dir = argv[2];
  constexpr int T = 200;
  using namespace ofb;

  std::vector<float> blocks(static_cast<size_t>(T) * M * block_len);
  std::vector<float> ref   (static_cast<size_t>(T) * block_len);
  std::vector<float> isir  (T);
  std::vector<float> ola   (static_cast<size_t>(T) * hop_len);
  if (!leer(dir + "/ofb_blocks.bin", blocks.data(), blocks.size() * 4) ||
      !leer(dir + "/ofb_out.bin",    ref.data(),    ref.size()    * 4) ||
      !leer(dir + "/ofb_isir.bin",   isir.data(),   isir.size()   * 4) ||
      !leer(dir + "/ofb_ola.bin",    ola.data(),    ola.size()    * 4)) return 2;

  OfbMvdr bf(model.c_str());

  double peor_abs = 0.0, peor_rel = 0.0, peor_isir = 0.0;
  int t_peor = -1;
  double esc = 0.0, s_dd=0, s_rr=0, s_cc=0, s_cr=0;
  std::vector<double> rms_frame(T, 0.0);
  std::vector<float> frames_cpp(static_cast<size_t>(T) * block_len);   // para el OLA
  for (float v : ref) esc = std::fmax(esc, std::fabs(v));

  for (int t = 0; t < T; ++t){
    Frame_real_multi in{};
    for (int m = 0; m < M; ++m)
      for (int i = 0; i < block_len; ++i)
        in[m][i] = blocks[(static_cast<size_t>(t)*M + m)*block_len + i];

    Frame_real out = bf.procces_frame(in);
    std::copy(out.begin(), out.end(), frames_cpp.begin() + static_cast<long>(t) * block_len);

    double e = 0.0;
    for (int i = 0; i < block_len; ++i)
      e = std::fmax(e, std::fabs(out[i] - ref[static_cast<size_t>(t)*block_len + i]));
    if (e > peor_abs){ peor_abs = e; t_peor = t; }
    for (int i = 0; i < block_len; ++i){
      const double r = ref[static_cast<size_t>(t)*block_len + i];
      const double d = out[i] - r;
      s_dd += d*d; s_rr += r*r; s_cc += out[i]*out[i]; s_cr += out[i]*r;
      rms_frame[t] += r*r;
    }
    peor_isir = std::fmax(peor_isir, std::fabs(bf.isir_db() - isir[t]));

    if (t < 3 || t == T/2 || t == T-1)
      std::printf("  t=%3d  |dif|max=%.3e   isir cpp=%+7.3f  py=%+7.3f\n",
                  t, e, bf.isir_db(), isir[t]);
  }
  peor_rel = peor_abs / esc;

  std::printf("\n  pico de la referencia      : %.5f\n", esc);
  std::printf("  PEOR |dif| (t=%d)          : %.3e  -> %.3e relativo al pico\n",
              t_peor, peor_abs, peor_rel);
  std::printf("  PEOR |dif| de isir_db      : %.3e dB\n", peor_isir);

  std::printf("  RMS(dif)/RMS(ref)          : %.3e\n", std::sqrt(s_dd/s_rr));
  std::printf("  correlacion cpp vs py      : %.9f\n", s_cr/std::sqrt(s_cc*s_rr));
  std::printf("  energia del frame peor     : %.3e  (media de los frames: %.3e)\n",
              std::sqrt(rms_frame[t_peor]/block_len),
              std::sqrt(s_rr/(double(T)*block_len)));

  // GUARDA DE REGRESION, no prueba de equivalencia. El lazo esta CERRADO: una
  // diferencia de 1e-6 en Y cambia la mascara del DTLN, que se eleva a la 8 y
  // vuelve por el SCM. El plan fija el criterio real de la Fase 3 en PESQ +-0.01.
  // El iSIR si es exigente: valida FFT + las dos mascaras + el tracker, que no
  // dependen de los pesos.
  const double rms = std::sqrt(s_dd/s_rr), corr = s_cr/std::sqrt(s_cc*s_rr);
  const bool ok = (peor_isir < 1e-2) && (corr > 0.999) && (rms < 2e-2);
  std::printf("\n  %s  (iSIR < 1e-2 dB, correlacion > 0.999, RMS < 2e-2)\n",
              ok ? "PASA" : "FALLA");

  // ======================= process_hop (streaming) =========================
  // Instancia NUEVA: el estado del lazo arranca de cero igual que el de `bf`.
  // Se le dan solo los hops: las ultimas hop_len muestras de cada bloque, que
  // es lo que el dump metio al buffer en ese frame.
  std::printf("\n  --- process_hop ---\n");
  OfbMvdr bh(model.c_str());
  std::vector<float> y(static_cast<size_t>(T) * hop_len);
  for (int t = 0; t < T; ++t){
    Hop_real_multi h{};
    for (int m = 0; m < M; ++m)
      for (int i = 0; i < hop_len; ++i)
        h[m][i] = blocks[(static_cast<size_t>(t)*M + m)*block_len + (block_len - hop_len) + i];
    const Hop_real o = bh.process_hop(h);
    std::copy(o.begin(), o.end(), y.begin() + static_cast<long>(t) * hop_len);
  }

  // (a) EXACTO, aisla el streaming del lazo: el OLA de los frames de `bf`,
  // armado aca con su propia ventana (Hann periodica; con kFft/kHop = 4 su
  // suma OLA es 2). Mismas entradas -> mismos frames bit a bit, asi que lo
  // unico que puede diferir es el buffer de analisis, la ventana o el OLA.
  // Un Hann simetrico en vez de periodico ya da ~1e-3: el criterio (b) no lo ve.
  std::vector<double> acc(static_cast<size_t>(T) * hop_len + block_len, 0.0);
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < block_len; ++i){
      const double w = (0.5 - 0.5*std::cos(2.0*3.14159265358979323846*i/block_len)) / 2.0;
      acc[static_cast<size_t>(t)*hop_len + i] += w * frames_cpp[static_cast<size_t>(t)*block_len + i];
    }
  double peor_self = 0.0, pico_y = 0.0;
  for (size_t n = 0; n < y.size(); ++n){
    peor_self = std::fmax(peor_self, std::fabs(y[n] - acc[n]));
    pico_y    = std::fmax(pico_y, std::fabs(acc[n]));
  }
  std::printf("  vs OLA de procces_frame    : PEOR |dif| %.3e  (%.3e relativo al pico)\n",
              peor_self, peor_self / pico_y);

  // (b) contra ola_taper de ofb.py: el mismo criterio de regresion que arriba,
  // porque lo que difiere es el lazo, no el OLA.
  double o_dd=0, o_rr=0, o_cc=0, o_cr=0;
  for (size_t n = 0; n < y.size(); ++n){
    const double d = y[n] - ola[n];
    o_dd += d*d; o_rr += double(ola[n])*ola[n]; o_cc += double(y[n])*y[n]; o_cr += double(y[n])*ola[n];
  }
  const double o_rms = std::sqrt(o_dd/o_rr), o_corr = o_cr/std::sqrt(o_cc*o_rr);
  std::printf("  vs ola_taper: RMS(dif)/RMS(ref) %.3e   correlacion %.9f\n", o_rms, o_corr);

  const bool ok_hop = (peor_self / pico_y < 1e-5) && (o_corr > 0.999) && (o_rms < 2e-2);
  std::printf("\n  %s  (vs OLA propio < 1e-5 rel, vs ola_taper: correlacion > 0.999, RMS < 2e-2)\n",
              ok_hop ? "PASA" : "FALLA");
  return (ok && ok_hop) ? 0 : 1;
}
