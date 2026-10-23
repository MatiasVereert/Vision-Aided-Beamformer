// Veredicto: OfbMvdr contra la referencia de ofb.py, frame a frame.
// Genera la referencia con tests/cpp_parity/dump_ofb_reference.py
#include <ofb/ofb.hpp>
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
  if (!leer(dir + "/ofb_blocks.bin", blocks.data(), blocks.size() * 4) ||
      !leer(dir + "/ofb_out.bin",    ref.data(),    ref.size()    * 4) ||
      !leer(dir + "/ofb_isir.bin",   isir.data(),   isir.size()   * 4)) return 2;

  OfbMvdr bf(model.c_str());

  double peor_abs = 0.0, peor_rel = 0.0, peor_isir = 0.0;
  int t_peor = -1;
  double esc = 0.0, s_dd=0, s_rr=0, s_cc=0, s_cr=0;
  std::vector<double> rms_frame(T, 0.0);
  for (float v : ref) esc = std::fmax(esc, std::fabs(v));

  for (int t = 0; t < T; ++t){
    Frame_real_multi in{};
    for (int m = 0; m < M; ++m)
      for (int i = 0; i < block_len; ++i)
        in[m][i] = blocks[(static_cast<size_t>(t)*M + m)*block_len + i];

    Frame_real out = bf.procces_frame(in);

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
  return ok ? 0 : 1;
}
