// ofb_offline: corre OfbMvdr::process_hop sobre un WAV de M canales, como si
// llegara del DMA (un hop por periodo), y mide cuanto tarda cada hop.
//
//   ofb_offline <modelo.tflite> <entrada.wav> [salida.wav]
//
// La pregunta que contesta: con este kBlockUpd, entra en tiempo real? Un hop
// son kHop/kFs = 8 ms de audio y es el presupuesto entero, con el servidor y el
// DMA compitiendo por los mismos nucleos. Compilado con -DOFB_PROFILE (lo hace
// el CMakeLists) ademas desglosa el hop por etapa.
//
// Entrada: WAV PCM de 16/24/32 bits o float32, M canales a kFs -- o sea lo que
// graba pdm_record. Los enteros se llevan a [-1, 1) dividiendo por 2^(bits-1),
// igual que soundfile, que es como lo lee el benchmark de Python.
// Salida: WAV float32 mono, ATRASADA kFft - kHop muestras (24 ms) respecto de
// la entrada -- el retardo del OLA, ver process_hop.
#include <ofb/ofb.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace ofb;

namespace {

struct Wav {
  int channels = 0, rate = 0, bits = 0;
  bool is_float = false;
  std::vector<float> data;       // entrelazado (n, channels), en [-1, 1)
};

uint32_t u32(const unsigned char* p){ return p[0] | p[1] << 8 | p[2] << 16 | uint32_t(p[3]) << 24; }
uint16_t u16(const unsigned char* p){ return uint16_t(p[0] | p[1] << 8); }

// Lo justo para lo que graba la Kria: recorre los chunks, toma "fmt " y "data"
// y saltea el resto (LIST, etc.). WAVE_FORMAT_EXTENSIBLE se resuelve por los
// dos primeros bytes del subformato.
bool read_wav(const std::string& path, Wav& w, std::string& err){
  std::ifstream f(path, std::ios::binary);
  if (!f){ err = "no pude abrir " + path; return false; }
  std::vector<unsigned char> b{std::istreambuf_iterator<char>(f), {}};
  if (b.size() < 12 || std::memcmp(b.data(), "RIFF", 4) || std::memcmp(b.data() + 8, "WAVE", 4)){
    err = "no es un WAV RIFF"; return false;
  }
  int fmt = 0;
  const unsigned char* pcm = nullptr; size_t pcm_bytes = 0;
  for (size_t p = 12; p + 8 <= b.size(); ){
    const uint32_t len = u32(&b[p + 4]);
    const unsigned char* c = &b[p + 8];
    if (p + 8 + len > b.size()){ err = "chunk truncado"; return false; }
    if (!std::memcmp(&b[p], "fmt ", 4)){
      fmt = u16(c); w.channels = u16(c + 2); w.rate = int(u32(c + 4)); w.bits = u16(c + 14);
      if (fmt == 0xFFFE && len >= 26) fmt = u16(c + 24);
    } else if (!std::memcmp(&b[p], "data", 4)){
      pcm = c; pcm_bytes = len;
    }
    p += 8 + len + (len & 1);      // los chunks impares llevan un byte de relleno
  }
  if (!pcm || !w.channels){ err = "falta el chunk fmt o data"; return false; }
  w.is_float = (fmt == 3);
  if (!(fmt == 1 && (w.bits == 16 || w.bits == 24 || w.bits == 32)) && !(w.is_float && w.bits == 32)){
    err = "formato no soportado (fmt=" + std::to_string(fmt) + ", bits=" + std::to_string(w.bits) + ")";
    return false;
  }
  const int bps = w.bits / 8;
  const size_t n = pcm_bytes / bps;
  w.data.resize(n);
  const float esc = 1.0f / float(1u << (w.bits - 1));
  for (size_t i = 0; i < n; ++i){
    const unsigned char* s = pcm + i * bps;
    if (w.is_float)       { float v; std::memcpy(&v, s, 4); w.data[i] = v; }
    else if (w.bits == 16)  w.data[i] = float(int16_t(u16(s))) * esc;
    else if (w.bits == 24)  // los 3 bytes arriba de un int32 y corrimiento aritmetico: extiende el signo
      w.data[i] = float(int32_t(uint32_t(s[0]) << 8 | uint32_t(s[1]) << 16 | uint32_t(s[2]) << 24) >> 8) * esc;
    else                    w.data[i] = float(int32_t(u32(s))) * esc;
  }
  return true;
}

bool write_wav_f32(const std::string& path, const std::vector<float>& y, int rate){
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  auto put32 = [&](uint32_t v){ f.write(reinterpret_cast<const char*>(&v), 4); };
  auto put16 = [&](uint16_t v){ f.write(reinterpret_cast<const char*>(&v), 2); };
  const uint32_t data_bytes = uint32_t(y.size() * 4);
  f.write("RIFF", 4); put32(36 + data_bytes); f.write("WAVE", 4);
  f.write("fmt ", 4); put32(16); put16(3); put16(1); put32(uint32_t(rate));
  put32(uint32_t(rate) * 4); put16(4); put16(32);
  f.write("data", 4); put32(data_bytes);
  f.write(reinterpret_cast<const char*>(y.data()), data_bytes);
  return bool(f);
}

struct Stats { double media, p50, p99, max; };
Stats stats(std::vector<double> v){
  double s = 0; for (double x : v) s += x;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return { s / n, v[n / 2], v[std::min(n - 1, n * 99 / 100)], v.back() };
}

}  // namespace

int main(int argc, char** argv){
  if (argc < 3 || argc > 4){
    std::printf("uso: ofb_offline <modelo.tflite> <entrada.wav> [salida.wav]\n");
    return 2;
  }
  Wav w; std::string err;
  if (!read_wav(argv[2], w, err)){ std::printf("[!!] %s\n", err.c_str()); return 2; }
  if (w.channels != M || w.rate != kFs){
    std::printf("[!!] el WAV tiene %d canales a %d Hz y el OFB esta compilado para M=%d a %d Hz\n",
                w.channels, w.rate, M, kFs);
    return 2;
  }
  const size_t n = w.data.size() / M, H = n / hop_len;
  const double presupuesto = 1000.0 * hop_len / kFs;
  std::printf("[..] %s: %d canales, %d bits%s, %.2f s -> %zu hops de %d muestras\n",
              argv[2], w.channels, w.bits, w.is_float ? " float" : "", double(n) / kFs, H, hop_len);
  std::printf("[..] M=%d  K=%d  kFft=%d  kHop=%d  kBlockUpd(P)=%d  -> presupuesto %.1f ms por hop\n",
              M, K, kFft, kHop, kBlockUpd, presupuesto);

  OfbMvdr bf(argv[1]);

  std::vector<float> y(H * hop_len);
  std::vector<double> total(H);
#ifdef OFB_PROFILE
  std::vector<StageTimes> etapas(H);
#endif
  for (size_t h = 0; h < H; ++h){
    Hop_real_multi in{};
    for (int i = 0; i < hop_len; ++i)
      for (int m = 0; m < M; ++m)
        in[m][i] = w.data[(h * hop_len + i) * M + m];

    const auto t0 = std::chrono::steady_clock::now();
    const Hop_real o = bf.process_hop(in);
    total[h] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
#ifdef OFB_PROFILE
    etapas[h] = bf.profile();
#endif
    std::copy(o.begin(), o.end(), y.begin() + h * hop_len);
  }

  // ---- reporte -------------------------------------------------------------
  // p99 y max por separado: el max lo suele poner el primer hop (XNNPACK arma
  // sus buffers en la primera invocacion) o una interrupcion del sistema; el
  // p99 es lo que dice si el regimen entra.
  std::printf("\n  %-10s %9s %9s %9s %9s %9s\n", "etapa", "media", "p50", "p99", "max", "%presup");
  auto fila = [&](const char* nom, const std::vector<double>& v){
    const Stats s = stats(v);
    std::printf("  %-10s %9.3f %9.3f %9.3f %9.3f %8.1f%%\n",
                nom, s.media, s.p50, s.p99, s.max, 100.0 * s.media / presupuesto);
  };
#ifdef OFB_PROFILE
  const struct { const char* nom; double StageTimes::* campo; } cols[] = {
    {"fft", &StageTimes::fft}, {"bf", &StageTimes::bf},
    {"dtln_out", &StageTimes::dtln_out}, {"dtln_ref", &StageTimes::dtln_ref},
    {"isir_pf", &StageTimes::isir_pf}, {"update", &StageTimes::update},
    {"solve", &StageTimes::solve}, {"ifft", &StageTimes::ifft}, {"ola", &StageTimes::ola},
  };
  for (const auto& c : cols){
    std::vector<double> v(H);
    for (size_t h = 0; h < H; ++h) v[h] = etapas[h].*c.campo;
    fila(c.nom, v);
  }
  std::printf("  %s\n", std::string(62, '-').c_str());
#endif
  fila("HOP", total);

  const size_t tarde = size_t(std::count_if(total.begin(), total.end(),
                                            [&](double t){ return t > presupuesto; }));
  double suma = 0; for (double t : total) suma += t;
  std::printf("\n  primer hop: %.3f ms   hops por encima de %.1f ms: %zu de %zu (%.2f%%)\n",
              total[0], presupuesto, tarde, H, 100.0 * tarde / H);
  std::printf("  factor de tiempo real: %.2fx  (audio %.2f s / computo %.2f s)\n",
              (double(H) * presupuesto) / suma, double(H) * presupuesto / 1000.0, suma / 1000.0);

  if (argc == 4){
    if (!write_wav_f32(argv[3], y, kFs)){ std::printf("[!!] no pude escribir %s\n", argv[3]); return 1; }
    std::printf("  salida: %s (float32 mono, atrasada %d muestras)\n", argv[3], kFft - kHop);
  }
  return 0;
}
