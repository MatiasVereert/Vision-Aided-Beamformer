// Envoltorio offline del DTLN: .bin float32 mono 16 kHz -> .bin float32.
//
//   dtln_test_a <model_0.tflite> <model_1.tflite> <in.bin> <out.bin>
//
// Replica apply_dtln_post_tflite_realtime() de src/dnn_denoise/dtln_lite.py:
// normalizacion por pico, buffer deslizante de 512 con salto de 128,
// overlap-add, y restauracion de la escala al final.
//
// Lo que corresponde al "envoltorio" vive aca; DtlnDenoise::process_block()
// es 512 -> 512 y no sabe nada del solapamiento.

#include <dtln/dtln_denoise.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// Lee un archivo entero como float32. Devuelve false si no existe o si el
// tamano no es multiplo de 4 (sintoma tipico de haber volcado mal desde Python).
bool read_f32(const char* path, std::vector<float>& out) {
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) return false;
  std::fseek(f, 0, SEEK_END);
  const long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (bytes < 0 || bytes % static_cast<long>(sizeof(float)) != 0) {
    std::fclose(f);
    return false;
  }
  out.resize(static_cast<size_t>(bytes) / sizeof(float));
  const size_t got = std::fread(out.data(), sizeof(float), out.size(), f);
  std::fclose(f);
  return got == out.size();
}

bool write_f32(const char* path, const std::vector<float>& v) {
  std::FILE* f = std::fopen(path, "wb");
  if (f == nullptr) return false;
  const size_t put = std::fwrite(v.data(), sizeof(float), v.size(), f);
  std::fclose(f);
  return put == v.size();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr,
                 "uso: %s <model_0.tflite> <model_1.tflite> <in.bin> <out.bin>\n",
                 argv[0]);
    return 2;
  }

  std::vector<float> audio;
  if (!read_f32(argv[3], audio)) {
    std::fprintf(stderr, "no pude leer %s como float32\n", argv[3]);
    return 1;
  }

  using namespace dtln;

  // --- normalizacion por pico (igual que la referencia) -------------------
  float peak = 0.0f;
  for (float s : audio) peak = std::max(peak, std::fabs(s));
  const float scale = (peak > 1e-6f) ? peak : 1.0f;
  for (float& s : audio) s /= scale;

  // num_blocks = (len(audio) - (block_len - block_shift)) // block_shift
  const long n = static_cast<long>(audio.size());
  const long num_blocks = (n - (block_len - hop_len)) / hop_len;
  if (num_blocks <= 0) {
    std::fprintf(stderr, "audio demasiado corto: %ld muestras\n", n);
    return 1;
  }

  std::vector<float> out(audio.size(), 0.0f);   // np.zeros_like: la cola queda en cero

  try {
    DtlnDenoise dtln(argv[1], argv[2]);

    Frame_real in_buffer{};
    Frame_real out_buffer{};

    for (long idx = 0; idx < num_blocks; ++idx) {
      const long start = idx * hop_len;

      // in_buffer[:-hop] = in_buffer[hop:]   (destino antes que origen: copy es seguro)
      std::copy(in_buffer.begin() + hop_len, in_buffer.end(), in_buffer.begin());
      // in_buffer[-hop:] = audio[start:start+hop]
      std::copy(audio.begin() + start, audio.begin() + start + hop_len,
                in_buffer.end() - hop_len);

      const Frame_real block = dtln.process_block(in_buffer);

      // out_buffer[:-hop] = out_buffer[hop:];  out_buffer[-hop:] = 0
      std::copy(out_buffer.begin() + hop_len, out_buffer.end(), out_buffer.begin());
      std::fill(out_buffer.end() - hop_len, out_buffer.end(), 0.0f);

      // out_buffer += block    (SUMA, no asignacion: esto es el overlap-add)
      for (int i = 0; i < block_len; ++i) out_buffer[i] += block[i];

      // out[start:start+hop] = out_buffer[:hop]
      std::copy(out_buffer.begin(), out_buffer.begin() + hop_len,
                out.begin() + start);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }

  // --- restaurar la escala original ---------------------------------------
  for (float& s : out) s *= scale;

  if (!write_f32(argv[4], out)) {
    std::fprintf(stderr, "no pude escribir %s\n", argv[4]);
    return 1;
  }

  std::fprintf(stderr, "ok: %ld muestras, %ld bloques, pico de entrada %.6g\n",
               n, num_blocks, static_cast<double>(peak));
  return 0;
}
