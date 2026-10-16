// Etapa A: solo el modelo 0, encadenando el estado.
//   dtln_stage_a <model_0.tflite> <mag.bin> <mask_out.bin>
// mag.bin: T x 257 float32 volcados desde Python. Criterio: cero exacto.
#include <dtln/dtln_denoise.hpp>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 4) { std::fprintf(stderr, "uso: %s <m0.tflite> <mag.bin> <mask.bin>\n", argv[0]); return 2; }
  std::FILE* f = std::fopen(argv[2], "rb");
  if (!f) { std::fprintf(stderr, "no pude abrir %s\n", argv[2]); return 1; }
  std::fseek(f, 0, SEEK_END); const long bytes = std::ftell(f); std::fseek(f, 0, SEEK_SET);
  if (bytes % static_cast<long>(dtln::bytes_mag) != 0) {
    std::fprintf(stderr, "%ld bytes no es multiplo de %zu (257 float32)\n", bytes, dtln::bytes_mag);
    return 1;
  }
  const long T = bytes / static_cast<long>(dtln::bytes_mag);
  std::vector<float> mag(static_cast<size_t>(T) * dtln::k_bins);
  if (std::fread(mag.data(), sizeof(float), mag.size(), f) != mag.size()) return 1;
  std::fclose(f);

  std::vector<float> out(mag.size());
  try {
    dtln::DtlnDenoise d(argv[1], argv[1]);   // el modelo 1 no se usa en la etapa A
    dtln::Frame_mag in{};
    for (long t = 0; t < T; ++t) {
      for (int k = 0; k < dtln::k_bins; ++k) in[k] = mag[t * dtln::k_bins + k];
      const dtln::Frame_mag m = d.mask_from_magnitude(in);
      for (int k = 0; k < dtln::k_bins; ++k) out[t * dtln::k_bins + k] = m[k];
    }
  } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }

  std::FILE* g = std::fopen(argv[3], "wb");
  std::fwrite(out.data(), sizeof(float), out.size(), g);
  std::fclose(g);
  std::fprintf(stderr, "ok: %ld frames\n", T);
  return 0;
}
