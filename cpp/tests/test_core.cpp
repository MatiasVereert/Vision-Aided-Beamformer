// Fase 2: SoudenCore con el lazo ABIERTO. Las mascaras vienen de ofb.py, asi que
// la red sale de la ecuacion y lo unico que se compara es el nucleo numerico.
#include <ofb/solve.hpp>
#include <cstdio>
#include <cmath>
#include <complex>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

static bool leer(const std::string& p, void* d, size_t b){
  std::ifstream f(p, std::ios::binary);
  if(!f){ std::printf("no pude abrir %s\n", p.c_str()); return false; }
  f.read(static_cast<char*>(d), static_cast<std::streamsize>(b));
  return f.gcount() == static_cast<std::streamsize>(b);
}

int main(int argc, char** argv){
  if (argc != 2){ std::printf("uso: test_core <dir_datos>\n"); return 2; }
  const std::string dir = argv[1];
  constexpr int T = 200;
  using namespace ofb;

  std::vector<std::complex<float>> X(size_t(T)*K*M), W(size_t(T)*K*M);
  std::vector<float> ms(size_t(T)*K), mn(size_t(T)*K);
  if(!leer(dir+"/core_X.bin",  X.data(),  X.size()*8)  ||
     !leer(dir+"/core_W.bin",  W.data(),  W.size()*8)  ||
     !leer(dir+"/core_ms.bin", ms.data(), ms.size()*4) ||
     !leer(dir+"/core_mn.bin", mn.data(), mn.size()*4)) return 2;

  SoudenCore core;
  double peor = 0.0; int t_peor=-1, k_peor=-1;
  double peor_t1 = 0.0, peor_rank = 0.0; std::vector<double> todos; double banda[4]={0,0,0,0};

  for (int t=0; t<T; ++t){
    Frame Xf{}; Reals s{}, n{};
    for (int k=0;k<K;++k){
      s[k] = ms[size_t(t)*K+k];
      n[k] = mn[size_t(t)*K+k];
      for (int m=0;m<M;++m) Xf[k][m] = X[(size_t(t)*K+k)*M+m];
    }
    core.update(Xf, s, n);

    for (int k=0;k<K;++k){
      Vec w = core.solve_bin(k);
      // escala del bin: |w| de python, para que el error sea relativo
      double esc = 0.0;
      for (int m=0;m<M;++m) esc = std::fmax(esc, std::abs(W[(size_t(t)*K+k)*M+m]));
      for (int m=0;m<M;++m){
        const double d = std::abs(std::complex<double>(w[m]) -
                                  std::complex<double>(W[(size_t(t)*K+k)*M+m]));
        const double rel = d / std::fmax(esc, 1e-12);
        if (t==0) peor_t1 = std::fmax(peor_t1, rel);
        if (t < M) peor_rank = std::fmax(peor_rank, rel);
        else { if (rel > peor){ peor = rel; t_peor = t; k_peor = k; }
               todos.push_back(rel); banda[k*4/K] = std::fmax(banda[k*4/K], rel); }
      }
    }
  }
  std::printf("  t=0        (Phi de RANGO 1)            : %.3e\n", peor_t1);
  std::printf("  t<M=12     (Phi aun deficiente de rango): %.3e\n", peor_rank);
  std::printf("  t>=12      (REGIMEN NORMAL)            : %.3e   (t=%d, bin=%d)\n",
              peor, t_peor, k_peor);
  std::sort(todos.begin(), todos.end());
  std::printf("\n  distribucion en regimen normal (%zu muestras):\n", todos.size());
  std::printf("    mediana   : %.3e\n", todos[todos.size()/2]);
  std::printf("    p99       : %.3e\n", todos[todos.size()*99/100]);
  std::printf("    p99.99    : %.3e\n", todos[todos.size()*9999/10000]);
  std::printf("  peor por banda de frecuencia:\n");
  const char* nom[4] = {"bins   0- 64 (0-2 kHz)","bins  64-128 (2-4 kHz)",
                        "bins 128-192 (4-6 kHz)","bins 192-257 (6-8 kHz)"};
  for (int b=0;b<4;++b) std::printf("    %s : %.3e\n", nom[b], banda[b]);
  // El criterio va sobre la MEDIANA, no sobre el maximo. Un bug de logica
  // (conjugar mal, un indice corrido) desplaza TODOS los bins y por lo tanto la
  // mediana; la cola la fija el numero de condicion -- medido ~1e6 en los bins
  // peores, donde la cota clasica eps*cond ya permite 1e-1 en float32.
  const double mediana = todos[todos.size()/2];
  const double p99     = todos[todos.size()*99/100];
  const bool ok = (mediana < 1e-5) && (p99 < 1e-3);
  std::printf("\n  %s  (criterio: mediana < 1e-5 y p99 < 1e-3)\n", ok?"PASA":"FALLA");
  return ok?0:1;
}
