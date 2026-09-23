# Plan de desarrollo: el OFB-MVDR en C++ para ARM

Estado a 2026-09-23. Este documento se edita a medida que avanza el port.

---

## 1. Lo que ya está decidido (y no se vuelve a discutir)

La fase de caracterización terminó. Todo lo de abajo está medido, no supuesto:

| decisión | por qué | dónde está el número |
|---|---|---|
| **Sin sustracción de covarianza** | se cae el `eigh`, que era el 77 % del sistema | `psd-projection-is-load-bearing`, 49 celdas MIRD + aro12 |
| **`solve` en forma Cholesky** | 1.55× medido en el ARM (y hay otro ~1.35× si se aprovechan los ceros) | `ofb_solve_variants.py` |
| **`float`, no `double`** | entra en un núcleo con P=1; double necesita P=2 | `arm-kernel-budget` |
| **`min_loading = 1e-5`** | el 1e-9 de Python está debajo del epsilon de float32 | barrido de carga, 8 celdas |
| **`update` con storage completo** | en float + fast-math empata al triangular y es más simple | `arm-kernel-budget` |
| **M fijo en compilación** | allocación estática, desenrollado total, sin aritmética de punteros | — |
| **Punto fijo: NO** | float cumple el presupuesto con margen | `arm-kernel-budget` |

**Lo que esto le hace a tu plan original**: tu fase 2 ("reescribirlo bajando la
precisión") desaparece. No se escribe en `double` para después bajarlo: se
escribe en `float` desde la primera línea, porque ya sabemos que cierra. Una
reescritura menos.

## 2. Cinco principios que valen para todo el port

1. **Nada de allocación dinámica en el camino de audio.** Todo se reserva en el
   constructor. Ni `new`, ni `malloc`, ni `std::vector::push_back`, ni `string`
   dentro de `step()`. Un `malloc` puede tardar microsegundos impredecibles y
   arruina el tiempo real.
2. **Desarrollar en la PC, medir en la Kria.** El código es el mismo; cambian
   las banderas. Depurar en el target es cinco veces más lento y no aporta nada
   hasta la fase de integración.
3. **Cada fase termina con una comparación contra Python.** Si no hay criterio
   de aceptación, no es una fase: es una esperanza.
4. **La comparación es por métricas, no bit a bit.** Dos razones independientes:
   el DTLN es int8 y su cuantización amplifica cualquier redondeo, y
   `-ffast-math` reordena sumas que numpy no reordena. La excepción es el
   núcleo con máscaras inyectadas (fase 2), donde el lazo está abierto y ahí sí
   se puede exigir error relativo.
5. **Primero que funcione, después que sea rápido.** Ya sabés que entra en el
   presupuesto. No optimices contra un problema que todavía no tenés.

## 3. Dónde vive el código

```
Vision-Aided-Beamformer/
  src/                  # SIN CAMBIOS: es la raiz de paquetes de PYTHON
    beamforming/ ...    # (pyproject.toml declara find = {where = ["src"]})
  cpp/                  # el port, al nivel de tests/ y tools/
    CMakeLists.txt
    README.md           # el formato de los .bin (el contrato de la fase 1)
    ofb/                # LA LIBRERIA, entera en headers
      config.hpp        #   M, K, hop, tipos, constantes  <- el UNICO lugar a tocar
      types.hpp         #   complejo, matriz MxM
      souden_core.hpp   #   update() + solve()
      isir_tracker.hpp
      dtln_stream.hpp   #   envoltorio de LiteRT
      stft.hpp          #   analisis rectangular + OLA con taper Hann
      ofb_mvdr.hpp      #   la clase: step(X_frame) -> Y
    tools/
      ofb_offline.cpp   # un main(): lee dumps, procesa, escribe dumps
    tests/
      test_core.cpp     # otro main(): los chequeos del nucleo
  tests/cpp_parity/
    dump_reference.py   # numpy -> .bin
    compare_cpp.py      # corre el binario y compara
```

**Por que `cpp/` en la raiz y no en `src/`**: `src/` en este repo no es "el
codigo fuente" en general, es la RAIZ DE PAQUETES DE PYTHON, declarada asi en
`pyproject.toml`. El C++ no es una extension de Python (no se importa, no viaja
con el `pip install`, tiene su propio build para otra arquitectura), asi que es
un artefacto del nivel superior, como `tests/` o `tools/`.

**Por que no hay separacion `include/` + `src/`**: es la convencion de C++ de
partir declaraciones (`.hpp`) de definiciones (`.cpp`), y en un proyecto
numerico de ocho archivos no aporta nada. Aca la libreria entera vive en
headers y los UNICOS `.cpp` son los que tienen `main()`. Eso elimina el error
mas frustrante del principiante --el `undefined reference` al enlazar-- y
permite que el compilador inlinee todo sin LTO. La unica regla: una funcion
libre definida en un header lleva `inline` adelante; los metodos definidos
adentro de una clase ya lo son implicitamente.

### Los TRES `main()`, y por que ninguno es "el final"

| main | donde vive | que hace | cuando |
|---|---|---|---|
| `ofb_offline.cpp` | `cpp/tools/` | lee `.bin`/`.wav`, procesa, escribe salida | fases 1-5 |
| `test_core.cpp` | `cpp/tests/` | chequeos con veredicto (exit 0 / 1) | fase 2 en adelante |
| el de tiempo real | **`mic-array-platform/embedded/`** | anillo o socket, hilos, senales; llama a `step()` | fase 6 |

Regla para separar `tools/` de `tests/`: **`tools/` produce algo que mirás;
`tests/` produce un veredicto.**

El de tiempo real NO vive en este repo porque es codigo de PLATAFORMA: habla de
`mmap`, del driver, de sockets y de hilos, que es el vocabulario de
`pdm_server.cpp` y `pdm_stream.cpp`. Ponerlo aca obligaria al repo de algoritmos
a saber de la Kria.

Y esa es la idea que sostiene toda la estructura: **`cpp/ofb/*.hpp` no tiene
ningun `main` y no hace ninguna entrada/salida.** No sabe si los datos vienen de
un archivo, de un socket o de DMA: recibe un frame y devuelve un frame. Los tres
mains son adaptadores distintos sobre la misma libreria.

```
       cpp/ofb/*.hpp   <- la libreria: step(X_frame) -> Y, y nada mas
              ^
    +---------+---------+
    |         |         |
 offline    test    tiempo real
 (archivos) (sintetico) (DMA/socket)
```

Es la misma separacion que ya existe en Python entre `OutputFeedbackMVDR.step()`
y el driver `output_feedback_run`.

### Como el repo de la plataforma consume la libreria

Al ser header-only no hay `.a` que construir ni ABI que hacer coincidir. En
desarrollo, el CMake de la plataforma apunta al repo vecino:

```cmake
target_include_directories(pdm_beamform PRIVATE ${OFB_INCLUDE_DIR})
# OFB_INCLUDE_DIR = ../Vision-Aided-Beamformer/cpp
```

y para desplegar, `build_app.sh` copia `cpp/ofb/*.hpp` dentro del tarball, como
ya hace con el resto. Sin submodulos, sin duplicar codigo, una sola fuente de
verdad. **No se duplica el codigo entre repos.**

### `config.hpp`, el corazón de la configurabilidad

```cpp
#pragma once
#include <complex>

namespace ofb {

inline constexpr int kMics = 12;              // M  <-- cambiar y recompilar
inline constexpr int kFft  = 512;
inline constexpr int kBins = kFft / 2 + 1;    // 257
inline constexpr int kHop  = 128;
inline constexpr int kRefMic = kMics / 2;
inline constexpr int kFs   = 16000;           // fijo: no varia con el array

using real_t = float;
using cplx_t = std::complex<real_t>;

inline constexpr real_t kAlpha      = 0.99f;
inline constexpr real_t kMinLoading = 1e-5f;  // OJO: 1e-9 desaparece en float
inline constexpr real_t kSharpen    = 8.0f;
inline constexpr int    kBlockUpd   = 1;      // P

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

static_assert(kRefMic >= 0 && kRefMic < kMics, "ref_mic fuera de rango");
static_assert(kBins == kFft / 2 + 1, "kBins tiene que salir de kFft");
static_assert(kIsirBinLo <= kIsirBinHi && kIsirBinHi < kBins,
              "la banda del ISIR tiene que caer dentro de los bins validos");

}  // namespace ofb
```

`static_assert` se evalúa **al compilar**: si ponés un `kRefMic` inválido, el
compilador te lo dice antes de que el programa exista. Es la primera herramienta
de C++ que te conviene incorporar. Si algún día `kFft` o `kFs` cambian, el
`static_assert` de la banda del ISIR es lo que te avisa si la banda se corrió
fuera de rango, en vez de fallar en silencio con un `slice` vacío como haría
el equivalente en Python.

---

## 4. Las fases

Cada una tiene: qué construís, cómo sabés que anduvo, y qué C++ aparece.

### Fase 1 — El contrato de datos (sin una sola cuenta)

**Construís**: el esqueleto de directorios, el `CMakeLists.txt`, y un programa
que lee un archivo binario, lo guarda en memoria y lo vuelve a escribir. Del
lado de Python, `dump_reference.py` que exporta de una corrida real:

```
X.bin        (K, T, M) complejo  -> la STFT de entrada
masks.bin    (K, T, 2)           -> m_out y m_ref CRUDAS de cada frame (antes
                                     de fusionar y de elevar a sharpen_exp)
W_ref.bin    (K, T, M) complejo  -> los pesos con los que Python filtro cada
                                     frame (`proc.w_used`, ver nota 2 abajo)
Y_ref.bin    (K, T)   complejo   -> la salida post-filtrada de Python
```

Tres detalles del contrato que hay que fijar ACA, porque si quedan implicitos
se descubren como "bug del algoritmo" en la Fase 2:

1. **`masks.bin` va CRUDO, no fusionado.** No son `m_s`/`m_n` (las que come
   `SoudenCore.update`), son `m_out` y `m_ref` tal como salen de las dos redes,
   antes de `m_scm = 0.5*(m_out+m_ref)` y de elevar a `sharpen_exp`. Eso es lo
   que permite que la Fase 2 -- que reemplaza las dos llamadas a la red por
   una lectura de este archivo -- corra el `step()` COMPLETO (fusion,
   `ISIRTracker`, agenda, post-filtro, `SoudenCore`) y no solo el nucleo. Ver
   Fase 2.
2. **`W_ref[t]` no es el `solve()` que corre durante el frame `t`.** `step()`
   arranca guardando `w_used = self.weights` (el resultado del `solve()` de
   ANTES) y con eso calcula `Y`; el `solve()` de este mismo `step()` -- si
   `t % P == 0` -- deja listos los pesos para `t+1`, no para `t`. `W_out[t]`
   en `output_feedback_run` es ese `w_used`: el peso usado para producir
   `Y[t]`, no el que salio de procesar `X[t]`. Esto no es un bug -- es la
   estructura causal correcta, y el registro `w_used` es puramente
   diagnostico, no participa del lazo -- pero si el arnes de comparacion en
   C++ hace "aplicar -> actualizar -> resolver" en la misma iteracion y
   compara el `solve()` recien calculado contra `W_ref.bin[t]`, va a
   encontrar un desfasaje de un frame que no es un bug del Cholesky sino un
   indice mal leido. `compare_cpp.py` tiene que comparar `W_ref.bin[t]`
   contra el peso que el C++ tenia ANTES de actualizar en la iteracion `t`.
3. **Precision del dump: `float64`/`complex128`**, la nativa de Python, sin
   tocar `ofb.py`. El C++ castea a `float` al leer. Asi el error de `1e-5` de
   la Fase 2 mide la perdida de precision real del puerto (`float32` +
   Cholesky), no un redondeo que ya venia metido en la referencia.

Fijá el resto del formato en el README: orden de los ejes, parte real e
imaginaria intercaladas o separadas, little-endian.

**Criterio**: el C++ lee `X.bin` y lo reescribe; Python verifica que los bytes
son idénticos. Cero matemática todavía.

**Por qué primero**: es el andamio de todas las fases siguientes. Si el
contrato de datos tiene un error de orden de ejes, lo vas a descubrir como un
"bug del algoritmo" tres fases más tarde.

**C++ que aparece**: unidades de compilación (`.hpp` vs `.cpp`), `#include`,
CMake mínimo, `std::vector`, `std::ifstream` en modo binario, `reinterpret_cast`
para leer bytes crudos, `constexpr`.

---

### Fase 2 — Todo menos la red: núcleo, tracker y post-filtro (lazo ABIERTO)

**Construís**: `SoudenCore`, `ISIRTracker`, y el `step()` COMPLETO de
`OutputFeedbackMVDR` menos las dos llamadas a la red -- esas se reemplazan por
una lectura de `masks.bin`.

Por que el tracker y el post-filtro entran ACA y no en la Fase 3 (donde viven
en el Python original, junto con `DTLNStream`): las dos llamadas a la red son
la UNICA fuente de no-determinismo de todo `step()`. Todo lo demas -- la
fusion `m_scm = 0.5*(m_out+m_ref)`, el `ISIRTracker` (el estado de 4 reales,
la inversion de la calibracion `(g,b)`, el sigmoide de la agenda), la
interpolacion `m_pf`, el post-filtro sobre `Y`, y `SoudenCore` -- es
aritmetica pura. Metido en la Fase 3, un bug de puerto en el tracker queda
escondido detras de la tolerancia floja de PESQ (±0.01) y se confunde con
"ruido de la red int8"; separado aca, se le puede exigir el mismo `1e-5` que
al nucleo.

`SoudenCore` tiene dos métodos:

- `update(X_frame, m_s, m_n)`: el producto externo y las dos acumulaciones.
- `solve()`: normalizar, carga diagonal, `chol(B)`, `chol(A)`, la sustitución
  triangular-triangular, `lam = ‖Z‖²_F`, y el único lado derecho.

**Criterio**: error relativo por debajo de **1e-5** en `float`, en DOS
archivos, no uno:

- `W_ref.bin`, respetando la convencion de indice de la Fase 1 (nota 2):
  comparar contra el peso que el C++ tenia ANTES de actualizar en esa
  iteracion.
- `Y_ref.bin`: con el tracker y el post-filtro adentro de esta fase, la
  salida completa es tan determinista como los pesos, asi que tambien se
  puede exigir numero aca en vez de esperar a la Fase 4.

Del lado de Python la referencia se genera con `OFB_MVDR(solve_mode='chol')`,
que implementa exactamente esta secuencia.

**Esta es la fase más importante y la que más tiempo te va a llevar.** Es el 90 %
del riesgo algorítmico. Si el error no baja de 1e-5 en cualquiera de los dos
archivos, el bug está acá y no tiene sentido seguir.

**Guía para el Cholesky** (es el único algoritmo no trivial):

```
para j = 0..M-1:
    L[j][j] = sqrt( real(A[j][j]) - suma_{k<j} |L[j][k]|² )
    para i = j+1..M-1:
        L[i][j] = ( A[i][j] - suma_{k<j} L[i][k]·conj(L[j][k]) ) / L[j][j]
```

La diagonal de `L` es **real y positiva** (por eso el `sqrt` de un número real).
Es el lugar clásico de errores: conjugar el factor equivocado da una matriz que
parece razonable y produce resultados sutilmente malos.

Y el detalle que el benchmark del ARM sugiere que puede costar 1.35×: al
resolver `LB·Z = LA`, la columna `j` de `LA` tiene ceros arriba de la fila `j`.
**El bucle tiene que arrancar en `j`, no en `0`.**

**C++ que aparece**: `std::complex<float>` (y su costo: puede inhibir la
vectorización por las reglas de aliasing y NaN — si más adelante el perfilado lo
señala, se reemplaza por dos arreglos separados de real e imaginaria, pero
**no empieces por ahí**), clases con constructor y métodos, referencias (`&`),
`const`, `std::array` de tamaño fijo.

---

### Fase 3 — La red y el lazo CERRADO

**Construís**: `DTLNStream` sobre LiteRT, y CERRÁS el lazo: las dos lecturas
de `masks.bin` que la Fase 2 usaba como muleta se reemplazan por las dos
llamadas reales a la red (`dtln_out`, `dtln_ref`). El resto de `step()` --
núcleo, tracker, post-filtro -- ya está construido y validado a `1e-5` desde
la Fase 2 y no se toca.

**Dos cosas que no son obvias y te van a morder**:

1. **Los dos intérpretes tienen que compartir UN `FlatBufferModel`.** Si cada
   uno carga el archivo por su cuenta son 756 kB de pesos en vez de 378, y el
   segundo `invoke` no encuentra nada caliente en caché.
2. **El desfasaje de un frame**: el bloque `i` del DTLN corresponde al frame
   `i-1` de la STFT. Está documentado en `dtln-mask-frame-offset`; en el
   streaming del lazo se resuelve solo, pero si comparás máscaras contra un
   dump, es la primera sospecha cuando no alinean.

**Criterio**: PESQ y STOI de la salida contra las de Python, sobre las mismas
escenas. Diferencia dentro de ±0.01 de PESQ. **No** error muestra a muestra: el
lazo cerrado con una red int8 diverge por construcción.

**C++ que aparece**: enlazar una biblioteca externa (la parte fea de CMake),
`std::unique_ptr` para manejar la vida de los objetos de LiteRT, punteros
crudos en la frontera con la API de C.

---

### Fase 4 — STFT y reconstrucción

**Construís**: el análisis con ventana rectangular y la síntesis por
overlap-add con taper de Hann (`ola_taper` en el Python). Hace falta una FFT
real de 512 puntos: KissFFT o PFFFT son chicas, autocontenidas y se compilan
sin fricción. Evitá FFTW (pesada y con licencia incómoda).

Ojo con el escalado: `scipy.signal.stft` con ventana rectangular divide por
`nperseg`, y el DTLN come la magnitud **sin** normalizar. En el Python eso
aparece como el factor `nperseg *` antes de cada llamada a la red. Es el tipo de
constante que, si te equivocás, hace que la red reciba una señal fuera de su
dominio de entrenamiento y devuelva máscaras sin sentido — sin ningún error
visible.

**Criterio**: cadena completa WAV → WAV en la PC, y las métricas contra la
salida de Python en las mismas escenas MIRD.

**Hito**: acá el sistema **existe**. Todo lo que sigue es rendimiento e
integración.

---

### Fase 5 — Rendimiento en la Kria

**Construís**: nada nuevo. Compilás lo de la fase 4 en el ARM y medís.

- Banderas: `-O3 -mcpu=cortex-a53` más el juego acotado de math
  (`-ffp-contract=fast -fno-math-errno -fno-trapping-math -fassociative-math
  -fno-signed-zeros`) en lugar de `-ffast-math`, para conservar NaN como señal
  de error.
- `-fopt-info-vec` te dice qué bucles vectorizó y cuáles no. El benchmark
  sugiere que en `float` hay un 2× sin explotar: el techo de NEON son 4 lanes y
  estás sacando 1.5×.
- Si no entra en 8 ms: subí `P` (repartiendo **bins** entre frames, no haciendo
  todo de golpe cada P frames) antes de tocar nada más.

**Criterio**: el frame completo por debajo de 8 ms, sostenido, con las métricas
de la fase 4 intactas.

---

### Fase 6 — Tiempo real

**Construís**: el nodo que vive en `mic-array-platform` y consume esta librería.

El regalo de la plataforma: **un período de DMA son 128 frames = 8 ms = exacto
un hop de tu STFT**. Entra un período, sale un hop, sin re-bufferear nada. Si
alineás el procesamiento a esa frontera te ahorrás una cola entera y su latencia.

**Criterio**: soak de varios minutos sin underruns, latencia medida con las
herramientas que ya existen en el otro repo.

---

## 5. Qué C++ aprender, y qué ignorar

C++ es un lenguaje enorme y la mayor parte no te sirve para esto. Este proyecto
es, en el fondo, C con clases y bucles numéricos.

**Lo que sí, en el orden en que lo vas a necesitar:**

| concepto | fase | para qué |
|---|---|---|
| `.hpp` / `.cpp`, `#include`, CMake | 1 | estructurar el proyecto |
| `constexpr`, `static_assert` | 1 | la configuración de compilación |
| `std::vector`, `std::array` | 1–2 | arreglos, con y sin tamaño fijo |
| referencias (`&`) y `const` | 2 | pasar arreglos sin copiarlos |
| clases: constructor + métodos | 2 | el estado del procesador |
| `std::complex<float>` | 2 | los datos |
| `std::unique_ptr` | 3 | la vida de los objetos de LiteRT |
| punteros crudos | 3 | la frontera con APIs de C |

**Lo que NO necesitás y solo te va a confundir**: templates (más allá de leer
`std::vector<float>`), herencia y funciones virtuales, excepciones, la biblioteca
de algoritmos (`std::transform` y compañía), *move semantics*, lambdas,
`std::thread` hasta la fase 6.

Si en algún momento sentís que necesitás una de esas para resolver algo, casi
seguro hay un camino más simple. Preguntame antes de meterla.

## 6. Herramientas que valen su peso en oro cuando estás aprendiendo

- **Sanitizers**: compilá la versión de depuración con
  `-fsanitize=address,undefined -g`. Un acceso fuera de rango te avisa en el
  momento exacto, con archivo y línea, en vez de corromper memoria en silencio y
  reventar diez funciones después. Es *la* herramienta que más te va a ahorrar.
  Cuesta velocidad, así que solo en la build de depuración.
- **`gdb`**: para ver el valor de una variable en el punto de la falla.
- **Imprimir y comparar contra Python**: la técnica más efectiva para bugs
  numéricos. Volcá una matriz, cargala en numpy, restala de la de Python.

## 7. Riesgos conocidos, para que no te sorprendan

| riesgo | cuándo aparece | qué hacer |
|---|---|---|
| Orden de ejes en los `.bin` | fase 1 | por eso la fase 1 existe |
| Conjugar el factor equivocado en el Cholesky | fase 2 | comparar `L·Lᴴ` contra `B` |
| El `sqrt` de un negativo si falta la carga diagonal | fase 2 | `kMinLoading = 1e-5`, y verificar que se aplica |
| Escalado de la magnitud que entra a la red | fase 4 | el factor `nperseg` |
| El desfasaje de 1 frame de la máscara | fase 3 | `dtln-mask-frame-offset` |
| `-ffast-math` tapando un NaN | fase 5 | el juego acotado de banderas |
| Pesos duplicados de los dos DTLN | fase 3 | un solo `FlatBufferModel` |
| `W_ref.bin[t]` corrido un frame contra el `solve()` de esa iteracion | fase 2 | `w_used` se guarda ANTES de actualizar (es diagnostico); comparar contra el peso previo, no el recien resuelto |

## 8. El orden, en una línea

> contrato de datos → núcleo con lazo abierto → red y lazo cerrado → STFT →
> medir en la Kria → tiempo real

Las fases 1 y 2 son el 90 % del riesgo algorítmico y probablemente el 60 % del
tiempo. Las fases 3 a 5 son plomería con un criterio claro en cada punta. La 6
es otro proyecto, en otro repo.
