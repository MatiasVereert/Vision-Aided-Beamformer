# ¿Biblioteca o a mano para el `solve`? — medido en la Kria

Fecha: 2026-09-25. Contesta la pregunta abierta sobre la regla del plan
("sin bibliotecas externas salvo LiteRT y una FFT chica") aplicada a UNA pieza
puntual: el `solve` de Souden, K=257 sistemas M×M hermitianos complejos en
float32 por frame.

Fuente del benchmark: `mic-array-platform/embedded/bench/bench_solve_lib.cpp`
(un solo archivo, sin dependencias salvo las que se comparan).
Precedente metodológico: `docs/prompt_kria_solve_forms.md`.

---

## Veredicto

**A mano.** No por la regla del plan, sino por una razón que la regla no dice:
la forma Cholesky de Souden resuelve `LB Z = LA` con el lado derecho
**triangular**, y ni Eigen ni BLAS tienen API para expresar eso. Ese ahorro
—medido, no estimado— es de **×2.26 en esa fase** y ninguna biblioteca lo puede
devolver por bien optimizada que esté.

En la Kria, sobre el lote completo de K=257:

| M | a mano | Eigen | OpenBLAS |
|---|---|---|---|
| 8 | **1.816 ms** | 4.962 ms (×2.73) | 4.532 ms (×2.49) |
| 12 | **4.944 ms** | 11.326 ms (×2.29) | 8.907 ms (×1.80) |

Con `-ffast-math`, que es lo que Eigen más necesita, la brecha se achica pero no
se da vuelta: M=8 ×1.98, M=12 ×1.74.

Las dos hipótesis que había que separar:

- **(a) overhead de gestión de memoria — REFUTADA.** Eigen con tamaño fijo hace
  **cero** asignaciones en el heap. No de oídas: interceptando `operator new`
  global y contando durante el lote. Cero en las cuatro variantes, en las dos
  arquitecturas.
- **(b) algoritmo pensado para matrices grandes — CONFIRMADA**, con tres
  evidencias independientes (abajo).

---

## Cómo se midió

Todas las variantes calculan **la misma forma**, la de `SoudenCore._solve_chol`
(`src/beamforming/mask/ofb.py`):

```
LB = chol(Phi_NN cargada) ;  LA = chol(Phi_XX cargada)
LB Z = LA   ->   lam = ||Z||_F^2
LB y = Phi_XX[:,ref] ;  LB^H w = y ;  w /= lam
```

- `mano (tri)`: Cholesky y sustituciones escritas a mano, **explotando** que el
  lado derecho `LA` es triangular (las filas `i<j` de `Z` son cero y no se
  calculan). Es lo que se portaría.
- `mano (RHS denso)`: idéntica, pero tratando el lado derecho como denso. Existe
  **solo** para comparar manzanas con manzanas contra las bibliotecas, que no
  saben del triángulo.
- `Eigen`: `Eigen::LLT<Matrix<complex<float>,M,M>, Lower>`, tamaño fijo en
  compilación, más `triangularView::solveInPlace`. Se midió además una segunda
  redacción idiomática (sin materializar `LB`, operando sobre las vistas del
  propio `LLT`): queda a menos del 2 % de la primera, así que el resultado no es
  un problema de cómo se escribió el Eigen.
- `LAPACK`: `cpotrf` ×2 + `ctrsm` + `ctrsv` ×2 (float complejo: `c*`, no `z*`),
  contra OpenBLAS y contra el LAPACK de referencia de netlib.

Reglas del banco de pruebas:

- Se recorre siempre el **lote de 257 bins en orden** sobre el arreglo completo
  (578 kB en M=12, 257 kB en M=8). Medir una matriz suelta en un bucle daría un
  número optimista: esa matriz vive en L1 y no es el caso real.
- Todas las variantes leen **el mismo buffer**, column-major, que es el layout
  que Eigen y LAPACK necesitan. Se comprobó que esa elección no le cobra nada al
  código a mano: la misma cuenta con la matriz **por filas** (bucle interno
  contiguo en vez de a paso M) da, sobre el lote completo, 1.787 ms contra
  1.810 ms en M=8 y 4.987 contra 4.931 en M=12 sobre la Kria — un empate dentro
  del 1.3 % —, y en x86 resulta un 4 % *más lenta*. El layout no es la palanca
  acá; para las matrices de 8×8 y 12×12 todo el bloque entra en L1 igual.
- Se excluye del cronómetro la normalización por `Den` y la carga diagonal:
  son idénticas para las cuatro variantes y solo diluirían la comparación.
- Mediana sobre ~175–350 repeticiones del lote, descartando el primer octavo.
- Un solo hilo, pinneado a un core (`taskset`), `OPENBLAS_NUM_THREADS=1`.

**Verificación antes de cronometrar nada**: cada variante se compara contra una
referencia del mismo algoritmo en `double`. Error relativo máximo sobre los 257
bins: entre 3.5e-07 y 5.8e-07 en las cuatro, o sea el epsilon de float32. Un
Cholesky mal escrito es más rápido y da cualquier cosa; sin este chequeo los
tiempos no significan nada.

---

## Kria KV260 — Cortex-A53, `-O3 -mcpu=cortex-a53`, un hilo

Lote de K=257 sistemas, mediana:

### M = 8

| variante | ms | µs/bin | vs mano |
|---|---|---|---|
| **a mano, RHS triangular** | **1.816** | 7.07 | **1.00** |
| a mano, RHS denso | 2.406 | 9.36 | 1.32 |
| OpenBLAS (`cpotrf`+`ctrsm`+`ctrsv`) | 4.532 | 17.63 | 2.49 |
| Eigen `LLT` tamaño fijo | 4.962 | 19.31 | 2.73 |

### M = 12

| variante | ms | µs/bin | vs mano |
|---|---|---|---|
| **a mano, RHS triangular** | **4.944** | 19.24 | **1.00** |
| a mano, RHS denso | 7.021 | 27.32 | 1.42 |
| OpenBLAS | 8.907 | 34.66 | 1.80 |
| Eigen `LLT` tamaño fijo | 11.326 | 44.07 | 2.29 |

### Con `-ffast-math`

Es la bandera que más le sirve a Eigen (le habilita reasociar y vectorizar la
aritmética compleja). No alcanza:

| M | a mano | Eigen | brecha |
|---|---|---|---|
| 8 | 1.641 ms | 3.257 ms | ×1.98 |
| 12 | 4.647 ms | 8.085 ms | ×1.74 |

---

## x86_64 de desarrollo — Ryzen AI 7 350, `-O3 -march=native`

Misma forma del resultado, con la brecha más ancha (el x86 tiene más ancho de
banda de vectorización que desperdiciar en overhead):

| variante | M=8 | vs mano | M=12 | vs mano |
|---|---|---|---|---|
| **a mano, RHS triangular** | **0.0931 ms** | 1.00 | **0.2515 ms** | 1.00 |
| a mano, RHS denso | 0.1375 | 1.48 | 0.3794 | 1.51 |
| OpenBLAS | 0.2091 | 2.25 | 0.4196 | 1.67 |
| Eigen `LLT` | 0.3731 | 4.01 | 0.7457 | 2.96 |
| Eigen idiomático (sin copias) | 0.3655 | 3.92 | 0.7358 | 2.93 |
| LAPACK de netlib (referencia) | 0.4400 | 4.72 | 0.8852 | 3.50 |

Con `-ffast-math` (M=12): a mano 0.2251, Eigen 0.4046 (×1.80), OpenBLAS 0.3997
(×1.78).

---

## (a) vs (b): la evidencia

### (a) memoria/construcción de objetos — refutada

`operator new` global interceptado, contando durante un lote de 257 resoluciones:

```
hand_tri      0 new()   0 bytes
hand_dense    0 new()   0 bytes
eigen         0 new()   0 bytes
eigen_v2      0 new()   0 bytes
lapack        0 new()   0 bytes
```

Eigen con tipos de tamaño fijo cumple lo documentado: todo en la pila, nada en el
heap. La hipótesis (a) queda descartada por medición, no por confianza.

### (b) algoritmo para matrices grandes — confirmada, tres veces

**1. No es la caché.** El mismo lote con los 257 bins apuntando a la MISMA matriz
(todo residente en L1) da los mismos tiempos, en las dos máquinas, dentro del
1 %. Ejemplo (Kria, M=12): lote real 11.326 ms / lote en L1 11.236 ms para Eigen;
4.944 / 4.919 a mano. Si la brecha fuera tráfico de memoria, desaparecería acá.
No desaparece: es trabajo de CPU.

**2. Está concentrada en la Cholesky.** Desglose por fase, con `LB` y `LA`
precalculados una sola vez para que las tres variantes coman lo mismo (Kria):

| fase | M=8 mano | Eigen | OpenBLAS | M=12 mano | Eigen | OpenBLAS |
|---|---|---|---|---|---|---|
| 1 — las dos Cholesky | 0.867 | 3.265 (×3.77) | 1.587 (×1.83) | 2.575 | 7.340 (×2.85) | 3.555 (×1.38) |
| 2 — `LB Z = LA` + `‖Z‖²` | 0.697 | 1.251 (×1.80) | 2.254 (×3.23) | 1.771 | 3.069 (×1.73) | 4.387 (×2.48) |
| 3 — dos triangulares vectoriales | 0.304 | 0.229 (×0.76) | 0.646 (×2.13) | 0.631 | 0.909 (×1.44) | 0.926 (×1.47) |

La fase 1 es donde se pierde casi todo.

**3. El overhead se amortiza recién en M≈64.** Costo de UNA Cholesky compleja
normalizado por el trabajo real (M³/6 multiplicaciones-acumulaciones):

Kria:

| M | a mano (ns/flop) | Eigen (ns/flop) | Eigen/mano |
|---|---|---|---|
| 4 | 20.4 | 102.6 | ×5.02 |
| 6 | 19.9 | 85.8 | ×4.32 |
| **8** | **18.9** | **74.3** | **×3.94** |
| **12** | **16.9** | **49.7** | **×2.95** |
| 16 | 16.0 | 37.4 | ×2.34 |
| 24 | 15.3 | 26.3 | ×1.72 |
| 32 | 14.8 | 24.5 | ×1.66 |
| 48 | 14.7 | 18.5 | ×1.26 |
| 64 | 14.5 | 15.6 | ×1.07 |

x86 (misma forma, cruza antes): a mano de 1.34 a 0.65 ns/flop; Eigen de 8.20 a
0.62, y en M=64 **gana** (×0.95).

El código a mano es **plano**: no tiene costo fijo por llamada, solo trabajo. El
de Eigen **cae monótonamente** un factor 6.6, que es la firma exacta de un
overhead fijo por llamada (y por columna) amortizándose. El cruce está en
M ≈ 64, o sea **cinco veces más arriba** del tamaño de este problema.

**Y se ve en el código de la biblioteca.** `Eigen/src/Cholesky/LLT.h`, 3.4.0: para
`size < 32` usa `llt_inplace<Scalar,Lower>::unblocked`, cuyo bucle interno es

```cpp
Block<MatrixType,Dynamic,1>       A21(mat, k+1, k, rs, 1);
Block<MatrixType,1,Dynamic>       A10(mat, k, 0, 1, k);
Block<MatrixType,Dynamic,Dynamic> A20(mat, k+1, 0, rs, k);
...
A21.noalias() -= A20 * A10.adjoint();
```

Bloques de tamaño **`Dynamic`** sobre una matriz de tamaño fijo. La
especialización por tamaño que da `Matrix<cf,12,12>` se pierde adentro de la
factorización: cada columna paga un despacho de producto matriz-vector genérico
en vez de un bucle con límites constantes y desenrollado. El tipo de afuera es
fijo; el algoritmo de adentro, no. Eso es la hipótesis (b), en la fuente.

---

## El costo de integración de LAPACK en la Kria

Aparte de la velocidad, que ya pierde:

1. **La placa no tiene LAPACK ni BLAS instalados.** `ls /usr/lib/aarch64-linux-gnu`
   no devuelve nada con `lapack` ni `blas`.
2. **El único BLAS que hay vive adentro de un snap**:
   `/snap/kria-dashboard/14/lib/python3.10/site-packages/numpy.libs/libopenblas64_p-r0-17488984.3.23.dev.so`,
   **25.8 MB**. Dos problemas de fondo: (i) la ruta lleva el número de revisión
   del snap —está la 6 y está la 14— así que cambia sola cuando el snap se
   actualiza; (ii) es **ILP64 con los símbolos sufijados `_64_`** y enteros de 64
   bits, o sea que las declaraciones `extern "C"` estándar ni siquiera enlazan.
   Hubo que compilar una variante aparte del benchmark para esto.
3. **Fricción de enlace real**: el primer intento falló, porque `libgfortran`
   también está adentro del snap y hay que pasarle `-Wl,-rpath-link`. Lo mismo
   pasó en la PC.
4. **Lo que costaría desplegarlo de verdad**: `apt install libopenblas0`
   (+ `libgfortran5`) en la imagen de la Kria, o meter ~26 MB de `.so` en el
   tarball de `build_app.sh`/`deploy_kria.sh` con su `rpath`. Contra **0 bytes**
   de la versión a mano o de Eigen.

Y con todo eso encima, sigue siendo ×1.80 más lento en M=12 sobre el ARM.

**Eigen, en cambio, no tiene costo de despliegue**: es header-only, y ya está en
la Kria (`/usr/include/eigen3`, 3.4.0, la misma versión que la PC). Lo único que
cuesta es tiempo de compilación: el mismo `.cpp` tarda **13.9 s sin Eigen y
39.7 s con Eigen** en la placa (×2.9).

---

## Donde Eigen gana, y qué dice eso

No todo el resultado es a favor del código a mano, y conviene decirlo:

- **Fase 3 (los dos triangulares con lado derecho vector), M=8 en la Kria**:
  Eigen **×0.76**, y con `-ffast-math` **×0.45**. Su solve triangular de tamaño
  fijo vectoriza con NEON mejor que el bucle escalar a mano.
- **Fase 2**: Eigen (1.251 ms) le gana a `mano denso` (1.290 ms). El solve
  triangular denso de Eigen está bien hecho. El código a mano gana esa fase
  **solo** porque explota el triángulo (0.697 ms), que es una ventaja
  algorítmica, no de redacción.

Conclusión honesta de esto: la brecha **no** es "Eigen está mal escrito". Es (i)
una estructura del problema que la API de la biblioteca no puede expresar, y (ii)
un overhead fijo por llamada que a M=8–12 no se amortiza. Y el punto de la fase 3
dice que el código a mano todavía deja NEON sobre la mesa en las sustituciones.

---

## Recomendación

**Escribir el Cholesky y las sustituciones a mano.** Argumento, en orden de peso:

1. **Es lo más rápido en el target**, por ×1.8 (M=12) a ×2.7 (M=8) contra Eigen a
   `-O3`, y por ×1.7–2.0 aun con `-ffast-math`. Contra OpenBLAS, ×1.8–2.5.
2. **El margen es algorítmico, no de artesanía.** `tr(Phi_NN⁻¹Phi_XX) =
   ‖LB⁻¹LA‖_F²` con `LA` triangular; `trsm` y `solveInPlace` toman el lado
   derecho como denso y no hay forma de decirles lo contrario. `mano denso` vs
   `mano tri` mide exactamente eso: **×2.26** en la fase 2 a M=12. Ninguna
   biblioteca devuelve ese factor.
3. **Son ~40 líneas**, ya verificadas a 5e-07 contra `double` sobre los 257 bins.
   No es un proyecto: es una tarde.
4. **LAPACK además agrega una dependencia de despliegue que la Kria hoy no
   satisface**, para terminar siendo más lento.

Dos matices que van con la recomendación:

- **La regla del plan es correcta acá, pero por otra razón que la que enuncia.**
  No es "las bibliotecas externas son malas": Eigen no cuesta nada desplegar, ya
  está en la placa, y a ×1.7–2.0 con `-ffast-math` seguiría entrando en el
  presupuesto. Lo que decide es la estructura triangular del lado derecho. Vale
  la pena que el plan lo diga así, porque la misma regla aplicada a ciegas a otra
  pieza (por ejemplo la FFT) daría la respuesta equivocada.
- **Si escribir el Cholesky a mano fuera un riesgo** —el plan está escrito para
  alguien aprendiendo C++—, Eigen es un repliegue perfectamente defendible: entra
  en el presupuesto, no cuesta despliegue, y es mucho más difícil de romper.
  Cuesta el ×1.7–2.7. Es una decisión de riesgo, no de rendimiento.

### Lo que esto NO cambia del plan

La forma Cholesky y `float` siguen igual. Lo único que se puede afinar del
enunciado actual: donde dice *"`solve` en forma Cholesky — 1.55× medido en el ARM
(y hay otro ~1.35× si se aprovechan los ceros)"*, ese segundo factor ahora está
medido de punta a punta y es más grande de lo estimado: **×1.32 (M=8) y ×1.42
(M=12)** sobre el `solve` completo, y ×1.85–2.26 sobre la fase que toca.
