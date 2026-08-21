# Auditoría técnica: port a punto fijo de BLOCK-ONLINE WPE (dereverberación multicanal) para FPGA

## Tu rol
Actuá como **ingeniero senior de DSP/FPGA + álgebra lineal numérica**. Tu tarea es
**auditar** una implementación de *block-online WPE* (dereverberación) que estamos
pasando a **punto fijo**, y **sugerir mejoras concretas y priorizadas**. No des por
sentada ninguna solución (en particular, ya probamos QRD y NO funcionó — ver §4E).
Desafiá nuestras premisas.

Te adjunto: (a) el **paper de NARA-WPE** (Drude et al.), (b) los archivos de código
`nara_wrappers.py` (float) y `nara_wrappers_fixed.py` (punto fijo). Usalos como
referencia. El resumen de abajo es autocontenido para entender el problema antes de
leer el código.

---

## 1. Objetivo

Correr un **front-end de dereverberación WPE multicanal en tiempo real** como IP en
**HLS (C++→RTL)** sobre una **Xilinx KV260 (Zynq UltraScale+ ZU5EV)**, idealmente
usando **SOLO memoria on-chip** (sin DDR). Es el pre-procesador de un beamformer
asistido por visión. Estudiamos el caso de **fuentes fijas**.

**Recursos KV260:** ~23 Mbit on-chip (5.1 Mbit BRAM + 18.4 Mbit URAM), ~1248 DSP,
~117k LUT. DDR disponible pero el objetivo es minimizarla/evitarla.

**Por qué block-online y no el RLS online de nara:** el RLS guarda el estado
recursivo `P = R⁻¹` (KM×KM por bin) → memoria O(F·KM²), que a taps=10 (KM=80) son
~27–79 Mbit → NO entra on-chip, y su recursión es frágil en punto fijo (nos colapsó
a ≤20 bits). El **block** re-estima el filtro resolviendo las ecuaciones normales
por bloque (R es *scratch*, no se guarda) → memoria O(F·KM·M + buffer), y al no
tener recursión es más estable. Queremos validar/mejorar su versión punto fijo.

---

## 2. El algoritmo (block-online WPE)

WPE elimina la **reverberación tardía** prediciéndola desde frames pasados
retardados del observado multicanal (predicción lineal en el dominio STFT, por bin
de frecuencia). Notación por bin `f`:

- `y_t ∈ ℂ^M`  : frame actual observado (M canales).
- `ỹ_t ∈ ℂ^{KM}` : regresor apilado = `[y_{t-Δ}; y_{t-Δ-1}; …; y_{t-Δ-K+1}]`
  (K=taps frames pasados desde el delay Δ, apilando los M canales) → dimensión KM.
- `λ_t` : potencia estimada de la señal dereverberada (peso `1/λ_t`).

**Ecuaciones normales por bloque** (ventana trailing de L frames pasados):
```
R = Σ_{t∈ventana} (1/λ_t) · ỹ_t ỹ_tᴴ        (KM×KM, Hermitiana ≥0)
P = Σ_{t∈ventana} (1/λ_t) · ỹ_t y_tᴴ        (KM×M)
resolver  R·G = P                            (G: KM×M, el filtro de predicción)
aplicar   X_t = y_t − Gᴴ ỹ_t                 (causal; X_t = frame dereverberado)
```
- Se **re-estima G cada `block_shift` frames** sobre la ventana de L; se aplica
  **congelado** hasta el próximo re-solve. Latencia de salida ~1 frame.
- **iterations** (típ. 3): en cada re-solve se itera (re-estimar `λ_t` desde el X
  dereverberado actual → re-armar R,P → resolver), imitando el WPE offline.
- Contraste con el RLS online de nara: éste mantiene `P=R⁻¹` y lo actualiza
  recursivamente por frame (lema de inversión matricial + ganancia de Kalman).

**Parámetros:** M=8, taps K=5–10, delay Δ=2, F=257 bins (STFT tamaño 512, shift
128, fs=16 kHz → frame=8 ms), L=256–512 frames, block_shift≈32, iterations=3.
KM = K·M = 40–80.

---

## 3. Implementación (qué revisar en el código)

**Float** (`nara_wrappers.py`), construido sobre primitivas de nara_wpe:
- `process_wpe_block_online(...)` : STFT → loop de bloques → iSTFT.
- `_block_filters(...)` / `_estimate_block_filter(...)` : el scheduling y la
  estimación de G por bloque. Núcleo de cada iteración:
  ```python
  inverse_power = get_power_inverse(X)                 # 1/λ  (F, Tw)
  R, P = get_correlations_v6(Y_win, Y_tilde_win, inverse_power)   # R:(F,KM,KM) P:(F,KM,M)
  G = _block_cholesky_solve(R, P, reg=reg)             # resolver R G = P
  X = perform_filter_operation_v5(Y=Y_win, Y_tilde=Y_tilde_win, filter_matrix=G)
  ```
- `_block_cholesky_solve(R, P, reg)` : Hermitianiza R, le suma **carga diagonal
  relativa** `reg·mean(diag(R))·I`, y resuelve (LAPACK; en HW sería Cholesky +
  sustitución). `reg` es el knob de regularización.

**Punto fijo** (`nara_wrappers_fixed.py`), reusando las estructuras del RLS fijo:
- `FixedPointConfig` / `Fx(bits, frac)` / `FxStats` : formatos de punto fijo por
  señal (`.q(x, rounding, saturate)`), diagnósticos (overflow, max|G|, diverged).
- `process_wpe_block_online_fixed(...)` + `_estimate_block_filter_fixed(...)`.
  **Qué se cuantiza (lo que un FPGA ALMACENA on-chip):**
  - `f("in")`  → **buffer de ventana** (STFT observado, L frames) — el que domina memoria.
  - `f("g")`   → **filtro G** (se aplica cada frame).
  - `f("pred")`→ salida X.
  - `R, P` → **NO se cuantizan** con formato fijo: son *scratch* y tienen rango
    dinámico enorme (por `1/λ`, >7 décadas) → block-float, igual que nom/pow en el
    RLS fijo. El solve/Cholesky se hace en float (modelamos la precisión de
    **almacenamiento**, que es la que muerde memoria, no el datapath del sqrt).
  - `float_ref` (precisión infinita) reproduce el float EXACTO (rel=0) → la
    emulación es fiel.

---

## 4. Hallazgos empíricos (el corazón del problema)

**(A) En float, el block > online (RLS) en PESQ**, y la ventaja crece con L
(ventana). L es el lever dominante de calidad; block_shift casi no importa (fuentes
fijas); L=512 > L=256.

**(B) En punto fijo, G EXPLOTA salvo con carga diagonal MUY alta.**
- Con `reg=1e-6` (perfecto en float): `max|G|` llega a **22–90** en ciertos bins →
  overflow → la salida se rompe (error relativo ~1.5 vs float).
- Con `reg=1e-2`: G acotada (≤1.3), overflow=0, **16-bit trackea el float** (rel≈0.11).

**(C) La raíz medida — R está severamente mal-condicionada por COHERENCIA
MULTICANAL** (escena real MIRD, M=8, taps=10, L=256):

| freq | cond(R) | max\|G\| | coherencia media entre mics |
|---|---|---|---|
| 60 Hz | 5·10⁵ | 3.4 | 0.78 |
| 250 Hz | 1.8·10⁵ | 0.6 | **0.99** |
| 2 kHz | 1.1·10⁶ | 3.3 | 0.90 |
| 6 kHz | 1.6·10⁶ | 8.0 | 0.97 |

  - `cond(R) ≈ 10⁵–10⁶` en TODOS los bins. `cond(R)` ↔ `max|G|` correlacionan 0.82.
  - Los M=8 mics (array compacto) están casi perfectamente correlacionados
    (0.78–0.99) → el regresor apilado ỹ es **colineal** → R casi-singular.
  - Como `R = AᴴA` (ecuaciones normales), el condicionamiento se **eleva al
    cuadrado**: `cond(R)=cond(A)²` → `cond(A)≈10³`. Float64 (52 bits) lo absorbe;
    fixed 16b (~15) no → la cancelación `X=y−Gᴴỹ` se rompe en los bins malos.
  - Esto es **inherente al WPE multicanal** con array coherente (el cov-RLS sufre
    lo mismo).

**(D) `reg=1e-2` funciona pero CUESTA CALIDAD.** El margen block−online (PESQ,
beamformer oracle mask/SCM) cae fuerte:

| config | rt60=0.16 | rt60=0.61 |
|---|---|---|
| float reg=1e-6 | +0.49 | +0.36 |
| float reg=1e-2 | +0.37 | **+0.12** |
| fixed 16b (reg1e-2, buffer 16b) | +0.32 | +0.12 |
| fixed 16b buffer 12b | +0.16 | +0.05 |
| fixed 16b buffer 10b | **−0.49** | **−0.37** |

  - La cuantización a 16b en sí es limpia (~0.04 de costo). **El costo dominante es
    `reg=1e-2`** (la regularización se come la dereverberación), sobre todo a rt60
    alto (610 ms, ya exigente para taps=10).
  - El **buffer** toca fondo en **12 bits**; 10b/8b colapsan. (El promedio ~√L del
    ruido de cuantización alivia R, pero NO la aplicación: ỹ_t entra crudo en
    `X=y−Gᴴỹ`.) Y 12b×L512 = 25 Mbit → sigue sin entrar on-chip.

**(E) QRD / square-root YA SE PROBÓ y NO alcanzó.** Implementamos un QRD-RLS
(square-root, trabaja sobre A en vez de R → la mitad de bits, estable a 16b). Fue
**más estable pero dio MALA dereverberación** (peor calidad que el cov). Así que
"usá QRD" NO es una respuesta aceptable por sí sola: o el problema no es solo el
condicionamiento del solve, o el QRD introdujo otro sesgo. Hay que entender por qué.

---

## 5. Lo que te pedimos auditar / responder

El dilema central: **queremos BUENA dereverberación Y estabilidad en punto fijo
(16-bit, on-chip) a la vez**, y hoy `reg` alto da estabilidad a costa de calidad,
mientras el condicionamiento de R (por coherencia multicanal) es la raíz.

Preguntas concretas (priorizá por impacto/esfuerzo):

1. **Condicionamiento / regularización:** ¿Hay estrategias mejores que la carga
   diagonal uniforme? (Tikhonov con λ adaptativo POR BIN según cond(R); loading
   proporcional al ruido; reduced-rank / truncated-SVD aprovechando que el rango
   efectivo < KM por la colinealidad; sub-banda; reducir taps a alta freq…). ¿Cuál
   preservaría más dereverberación que `reg=1e-2` uniforme?

2. **Por qué QRD dio peor calidad** siendo numéricamente superior. ¿Es esperable
   que un square-root/QR degrade la dereverberación, o sugiere un bug/sesgo en
   nuestra implementación (ej. cómo aplicamos la regularización, el manejo de `1/λ`,
   el orden de las iteraciones, el reuso del filtro entre bloques)? ¿Cómo hacer
   square-root SIN perder calidad?

3. **La coherencia multicanal como raíz:** ¿Tiene sentido decorrelacionar los
   canales antes del solve (blanqueo espacial / PCA / GSC-like) para bajar cond(A)
   sin sesgar el filtro? ¿O reformular el WPE para arrays compactos coherentes?

4. **Precisión / datapath:** ¿Está bien nuestra decisión de dejar R/P y `1/λ` en
   block-float y cuantizar solo in/g/pred? ¿Escalados o block-floating-point que nos
   dejen bajar bits sin romper la cancelación? ¿La sensibilidad de la aplicación
   `X=y−Gᴴỹ` al buffer se puede mitigar (ej. guardar el residuo en vez del
   observado; predecir en mayor precisión)?

5. **Memoria on-chip:** dado que el buffer no baja de ~12 bits y L grande es lo que
   da calidad, ¿hay forma de tener el efecto de L grande con menos almacenamiento?
   (ventana con decaimiento/taper sin R persistente; estadística incremental;
   sub-muestreo temporal del regresor; compresión del buffer…).

6. **Formulación algorítmica:** ¿Existe una variante de WPE (o de su solve) mejor
   condicionada por construcción para este caso (M grande, canales coherentes, KM
   grande), manteniendo la ventaja de memoria del block (R scratch)?

**Entregable:** una lista **priorizada** de mejoras concretas al port punto-fijo del
block WPE (calidad + estabilidad + memoria on-chip), con el razonamiento numérico de
cada una, y para las top-2 un boceto de cómo implementarlas en este código. Señalá
también cualquier error o supuesto flojo que veas en nuestro enfoque.
