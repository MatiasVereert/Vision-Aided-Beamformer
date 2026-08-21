# Auditoría técnica: solve QRD / raíz-cuadrada para WPE en punto fijo (FPGA)

## Tu rol
Sos ingeniero senior de FPGA/DSP y aritmética de punto fijo. Tu trabajo es **auditar
una implementación de RLS por QR-inversa (square-root) en punto fijo, entender por qué
NO rindió lo esperado, y proponer mejoras concretas** — y, con eso aprendido, aconsejar
cómo aplicar un solve tipo raíz-cuadrada/QRD al **block-online WPE** (el objetivo real).
Sé crítico y cuantitativo; desafiá las premisas. Adjunto el paper de NARA-WPE.

## Objetivo del proyecto
Portar **dereverberación WPE multicanal a punto fijo** para implementarla como IP en
**Vitis HLS (C++→RTL)** sobre una **Xilinx KV260 (Zynq UltraScale+ ZU5EV)**, idealmente
con **memoria SOLO on-chip (sin DDR)**. Es el front-end de un beamformer asistido por
visión. Recursos on-chip: **~23 Mbit** (BRAM ~5.1 Mbit + URAM ~18.4 Mbit), ~1248 DSP,
~117k LUT.

Parámetros del sistema: taps K∈{5..10}, delay=2–3, canales **M=8**, bins F=257
(STFT size=512, shift=128, fs=16 kHz). KM = K·M (=40 a K=5, =80 a K=10). El WPE estima,
por bin de frecuencia, un filtro de predicción G (KM×M) que resta la reverberación tardía
predecible del multicanal: `X_t = Y_t − G^H · ỹ_t`, con `ỹ_t` = regresor apilado de los
K frames pasados (a partir de delay).

## Las dos arquitecturas WPE en juego
- **Online RLS (recursivo):** mantiene por bin `P = R⁻¹` (KM×KM) y actualiza P y G cada
  frame (lema de inversión + ganancia de Kalman). Memoria O(F·KM²) (cuadrática en taps).
- **Block-online (batch por bloque):** cada `block_shift` frames rearma
  `R = Σ (1/λ_t) ỹ_t ỹ_t^H` (KM×KM) y `P = Σ (1/λ_t) ỹ_t y_t^H` sobre una ventana trailing
  de L frames pasados, factoriza R (Cholesky) y resuelve `R·G = P`; aplica G congelado.
  R es SCRATCH (no se guarda); memoria persistente O(F·KM·M) (lineal en taps) + el buffer
  de ventana (L·F·M). SIN recursión → estable estructuralmente.

## HALLAZGO CENTRAL de nuestra investigación (el motivo de todo esto)
Medimos, en escena real (RIRs medidas MIRD, M=8, L=256):
- **cond(R) ≈ 10⁵–10⁶ en TODOS los bins** (R severamente mal-condicionada).
- **Coherencia entre micrófonos 0.78–0.99** (array compacto; a 250 Hz ≈ 0.99): los 8 mics
  son casi la misma señal → el regresor apilado ỹ (KM dims) es **severamente colineal**.
- **cond(R) ↔ max|G| correlacionan 0.82 (log-log):** los bins peor condicionados son los
  de G más grande. Es decir, **G explota porque R⁻¹ amplifica en una R casi-singular.**
- Como `R = A^H A` (A = matriz de datos/regresor ponderado), **las ecuaciones normales
  ELEVAN AL CUADRADO el número de condición:** cond(R)=cond(A)². Con cond(R)≈10⁶ →
  cond(A)≈10³. En bits: R pide ~20 bits, A pide ~10.

Consecuencias en punto fijo que YA observamos:
- **cov-RLS (guarda P=R⁻¹):** cliff duro — fiel a 24 bit, diverge a ≤20 bit (P pierde
  positividad bajo redondeo). Emulado en `nara_wrappers_fixed.py`.
- **block (Cholesky de R):** con carga diagonal chica (reg=1e-6, buena en float) la G se
  va a 22–90 → no cabe en palabra fija → se rompe. Hay que subir la carga diagonal a
  reg≈1e-2 para acotar G (a ≤1.3) y que 16-bit funcione — **pero reg=1e-2 cuesta calidad**
  (~0.13–0.24 de ΔPESQ de margen perdido). La reg alta es un parche por *formar R*.
- **QRD-RLS (raíz-cuadrada):** propaga un factor triangular L con `P = L·L^H`, PSD POR
  CONSTRUCCIÓN → tolera ~la mitad de bits. Target: estable/fiel a **16 bit**.

La tesis que queremos que audites: **un solve tipo QRD / raíz-cuadrada (trabajar sobre A,
cond≈10³, en vez de sobre R=A^HA, cond≈10⁶) debería dar estabilidad de 16 bits SIN la
carga diagonal pesada** → recuperando el margen de calidad. Esto es lo que motivó la
implementación QRD-RLS. **PERO esa implementación QRD "no tuvo mucho éxito"** (no alcanzó
la ventaja esperada / rindió por debajo del objetivo de 16 bit). Necesitamos entender por qué.

## TU TAREA
1. **Auditá `qrd_wpe_fixed.py`** (RLS inverse-QRD square-root, punto fijo):
   - ¿Es correcta la derivación (pre-array de QR-inversa, rotaciones de Givens, factor de
     conversión β, ganancia de Kalman k)? ¿El float_ref reproduce nara exactamente?
   - ¿Por qué NO llega al beneficio esperado (16 bit)? Buscá: normalización/rango dinámico
     de L y de los datapaths transitorios; formatos de las rotaciones (coseno/seno de
     Givens, sqrt, recíproco); acumulación de fase; saturación; el `int_bits` por señal;
     si algún transitorio de rango dinámico enorme (power/1/λ, β, s=||v||²) se está
     cuantizando y rompe la positividad/estabilidad.
   - Sugerí mejoras concretas (formatos, escalado, orden de operaciones, variante del
     algoritmo — p.ej. QRD-RLS directa vs inversa, Givens vs Householder, CORDIC).
2. **Con eso, recomendá cómo aplicar raíz-cuadrada/QRD al BLOCK** (el objetivo real):
   - En el block NO hay recursión: por bloque se arma A (ỹ ponderado) y se resuelve
     `R·G=P`. En vez de `R=A^HA` + Cholesky, un **QR de A** (o Cholesky de R vía QR de A)
     evitaría el cuadrado del condicionamiento. ¿Cómo se mapea a HLS on-chip? ¿Conviene
     back-substitution sobre el factor R triangular de la QR, o resolver por Q,R?
   - ¿Reduce/elimina la necesidad de la carga diagonal reg=1e-2 (y su costo de calidad)?
   - Trade-offs de recursos (DSP/LUT/BRAM, sqrt/div, latencia) en HLS.

## Archivos de referencia (repo Vision-Aided-Beamformer, env conda `tesis_beam`)
- **`src/dereverberation/qrd_wpe_fixed.py`** ← AUDITÁ ESTO. Inverse-QRD-RLS square-root en
  punto fijo. Tiene `QRDFixedPointConfig` (formatos por señal), `qrd_wpe_step_fixed`,
  `process_qrd_wpe_online_fixed`, y un self-test en `__main__` (barrido de word length +
  comparación contra float y contra nara).
- **`src/dereverberation/nara_wrappers_fixed.py`** ← referencia de la emulación de punto
  fijo del cov-RLS (`FixedPointConfig`/`Fx`/`FxStats`, `online_wpe_step_fixed`,
  `process_wpe_online_fixed`) Y del **block fixed** (`process_wpe_block_online_fixed`,
  `_estimate_block_filter_fixed`). Misma filosofía: se cuantiza SOLO lo que un FPGA
  ALMACENA (buffer STFT `in`, filtro `g`, salida `pred`; en el cov-RLS también P); los
  transitorios de rango dinámico enorme (power/nom/denom/recíproco/Kalman) quedan
  float/block-float.
- **`src/dereverberation/nara_wrappers.py`** ← versiones FLOAT de referencia:
  `process_wpe_online` (RLS float, == nara) y `process_wpe_block_online` /
  `process_wpe_block_online_with_components` (block float; el solve es
  `_block_cholesky_solve` = Hermitianiza + carga diagonal + `np.linalg.solve`).

## Referencia del algoritmo NARA-WPE (float, la "verdad")
- Paquete `nara_wpe` (pip). Módulos clave:
  - `nara_wpe.wpe.online_wpe_step(input_buffer, power_estimate, inv_cov, filter_taps,
    alpha, taps, delay)` — el paso RLS recursivo que emulan tanto el cov-fixed como el QRD.
  - `nara_wpe.wpe.wpe(Y, taps, delay, iterations)` — WPE batch/offline (EM alternante:
    estima potencia 1/λ desde el X dereverberado, arma R,P y resuelve). El block reusa sus
    primitivas: `build_y_tilde`, `get_power_inverse`, `get_correlations_v6`
    (`R=(ỹ·1/λ)·ỹ^H`, `P=(ỹ·1/λ)·y^H`), `perform_filter_operation_v5` (`X=Y−G^H·ỹ`).
  - `nara_wpe.utils.stft/istft`.
- **Paper adjunto:** Drude et al., "NARA-WPE: A Python package for weighted prediction
  error dereverberation…". Define WPE offline (iterativo) y online (recursivo con factor
  de olvido alpha). El olvido por defecto alpha=0.9999 (memoria efectiva ~80 s).

## Qué NO hace falta que cuestiones (ya está decidido / medido)
- El WPE en sí (que dereverbera y ayuda al beamformer) — ya validado con métricas
  objetivas (PESQ/STOI/SIR) sobre RIRs reales. El block ≥ online en calidad float.
- Que la memoria on-chip sea la restricción y que el cuadrado del condicionamiento sea el
  problema numérico — está medido (cond(R)≈10⁶, coherencia mics ≈0.9).
El foco es puramente: **por qué el QRD fixed no rindió, cómo arreglarlo, y cómo llevar la
raíz-cuadrada al solve del block para ganar 16-bit sin el costo de calidad de la carga
diagonal.**
