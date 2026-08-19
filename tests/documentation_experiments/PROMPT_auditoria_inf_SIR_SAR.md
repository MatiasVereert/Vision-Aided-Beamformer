# Auditoría: `inf`/`NaN` en SIR y SAR tras migrar `metrics.py` a `fast_bss_eval`

Vos rearmaste `src/evaluation/metrics.py` para calcular SDR/SIR/SAR con
`fast_bss_eval` (en reemplazo de `mir_eval`, ~3x más rápido). Al graficar los
resultados aparecen `inf`/`-inf` en las columnas de SIR y SAR, que contaminan los
promedios y las figuras. Necesito que audites el diagnóstico de abajo y me digas
cuál es el tratamiento correcto (a nivel métrica y/o diseño experimental), porque el
autor del pipeline y yo llegamos a una conclusión que quiero validar con vos.

## Contexto del sistema

- Tesis de realce de voz. Sistema NM-MVDR = 8 mics → WPE → máscara DTLN → Souden
  MVDR → post-filtro (BAN/PF). fs=16 kHz, arreglo lineal MIRD, RT60 ∈ {160,360,610} ms.
- Benchmark: `src/evaluation/full_benchmark_test_dtln_mird.py` (orquestador
  `run_mird_grid_search`). Por cada escena computa métricas para varias señales:
  `base` (mic 0 crudo, entrada), `wpe`, `proc` (salida del procesador),
  `dtln_alone` (DTLN mono). Las columnas `Delta_tot_M = proc − base`, etc.
- Métrica primaria de fidelidad = **SI-SDR** (Le Roux 2019). SIR/SAR son secundarias.
- Ruido de sensor: se inyecta en `microphone.py::emulate()` normalizado en RMS
  **ponderado-A** a `snr_db=60` (dBA). Medido: para voz real eso da **~63 dB de SNR
  de banda ancha** (la voz pierde energía bajo ponderación-A ⇒ el piso ancho queda
  aún más profundo que 60). BSS-Eval trabaja en banda ancha.

## Cómo se calcula hoy (código actual)

```python
import fast_bss_eval

def _fast_sdr_sir_sar(reference, estimation):
    ref = np.ascontiguousarray(reference, dtype=np.float64)
    est = np.ascontiguousarray(estimation, dtype=np.float64)
    sdr, sir, sar = fast_bss_eval.bss_eval_sources(
        ref, est, compute_permutation=False, use_cg_iter=10
    )
    return float(sdr[0]), float(sir[0]), float(sar[0])
```

Contexto multi-fuente (para poder separar interferencia de artefactos) se arma así:

```python
# ref_crop = target early (referencia del target)
# noise_total_crop = interf_early + interf_late + target_late  (todo lo que NO queremos)
speech_source     = np.stack([ref_crop, noise_total_crop])          # (2, N)
dummy_noise_floor = np.random.default_rng(0).standard_normal(len(deg_crop)) * 1e-10
speech_prediction = np.stack([deg_crop, dummy_noise_floor])          # (2, N)
sdr, sir, sar = _fast_sdr_sir_sar(speech_source, speech_prediction)  # se leen los [0]
```

Notas del autor en el código:
- `use_cg_iter` es OBLIGATORIO: el solver directo (`use_cg_iter=None`) tira
  `ValueError` en `np.linalg.solve` en este entorno (numpy/fast_bss_eval). El
  gradiente conjugado lo evita. **Reproducido: `use_cg_iter=None` → "solve: Input
  operand ..." ValueError.**
- `compute_permutation=False` ⇒ devuelve 3 valores; se asume el orden de fuentes.
- La fila `[0]` es independiente de las demás; el 2º canal dummy (1e-10) es solo para
  cumplir la forma; se afirma que no afecta los métricos del canal [0].

## Síntoma observado (parquet real de P4, 360 filas)

- `base_SAR_early`: **20 valores = +inf** ⇒ `Delta_tot_SAR = proc − base = −inf`.
- `proc_SIR_early`: **9 valores = +inf** ⇒ `Delta_tot_SIR = +inf`.
- Aparecen **solo** en la intersección de escenas triviales: `rt60=160 ms`
  (casi anecoico) **+ iSIR=10 dB** (target 10 dB sobre interferencia) **+ 1 interferente**.
- En una fila con `base_SAR=inf`: `base_SDR=8.25 dB`, `base_SIR=8.22 dB`,
  `base_SI-SDR=4.96 dB`. **Es decir NO es "base == salida"** (el SDR es modesto).

## Diagnóstico (a validar)

El `inf` es `10·log10(energía_de_un_componente ≈ 0)`. Un componente de la
descomposición BSS-Eval cae por debajo de la resolución numérica del solver truncado
(`use_cg_iter=10`, ~55–58 dB por debajo de la energía dominante) ⇒ se computa como
`0.0` exacto ⇒ `log10(0) = inf`. Evidencia reproducida con señales sintéticas:

1. **SAR (caso `base`)** — el término de "artefactos" ES el ruido térmico (lo único
   de la mezcla cruda que no explica ninguna referencia). A −63 dB (piso real) el
   solver lo redondea a 0 → `inf`. Barrido `use_cg_iter`:

   ```
   use_cg_iter |  SDR   SIR   SAR
        10     |  8.3   8.3   inf
        50     |  8.3   8.3   inf
       100     |  8.3   8.3   59.00   <- ya resuelve el piso térmico ~ -60 dB
   ```
   `mir_eval` sobre el mismo input da SAR finito pero absurdo (60–277 dB según el
   piso). O sea el valor "correcto" tampoco es informativo.

2. **SIR (caso `proc`)** — el término de "interferencia" es la fuga del interferente
   tras el beamformer, que es **independiente** del ruido térmico (el térmico es
   incorrelado ⇒ cae en artefactos, no en interferencia). Con la fuga fija en −60 dB
   y variando el piso térmico:

   ```
   térmico[dB] |  SDR    SIR    SAR
      -63      |  57.0   inf   56.6
      -45      |  45.2   inf   44.3
      -35      |  35.0   inf   35.0
      -20      |  20.3   42.7  20.4
   ```
   Subir el piso mueve el SAR pero **no** toca el SIR: la fuga sigue por debajo de la
   resolución → SIR sigue `inf` hasta que el térmico es tan alto (−20 dB) que
   contamina la proyección.

**Conclusión del pipeline:** son bordes degenerados de BSS-Eval en las escenas más
fáciles (nulos muy profundos y/o artefactos bajo el piso), no un bug de construcción
de señales. En el HW real reverberante estos nulos de >50 dB no ocurrirían; son
idealizaciones del simulador.

**Corolario importante:** bajar `snr_db` (subir el piso) arreglaría el `SAR=inf`
(base) pero **NO** el `SIR=inf` (proc). Por eso descartamos "arreglarlo desde el
origen" vía el nivel de ruido.

## Mitigación aplicada hasta ahora

- `metrics.py`: se colapsa `±inf → NaN` en SDR/SIR/SAR en la fuente
  (`results['SIR'] = sir if np.isfinite(sir) else np.nan`, etc.).
- Notebook de análisis: `load_latest()` reemplaza `±inf → NaN` al cargar, y los
  `groupby().mean()` (nan-aware) promedian sobre las escenas no-degeneradas.
- Efecto: se descartan ~5.5% de filas (solo las triviales) del promedio de SIR/SAR;
  SI-SDR y SDR quedan intactos (nunca dan inf).

## Lo que necesito que audites / respondas

1. **¿Es `inf→NaN` el tratamiento correcto**, o hay un uso numéricamente estable de
   `fast_bss_eval` que devuelva valores finitos y confiables en estos bordes sin
   pagar el costo de `use_cg_iter=100`? (¿`fast_bss_eval` expone un piso/regularización
   para el residual? ¿conviene un `use_cg_iter` intermedio + clamp?)
2. **¿El armado 2-fuentes + `dummy_noise_floor=1e-10` es sólido?** ¿Podría ese dummy
   casi-nulo estar mal-condicionando la proyección LS y *causando* o agravando la
   degeneración del canal [0]? ¿Convendría un enfoque single-source para SDR/SI-SDR y
   uno multi-fuente aparte solo para SIR? ¿Verificaste que `[0]` es realmente
   independiente del dummy en TODOS los regímenes, no solo en el típico?
3. **`use_cg_iter=10`**: ¿es una elección deliberada por velocidad? ¿Qué error vs
   `mir_eval` introduce en las celdas NO degeneradas (las que sí importan)? ¿Hay
   riesgo de sesgo sistemático (no solo en los bordes) por truncar el CG?
4. **Metodológico**: dado que SIR/SAR son degenerados justo donde el sistema anda
   "demasiado bien", ¿tiene sentido reportarlos ahí, o directamente apoyarse en SI-SDR
   (primaria) y reportar SIR/SAR solo en el régimen no-degenerado? ¿Cómo lo
   documentarías para una tesis (exclusión honesta vs. techo físico vs. otra)?
5. **Chequeo cruzado**: ¿podés confirmar que `fast_bss_eval` y `mir_eval` coinciden a
   <0.1 dB en las celdas no degeneradas con ESTE armado (2 fuentes + dummy), no solo
   en el caso canónico? Si hay discrepancia, ¿en qué dirección y magnitud?

## Cómo reproducir

Entorno: `/home/matias/miniconda3/envs/tesis_beam/bin/python`.
Parquet real: `tests/documentation_experiments/results/P4_mono_postfiltro_*/**/mird_benchmark_metrics.parquet`.
Función bajo prueba: `src/evaluation/metrics.py::_fast_sdr_sir_sar` y
`evaluate_full_pipeline`. Ver los snippets de barrido de `use_cg_iter` y de piso
térmico arriba (señales coloreadas sintéticas, N=16000·5).
