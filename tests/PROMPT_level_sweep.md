# PROMPT: barrido de nivel de entrada a lazo abierto (OFB_MVDR)

## Contexto

Todo el benchmark MIRD corre con la mezcla normalizada por PICO: `mix_and_normalize`
([simulate_acoustics_v1.py:517](../src/propagation/simulate_acoustics_v1.py:517)) escala
globalmente para que el pico sobre todos los micrófonos quede en 0.99. Eso es correcto
como control experimental --- fija el nivel para que no confunda los ejes que sí se
estudian (iSIR, sala, ángulo) --- pero significa que **el sistema nunca fue evaluado
fuera de ese nivel**, y el port a C++ para tiempo real sí va a ver nivel variable
(distancia al hablante, ganancia del ADC, volumen del locutor).

El bloque sensible es el DTLN. Medido fuera de este harness (un archivo de voz limpia,
`model_quant_1.tflite`, máscara comparada contra la de nivel nominal):

| offset | corr. con nominal | media abs | masa `(1-m)^8` |
|---|---|---|---|
| −40 dB | −0.11 | 0.63 | 0.157 (×100 vs nominal) |
| −20 dB | 0.81 | 0.068 | 0.0018 |
| −12 dB | 0.89 | 0.053 | 0.0023 |
| −6…+6 dB | ≥ 0.94 | ≤ 0.033 | ~0.0015 |
| +12 dB | 0.77 | 0.055 | 0.0020 |
| +20 dB | 0.52 | 0.163 | 0.0167 |
| +30 dB | 0.45 | 0.272 | 0.0388 |

O sea: la degradación es suave y de dos lados, y fuera de rango además **desbalancea las
dos ramas del SCM** (la masa de la rama de ruido se va por un factor 25-100). Pero eso
es correlación de máscara sobre un archivo, no PESQ sobre las celdas reales.

## Objetivo

Medir la **curva de degradación del sistema completo contra el offset de nivel de
entrada**, a lazo abierto (ganancia fija por corrida, sin control automático).

Esto decide dos cosas que hoy no sabemos:

1. **si hace falta un control de nivel automático**, y en qué rango;
2. **cuánto tiene que valer la zona muerta** de ese control, con datos del benchmark en
   vez de con la tabla de arriba.

**Fuera de alcance: NO implementar el control automático.** Este test es el que decide
si ese bloque se construye. Si la curva resulta plana en el rango operativo realista,
el bloque no se justifica y hay que decirlo.

## Diseño

Un decorador de procesador que aplica la ganancia a la entrada y la **deshace a la
salida**, de modo que el camino de métricas quede idéntico y lo único que cambie sea lo
que ve la red:

```python
class LevelScaled:
    """Aplica g a la entrada del procesador y 1/g a su salida.

    Aísla el efecto del nivel SOBRE LA RED: el beamformer es invariante a escala
    (Souden normaliza por la traza) y las métricas ven exactamente la misma señal
    que sin el decorador. Lo único que cambia es el punto de operación del DTLN.
    """
    def __init__(self, inner, gain_db):
        self.inner, self.gain_db = inner, float(gain_db)
    def process(self, mic_signals, scene_config):
        g = 10.0 ** (self.gain_db / 20.0)
        y, w = self.inner.process(mic_signals * g, scene_config)
        return y / g, w
```

La ganancia se aplica DESPUÉS de `mix_and_normalize`, así que el iSIR y el SNR de la
escena no cambian (el iSIR es un cociente, invariante a ganancia común).

El procesador interno es `OFB_MVDR` ([bf_wrappers.py:3350](../src/evaluation/bf_wrappers.py:3350)),
construido igual que en `tests/ofb_auto_benchmark.py`. Grilla de procesadores:

```python
LEVELS_DB = [-24, -18, -12, -6, 0, +6, +12, +18, +24]
processors = {f"L{db:+03d}": LevelScaled(OFB_MVDR(smooth=...), db) for db in LEVELS_DB}
```

## Verificaciones OBLIGATORIAS antes de creerle al barrido

Si alguna falla, el barrido no mide lo que dice medir. Ponelas como asserts en el script.

1. **`gain_db=0` es bit a bit idéntico al procesador sin decorar.** `g = 1.0` exacto en
   IEEE754, así que `x*1.0/1.0` no pierde nada: usar `np.array_equal`, no `allclose`.
   Esto es lo que garantiza que agregar este eje no invalida ningún resultado previo.
2. **El iSIR de la escena no cambia con el offset.** Verificar sobre una celda que las
   métricas de ENTRADA (las `*_in_*` del dataframe, no las `Delta_bf_*`) son idénticas
   entre offsets. Si cambian, la ganancia se está aplicando en el lugar equivocado.
3. **La invarianza de escala del beamformer.** Comparar los pesos `w` devueltos a 0 dB
   y a +24 dB con máscara CONGELADA (si no hay un gancho para eso, saltear este punto y
   anotarlo): deberían coincidir a ~1e-12. Toda la diferencia del barrido tiene que
   venir de la máscara, no del core.

## Grilla

Un solo eje de iSIR alcanza: el nivel y el iSIR son independientes por construcción, así
que cruzar toda la grilla es trabajo sin información. Partí de la config base de
`tests/ofb_auto_benchmark.py` y usá:

```python
'rt60': [0.360, 0.610], 'target_angle': [0], 'target_dist': [1.0],
'interf_configs': [[(45, 1.0)]], 'isir_db': [0, 10],
```

Agregá `--quick` (una sola celda, 3 niveles) para plomería, igual que los otros scripts.

## Qué reportar

1. **La tabla principal**: Δ métricas (`Delta_bf_PESQ_early`, `STOI`, `SDR`, `SIR`,
   `SAR`) contra el offset, mediana sobre celdas. La lectura es la forma de la curva,
   no el valor absoluto: dónde empieza a caer y cuán rápido.
2. **La explicación**, con `return_diag=True`: masa media de cada rama del SCM por
   offset (`mean(m^8)` y `mean((1-m)^8)` sobre `m_out` y `m_ref`), y el `isir_db`
   estimado. Si la hipótesis de arriba es correcta, la caída de PESQ tiene que
   correlacionar con el desbalance de masa, y el estimador de iSIR tiene que
   descalibrarse fuera de rango --- lo que además movería la agenda del post-filtro.
   Esos son dos mecanismos distintos y conviene poder separarlos.
3. **La conclusión en dB**: el rango donde ΔPESQ se mantiene dentro de 0.05 del nominal.
   Ese número es la zona muerta del control de nivel, si es que se construye.

## Regla de decisión, fijada ANTES de ver los resultados

- Si ΔPESQ cae **más de 0.1 dentro de ±12 dB** → el control de nivel se justifica, y la
  zona muerta sale del punto 3.
- Si se mantiene dentro de 0.05 en todo ±12 dB y solo cae en los extremos → alcanza con
  una constante de calibración fija y documentada; nada de control automático.
- Si la curva es asimétrica (probable: la tabla de máscara sugiere que hacia abajo
  aguanta más), la zona muerta también tiene que serlo.

## Entregable

`tests/ofb_level_sweep.py`, siguiendo la convención del repo:

- docstring de módulo con la pregunta, el diseño, **la tabla de resultados medidos** y la
  conclusión (mirá `tests/ofb_auto_benchmark.py` como molde);
- `--quick`, `--out-dir`, `--duration`, `--smooth` como los demás;
- salida a `tests/dataset_out/ofb_level/`;
- las tres verificaciones como asserts que corren siempre, no como comentarios.

Ejecutar con `conda activate tesis_beam`.
