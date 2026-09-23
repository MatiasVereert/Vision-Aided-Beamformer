# Prompt para el benchmark de kernels en la Kria

*(pasar tal cual al agente con acceso a la Kria)*

---

Necesito medir, **sobre la Kria KV260 (Cortex-A53, ARMv8-A)**, cuánto cuestan los
dos kernels de un beamformer MVDR que voy a portar a C++. No hace falta el
algoritmo completo ni audio: son dos kernels aislados con datos sintéticos.

## Dimensiones (fijas)

- `K = 257` bins de frecuencia
- `M = 12` micrófonos
- Los datos son **complejos**. Medir **dos veces**: una con `double`
  (`complex<double>`) y otra con `float` (`complex<float>`).
- El sistema corre a **125 frames por segundo** (hop de 128 muestras a 16 kHz),
  o sea **8 ms de presupuesto por frame**.

## Estado persistente (alocarlo completo, esto importa)

```
Num_XX, Num_NN :  2 arreglos de K*M*M complejos
Den_XX, Den_NN :  2 arreglos de K reales
```

Con `double` eso son **1.16 MB**; con `float`, **578 kB**. La L2 compartida del
chip es de 1 MB, así que parte de lo que quiero saber es si el estado entra o no.

**Por eso el benchmark tiene que recorrer los 257 bins de verdad, en orden, sobre
el arreglo completo.** Medir una sola matriz de 12×12 en un bucle daría un número
optimista y sin valor: esa matriz vive en L1 y no es el caso real.

## Kernel 1 — `update` (corre en TODOS los frames)

Entrada por frame: `X[K][M]` complejo, y dos vectores reales `m_s[K]`, `m_n[K]`
en [0,1]. Con `alpha = 0.99` escalar:

```
para cada bin k:
    R = X[k] * X[k]^H                     # producto externo, M x M complejo
    Num_XX[k] = alpha*Num_XX[k] + m_s[k]*R
    Den_XX[k] = alpha*Den_XX[k] + m_s[k]
    Num_NN[k] = alpha*Num_NN[k] + m_n[k]*R
    Den_NN[k] = alpha*Den_NN[k] + m_n[k]
```

No tiene divisiones ni ramas: es lectura + multiplicar-acumular sobre todo el
estado. Es el kernel que sospecho limitado por memoria.

## Kernel 2 — `solve` (corre cada P frames; P es lo que quiero dimensionar)

```
para cada bin k:
    A = Num_XX[k] / (Den_XX[k] + 1e-15)        # M x M complejo
    B = Num_NN[k] / (Den_NN[k] + 1e-15)
    A = (A + A^H)/2 ;  B = (B + B^H)/2         # forzar hermitianas
    t = real(traza(B))
    B = B + I * (1e-9 * t/M + 1e-12)           # carga diagonal
    X = B^{-1} A                               # resolver B*X = A, M columnas
    lam = real(traza(X))
    w[k] = X[:, ref] / (lam + 1e-15)           # ref = 6; w[k] es M complejo
```

`B` es hermitiana definida positiva después de la carga, así que lo natural es
**Cholesky + sustitución hacia adelante y hacia atrás**, con las M columnas de
`A` como lados derechos. No hace falta ninguna descomposición espectral.

## Qué quiero que hagas

1. Escribí un programa C++ autónomo (un solo archivo, sin dependencias externas;
   nada de Eigen ni BLAS — quiero medir código propio, que es lo que voy a
   portar). Datos sintéticos: `X` aleatorio, máscaras aleatorias en [0,1].
2. Compilalo **en la Kria** con `-O3 -mcpu=cortex-a53` (probá también
   `-ffast-math` y reportá si cambia algo).
3. Medí, **un solo hilo**, con el estado ya caliente (descartá las primeras
   iteraciones) y sobre muchas repeticiones:
   - `update`: tiempo por frame (los 257 bins).
   - `solve`: tiempo por pasada completa (los 257 bins).
   - Las dos cosas en `double` y en `float`.
4. **Verificá que el `solve` está bien** antes de creerle al tiempo: para algún
   bin, comprobá que `B * X ≈ A` (residuo relativo). Un Cholesky mal escrito es
   más rápido y da cualquier cosa.
5. Reportame:
   - una tabla: kernel × precisión → ms, y qué porcentaje de los 8 ms es;
   - la aceleración de `float` contra `double` en cada kernel;
   - el **P mínimo** que hace que `update + solve/P` entre en 8 ms, en cada
     precisión, con el cálculo a la vista;
   - si tenés `perf` disponible, fallos de caché de última instancia (LLC) de
     cada kernel y precisión — es el dato que explicaría una aceleración de
     `float` mayor que 2×.

## Contexto, por si ayuda a interpretar

En una x86 de escritorio, con numpy, `solve` tarda ~1.44 ms y todo lo demás del
sistema ~0.25 ms por frame. Espero que el A53 sea varias veces más lento, y lo
que necesito saber es **cuánto** y **si `float` cambia el panorama** — no si el
algoritmo funciona, que eso ya está validado.

Si algo del enunciado no cierra con lo que ves en la máquina, decímelo en vez de
adaptarlo por tu cuenta.
