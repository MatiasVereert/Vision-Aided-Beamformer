# Prompt 3 para la Kria: las formas del `solve`, el `update`, y la caché

*(reemplaza al prompt 2; incluye todo lo que falta medir)*

---

Sigo dimensionando el port de un beamformer MVDR a la Kria KV260 (Cortex-A53).
El primer benchmark ya me dio, sobre 257 bins:

```
float   chol 3.605 ms   eigh 21.324 ms   solve 4.549 ms
double  chol 4.528 ms   eigh 30.266 ms   solve 6.018 ms
```

Con eso descarté el `eigh` (no entra ni cerca en los 8 ms por frame). Ahora
quiero medir **cuatro variantes de implementación** que cambian bastante la
cuenta, y una interacción con la caché.

## Dimensiones (fijas)

- `K = 257` bins, `M = 12` micrófonos, datos **complejos**.
- **125 frames por segundo**: 8 ms de presupuesto por frame.
- Todo en `float` y en `double` (o sea cada medición, dos veces).
- **Un solo hilo.**

## El estado persistente

```
Num_XX, Num_NN :  2 arreglos de K matrices M x M complejas    <- HERMITIANAS
Den_XX, Den_NN :  2 arreglos de K reales
```

Las dos matrices son **hermitianas** ($A = A^H$), así que se pueden guardar
enteras o sólo el triángulo inferior. El triángulo son 66 complejos fuera de la
diagonal más 12 reales en la diagonal (la diagonal de una hermitiana es real):
exactamente la mitad de la memoria.

| | completo | triángulo |
|---|---|---|
| `double` | 1.19 MB | 592 kB |
| `float` | 592 kB | 296 kB |

La L2 del chip es de 1 MB compartida, así que dónde cae cada variante importa.

## Los dos kernels

### `update` — corre en TODOS los frames (no se amortiza)

Entrada: `X[K][M]` complejo, `m_s[K]`, `m_n[K]` reales en [0,1]. `alpha = 0.99`:

```
para cada bin k:
    R = X[k] * X[k]^H                      # producto externo M x M, hermitiano
    Num_XX[k] = alpha*Num_XX[k] + m_s[k]*R
    Den_XX[k] = alpha*Den_XX[k] + m_s[k]
    Num_NN[k] = alpha*Num_NN[k] + m_n[k]*R
    Den_NN[k] = alpha*Den_NN[k] + m_n[k]
```

Con almacenamiento triangular sólo calculás y guardás la mitad de `R` (66
complejos + 12 reales `|X[k][m]|^2` en vez de 144 complejos).

### `solve` — corre cada P frames, y tiene DOS formas

Partiendo de:

```
A = Num_XX[k] / (Den_XX[k] + 1e-15)          # M x M hermitiana, y PSD
B = Num_NN[k] / (Den_NN[k] + 1e-15)          # M x M hermitiana
t = real(traza(B)) ;  B = B + I*(1e-9*t/M + 1e-12)      # carga diagonal
```

**Forma 1, "directa"** (la que ya mediste):

```
X = B^{-1} A                  # Cholesky de B + M lados derechos
lam = real(traza(X))
w[k] = X[:, ref] / lam        # ref = 6 ; w[k] es M complejo
```

Calcula las M columnas de `X` sólo para sumarle la diagonal, y después usa una
sola. Costo ≈ `M³/6 + M³` = **1.17 M³**.

**Forma 2, "Cholesky"** (la que quiero medir). Usa la identidad
`traza(B⁻¹A) = ‖L_B⁻¹ L_A‖_F²`, con `B = L_B L_B^H` y `A = L_A L_A^H`:

```
LB = cholesky(B)
LA = cholesky(A)                      # existe: A es PSD por construcción
Z  = resolver LB * Z = LA             # sustitución adelante, lado derecho TRIANGULAR
lam = suma de |Z_ij|^2                # norma de Frobenius al cuadrado
y  = resolver LB   * y = A[:, ref]    # UN solo lado derecho
w  = resolver LB^H * w = y
w[k] = w / lam
```

Costo ≈ `M³/6 + M³/6 + M³/6` = **0.50 M³**, o sea **2.3× menos trabajo**.

Verifiqué en Python que las dos formas dan **el mismo resultado hasta 4e-16**
(con el estado en régimen; al arranque difieren mientras `A` no alcanza rango
completo, que son los primeros M frames y no me preocupa).

Aprovechá que `Z` se resuelve contra un lado derecho **triangular**: las columnas
de `LA` tienen ceros abajo y no hace falta tocarlos (ahí está buena parte del
ahorro). Si lo implementás como un lado derecho denso, el ahorro se te va.

## Qué quiero medir

Las combinaciones que me importan, siempre en `float` y en `double`:

1. `update` con almacenamiento **completo**.
2. `update` con almacenamiento **triangular**.
3. `solve` forma **directa** (ya lo tenés, pero rehacelo en el mismo binario
   para que sea comparable).
4. `solve` forma **Cholesky**.
5. Las combinaciones 2+4 (la que creo que va a ganar) y 1+3 (la de referencia),
   corridas **como un frame completo**.

Y además, la interacción que sospecho que domina:

6. Lo mismo que (5), pero **intercalando entre el `update` y el `solve` dos
   barridos de un buffer de 378 kB** recorrido secuencialmente (es el sustituto
   de una red neuronal que corre ahí en el sistema real y barre la caché en el
   medio). **Los dos barridos tienen que recorrer el MISMO buffer.** Usá el
   acumulado en un `sink` al final para que el compilador no lo borre.

   Lo que busco de acá: **cuánto se degrada cada precisión** respecto de (5). La
   hipótesis es que `double` se degrada más que `float`, porque `double`
   triangular (592 kB) + red (378 kB) = 970 kB queda al filo del megabyte de L2,
   y `double` completo (1.19 MB) ya no entra. Puede que la hipótesis sea falsa:
   el tráfico son ~290 MB/s con acceso secuencial y el prefetcher puede
   absorberlo. Quiero el número, no la confirmación.

## Cómo

- C++ autónomo, un solo archivo, **sin Eigen ni BLAS** (quiero medir código
  propio, que es lo que voy a portar). Datos sintéticos.
- Compilado **en la Kria** con `-O3 -mcpu=cortex-a53`. Reportá también si
  `-ffast-math` cambia algo.
- Recorré siempre los 257 bins en orden sobre el arreglo completo. Medir una
  matriz suelta en un bucle daría un número optimista y sin valor: esa matriz
  vive en L1 y no es el caso real.
- Descartá las primeras iteraciones y promediá sobre muchas.

## Verificación antes de creerle a los tiempos

- Comprobá que la forma Cholesky y la directa dan **el mismo `w`** (error
  relativo, debería ser ~1e-6 en `float` y ~1e-15 en `double`). Un Cholesky mal
  escrito es más rápido y da cualquier cosa.
- Comprobá que el `update` triangular y el completo dan **el mismo estado**
  (comparando el triángulo).

Sin esas dos comprobaciones los tiempos no significan nada.

## Qué quiero de vuelta

1. Tabla: variante × precisión → ms (y % de los 8 ms).
2. La aceleración de: triangular contra completo (`update`), Cholesky contra
   directa (`solve`), y `float` contra `double` en cada una.
3. La degradación por la competencia de caché, punto (6) contra punto (5).
4. El **P mínimo** que hace que `update + 2 barridos + solve/P` entre en 8 ms,
   para la mejor combinación de cada precisión, con el cálculo a la vista.
5. Si tenés `perf`: fallos de LLC por frame en los casos de (5) y (6).

## Una observación sobre el margen

31.7 µs por bin para un Cholesky de 12×12 con 12 lados derechos son unas 10 mil
operaciones de punto flotante; a 1.33 GHz eso debería costar ~7 µs. O sea que la
primera implementación está ~4× por encima de lo que predice un cálculo grueso.
No es un reproche: es que probablemente haya un 2–3× ahí antes de tocar NEON. Si
ves algo evidente (divisiones dentro del bucle, indexación que impide vectorizar,
complejos como `std::complex` con aliasing), decímelo — pero **no optimices antes
de medir**, quiero la línea de base primero.

Si algo del enunciado no cierra con lo que ves en la máquina, decímelo en vez de
adaptarlo por tu cuenta. Un resultado que refute mi hipótesis me sirve igual.
