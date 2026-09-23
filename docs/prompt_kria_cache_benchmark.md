# Prompt 2 para la Kria: el `update` y la competencia por la caché

*(continuación del benchmark de kernels; pasar tal cual)*

---

El benchmark anterior midió `chol`, `eigh` y `solve` aislados y dio, sobre 257
bins:

```
float   chol 3.605 ms   eigh 21.324 ms   solve 4.549 ms
double  chol 4.528 ms   eigh 30.266 ms   solve 6.018 ms
```

Con eso descarté el `eigh` (no entra en el presupuesto de 8 ms por frame ni
cerca). Quedan dos preguntas que ese test no contesta.

## Pregunta 1 — falta medir el kernel `update`

Corre en **todos** los frames (el `chol`/`solve` puede correr cada P frames, éste
no) y toca todo el estado. Con `K=257`, `M=12`:

```
estado: Num_XX, Num_NN  (2 arreglos de K*M*M complejos)
        Den_XX, Den_NN  (2 arreglos de K reales)

por frame, con alpha = 0.99 y entradas X[K][M] complejo, m_s[K], m_n[K] reales:
    para cada bin k:
        R = X[k] * X[k]^H                      # producto externo M x M
        Num_XX[k] = alpha*Num_XX[k] + m_s[k]*R
        Den_XX[k] = alpha*Den_XX[k] + m_s[k]
        Num_NN[k] = alpha*Num_NN[k] + m_n[k]*R
        Den_NN[k] = alpha*Den_NN[k] + m_n[k]
```

Medilo en `float` y en `double`, recorriendo los 257 bins en orden sobre el
arreglo completo (no una matriz suelta en un bucle: eso viviría en L1 y mentiría).

## Pregunta 2 — la competencia por la caché, que es la que me importa

El estado del beamformer son **1.16 MB en `double`** y **578 kB en `float`**. La
L2 del chip es de 1 MB compartida. Pero en el sistema real, entre el `update` y
el `solve` de cada frame corren **dos invocaciones de una red neuronal cuyos
pesos son 378 kB** (int8), que barren la caché en el medio.

Sospecho que por eso el test aislado es optimista, y **más optimista para
`double` que para `float`**: la relación que midió (1.29x) es demasiado baja para
un kernel limitado por memoria, o sea que el estado todavía le entraba.

Quiero el mismo benchmark pero con la competencia presente. El bucle por frame
tiene que ser:

```
update(...)                    # los 257 bins
tocar_378kB()                  # el sustituto de la red
tocar_378kB()                  # la segunda red (los MISMOS 378 kB)
si toca: chol + solve(...)     # los 257 bins
```

Para `tocar_378kB()` alcanza con un buffer de 378 kB recorrido secuencialmente
acumulando (que el compilador no lo pueda eliminar: usá el acumulador en un
`sink` al final). **Los dos "invokes" tienen que recorrer el MISMO buffer**, que
es como va a quedar el sistema real: las dos redes comparten los pesos.

Si tenés forma de correr el tflite de verdad en la Kria, mejor que el sustituto,
pero no te trabes con eso: lo que quiero medir es la presión de caché, y un
barrido del mismo tamaño la reproduce.

## Qué quiero de vuelta

1. `update` solo: ms por frame, en `float` y `double`.
2. `update` con la competencia intercalada: ms por frame, las dos precisiones.
   **Cuánto se degrada cada una** respecto del punto 1 — ése es el número que
   busco.
3. Lo mismo para `chol + solve` (aislado contra intercalado).
4. El presupuesto total por frame, en las dos precisiones:
   `update + 2 invokes + (chol+solve)/P`, y el **P mínimo** que entra en 8 ms.
5. Si tenés `perf`: fallos de LLC por frame en los cuatro casos.

## Contexto

- Cortex-A53, 4 núcleos, L1D 32 kB por núcleo, L2 1 MB compartida.
- 125 frames por segundo, o sea 8 ms de presupuesto por frame.
- La decisión que depende de esto: si el port va en `double` o en `float`. Si
  `float` se degrada poco y `double` mucho, el salto no va a ser el 2x de los
  lanes de NEON sino bastante más, y eso define la fase 2 del proyecto.

Si los números no dan lo que la hipótesis predice, decímelo así como sale — un
resultado que refuta la hipótesis me sirve igual, y ajustar el test para que dé
lo esperado no.
