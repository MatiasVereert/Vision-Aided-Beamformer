import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Parámetros del array
# -----------------------------------------------------------
c = 343.0                     # velocidad del sonido
f = 2000                      # frecuencia de trabajo (Hz)
w = 2 * np.pi * f
lam = c / f

M = 8                         # número de micrófonos
d = lam / 2                   # espaciado inter-elemento
x_mic = np.arange(M) * d      # posiciones en un ULA 1D en el eje x

# -----------------------------------------------------------
# Función: steering vector near-field
# -----------------------------------------------------------
def steering_vector(theta_deg, r):
    """
    Near-field steering vector para un ULA 1D.
    theta: ángulo en grados
    r: distancia a la fuente
    """
    theta = np.deg2rad(theta_deg)
    
    # posición de la fuente en coordenadas cartesianas
    xs = r * np.cos(theta)
    ys = r * np.sin(theta)
    
    # distancia desde cada micrófono a la fuente
    dist = np.sqrt((x_mic - xs)**2 + ys**2)
    
    # steering vector (fase es k * d)
    k = w / c
    return np.exp(-1j * k * dist)

# -----------------------------------------------------------
# Construcción de la matriz A (región objetivo)
# -----------------------------------------------------------
# Definimos una región pequeña alrededor del ángulo 90°
thetas_region = np.linspace(80, 100, 5)   # +-10 grados
r_region = 1.0                             # distancia fija de referencia

A = []

for th in thetas_region:
    a = steering_vector(th, r_region)
    
    # Según el paper, A se arma con partes real e imaginaria separadas:
    A.append(np.real(a))
    A.append(np.imag(a))

A = np.array(A)   # dimensión (2*N_region) x M

# -----------------------------------------------------------
# SVD de A y construcción de C y f
# -----------------------------------------------------------
U, S, Vh = np.linalg.svd(A, full_matrices=False)

# Seleccionamos las L singular values más grandes.
# Para ejemplo, tomemos L=2
L = 2
C = Vh[:L, :].T     # cada columna es un vector de restricción
f = np.ones(L)      # unit gain en la región objetivo

# -----------------------------------------------------------
# Construcción de w_opt usando R = I
# -----------------------------------------------------------
R = np.eye(M)
Rinv = R            # trivial
w_opt = Rinv @ C @ np.linalg.inv(C.conj().T @ Rinv @ C) @ f

# -----------------------------------------------------------
# Cálculo de la sensibilidad angular
# -----------------------------------------------------------
thetas_test = np.linspace(0, 180, 720)
gain = []

for th in thetas_test:
    a = steering_vector(th, r_region)
    g = np.abs(np.vdot(w_opt, a))**2   # |w^H a|^2
    gain.append(g)

gain = np.array(gain)

# Normalizar por el valor en el ángulo focal (aprox 90°)
idx_focal = np.argmin(np.abs(thetas_test - 90))
gain_norm = gain / gain[idx_focal]

# -----------------------------------------------------------
# Graficar
# -----------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(thetas_test, 10*np.log10(gain_norm + 1e-12))
plt.axvline(90, color='r', linestyle='--')
plt.title("Array Gain Sensitivity (normalizada)")
plt.xlabel("Ángulo (deg)")
plt.ylabel("Sensibilidad (dB)")
plt.grid(True)
plt.show()
