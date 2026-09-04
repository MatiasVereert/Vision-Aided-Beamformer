"""
Tests del banco de calibracion (`beamforming/mask/scm_calibration.py`).

Lo que se verifica es lo que hace falsable a todo el experimento:

  1. La familia parametrica CONTIENE a los dos cores en produccion:
         (nu=0, gamma=0) == MVDR_Souden_recursive_mask_fixed
         (nu=1, gamma=0) == MVDR_Souden_recursive_mask_subtract (mu=0)
     Si esto no se cumple bit a bit (a menos de round-off), el "ajuste" estaria
     comparando contra un sistema que no es el que corre en el benchmark, y
     cualquier ganancia reportada seria un artefacto.

  2. El PISO de la loss es cero: los pesos de Souden calculados con las SCM
     oracle tienen L_sinr ~ 0 dB frente a lambda_max(Phi_N^-1 Phi_S).

  3. L_sinr es invariante a la escala de w (es la invariancia de Souden) y
     L_dist no lo es (es la que fija la escala).

  4. La coherencia difusa es la matriz que corresponde: Hermitiana, diagonal 1,
     PSD, y -> matriz de unos en continua.

    pytest tests/test_scm_calibration.py -q
"""

import numpy as np
import pytest

from beamforming.mask.scm_calibration import (
    diffuse_coherence, eval_frame_indices, snapshot_scms_masked,
    snapshot_scms_oracle, parametric_weights, souden_weights,
    oracle_references, weight_loss, make_bands, bands_to_bin_params,
)
from beamforming.mask.souden_mvdr import (
    MVDR_Souden_recursive_mask_fixed, MVDR_Souden_recursive_mask_subtract,
)

K, T, M = 12, 60, 5
REF = M // 2


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20260831)
    X = (rng.standard_normal((K, T, M)) + 1j * rng.standard_normal((K, T, M))) / np.sqrt(2)
    # correlacion espacial no trivial: mezcla los canales con una matriz fija
    A = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    X = np.einsum("mn,ktn->ktm", A, X)
    mask_s = rng.uniform(0.05, 1.0, size=(K, T))
    mask_n = 1.0 - mask_s
    return X, mask_s, mask_n


def test_diffuse_coherence_es_valida():
    coords = np.array([[0.0, 0, 0], [0.04, 0, 0], [0.09, 0, 0], [0.17, 0, 0]])
    freqs = np.array([0.0, 100.0, 1000.0, 4000.0])
    G = diffuse_coherence(coords, freqs)

    assert G.shape == (4, 4, 4)
    assert np.allclose(G, np.swapaxes(G, -1, -2))              # simetrica
    assert np.allclose(np.diagonal(G, axis1=-2, axis2=-1), 1.0)
    # en continua el campo es perfectamente coherente -> matriz de unos
    assert np.allclose(G[0], np.ones((4, 4)))
    # PSD (la coherencia difusa lo es por construccion)
    assert np.linalg.eigvalsh(G).min() > -1e-9


def test_nu0_gamma0_reproduce_el_core_fixed(data):
    """(nu=0, gamma=0) tiene que dar los MISMOS pesos que el core base _fixed."""
    X, ms, mn = data
    ev = eval_frame_indices(T, 6, start_frame=10)

    _, W_core = MVDR_Souden_recursive_mask_fixed(
        X, ms, mn, min_loading=1e-2, alpha=0.97, save_weights=True,
        ref_mic_idx=REF)

    Pxx, Pnn = snapshot_scms_masked(X, ms, mn, ev, alpha=0.97)
    W_par, _ = parametric_weights(Pxx, Pnn, None, REF, nu=0.0, gamma=0.0,
                                  min_loading=1e-2)

    np.testing.assert_allclose(W_par, np.transpose(W_core[:, ev, :], (0, 1, 2)),
                               rtol=1e-8, atol=1e-12)


def test_nu1_gamma0_reproduce_el_core_subtract(data):
    """(nu=1, gamma=0) tiene que dar los MISMOS pesos que el core _subtract."""
    X, ms, mn = data
    ev = eval_frame_indices(T, 6, start_frame=10)

    _, W_core = MVDR_Souden_recursive_mask_subtract(
        X, ms, mn, min_loading=1e-9, alpha=0.97, mu=0.0, lambda_floor=1e-3,
        psd_project=True, save_weights=True, ref_mic_idx=REF)

    Pxx, Pnn = snapshot_scms_masked(X, ms, mn, ev, alpha=0.97)
    W_par, _ = parametric_weights(Pxx, Pnn, None, REF, nu=1.0, gamma=0.0, mu=0.0,
                                  min_loading=1e-9, lambda_floor=1e-3)

    np.testing.assert_allclose(W_par, W_core[:, ev, :], rtol=1e-7, atol=1e-12)


def test_shrinkage_interpola_entre_scm_y_modelo_difuso(data):
    """gamma=0 no toca nada; gamma=1 deja EXACTAMENTE (tr/M)*Gamma."""
    from beamforming.mask.scm_calibration import parametric_scms
    X, ms, mn = data
    ev = eval_frame_indices(T, 4, start_frame=10)
    Pxx, Pnn = snapshot_scms_masked(X, ms, mn, ev, alpha=0.97)

    coords = np.stack([np.linspace(0, 0.2, M), np.zeros(M), np.zeros(M)], axis=1)
    freqs = np.linspace(0, 8000, K)
    G = diffuse_coherence(coords, freqs)

    _, N0 = parametric_scms(Pxx, Pnn, G, nu=0.0, gamma=0.0)
    np.testing.assert_allclose(N0, Pnn, rtol=1e-12, atol=1e-14)

    _, N1 = parametric_scms(Pxx, Pnn, G, nu=0.0, gamma=1.0)
    tr = np.real(np.trace(Pnn, axis1=-2, axis2=-1))[..., None, None] / M
    np.testing.assert_allclose(N1, tr * G[:, None], rtol=1e-10, atol=1e-14)
    # el shrinkage conserva la traza (el target esta normalizado a tr/M en la diagonal)
    np.testing.assert_allclose(np.real(np.trace(N1, axis1=-2, axis2=-1)),
                               np.real(np.trace(Pnn, axis1=-2, axis2=-1)), rtol=1e-10)


def test_el_piso_de_la_loss_es_cero():
    """
    Con target de RANGO 1, los pesos de Souden calculados sobre las SCM oracle
    son el beamformer de maximo SINR -> L_sinr = 0 dB exacto.
    """
    rng = np.random.default_rng(7)
    a = rng.standard_normal((K, M)) + 1j * rng.standard_normal((K, M))
    s = rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))
    S = a[:, None, :] * s[:, :, None]                       # rango 1 por bin
    N = (rng.standard_normal((K, T, M)) + 1j * rng.standard_normal((K, T, M)))
    B = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    N = np.einsum("mn,ktn->ktm", B, N)

    ev = eval_frame_indices(T, 5, start_frame=20)
    Ps, Pn = snapshot_scms_oracle(S, N, ev, alpha=0.98)
    refs = oracle_references(Ps, Pn, REF, snr_floor_db=-60.0)

    W, _ = souden_weights(Ps, Pn, REF, mu=0.0, min_loading=1e-10)
    out = weight_loss(W, Ps, Pn, refs, eta=1.0)

    assert np.nanmax(out["L_sinr"]) < 1e-4, "el oracle debe alcanzar SINR_max"
    # y ademas es distortionless respecto de la RTF oracle
    assert np.nanmax(out["L_dist"]) < 1e-4


def test_invariancias_de_la_loss():
    """
    L_sinr no ve la escala de w (es la invariancia de Souden: escalar Phi_NN o
    Phi_XX no cambia el filtro). L_dist si la ve: es el termino que la fija, y
    por eso es el que se corresponde con PESQ.

    Se usa target de RANGO 1 para que el punto de partida sea EXACTAMENTE
    distortionless (L_dist = 0); cualquier reescalado tiene que empeorarlo.
    """
    rng = np.random.default_rng(11)
    a = rng.standard_normal((K, M)) + 1j * rng.standard_normal((K, M))
    s = rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))
    S = a[:, None, :] * s[:, :, None]
    N = rng.standard_normal((K, T, M)) + 1j * rng.standard_normal((K, T, M))
    ev = eval_frame_indices(T, 4, start_frame=20)
    Ps, Pn = snapshot_scms_oracle(S, N, ev, alpha=0.98)
    refs = oracle_references(Ps, Pn, REF, snr_floor_db=-60.0)

    W, _ = souden_weights(Ps, Pn, REF)
    base = weight_loss(W, Ps, Pn, refs)
    scaled = weight_loss(2.7 * W, Ps, Pn, refs)

    np.testing.assert_allclose(base["L_sinr"], scaled["L_sinr"], rtol=1e-9, atol=1e-12)
    assert np.nanmax(base["L_dist"]) < 1e-6
    assert np.nanmin(scaled["L_dist"]) > 1.0


def test_bandas_cubren_todos_los_bins():
    freqs = np.linspace(0, 8000, 257)
    _, band_of_bin, bands = make_bands(freqs, n_bands=20, f_min=60.0, f_max=7000.0)
    assert band_of_bin.shape == (257,)
    assert sum(b.size for b in bands) == 257            # particion exacta
    assert np.array_equal(np.sort(np.concatenate([b for b in bands if b.size])),
                          np.arange(257))

    rows = [{"_bins": b, "nu": 0.5, "gamma": 0.2} for b in bands if b.size]
    nu, gam = bands_to_bin_params(rows, 257)
    assert np.allclose(nu, 0.5) and np.allclose(gam, 0.2)


# =====================================================================
# El core CALIBRADO que corre en produccion vs el banco
# =====================================================================
def test_core_calibrado_reduce_a_los_cores_existentes(data):
    """
    El core nuevo no es una variante mas: es el mapa que contiene a los dos que
    ya estaban. nu=0 -> _fixed ; nu=1 -> _subtract. Si esto se rompe, el
    benchmark estaria comparando contra otra cosa.
    """
    from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask_calibrated
    X, ms, mn = data

    _, W_fixed = MVDR_Souden_recursive_mask_fixed(
        X, ms, mn, min_loading=1e-2, alpha=0.97, save_weights=True, ref_mic_idx=REF)
    _, W_cal0 = MVDR_Souden_recursive_mask_calibrated(
        X, ms, mn, nu=0.0, gamma=0.0, min_loading=1e-2, alpha=0.97,
        save_weights=True, ref_mic_idx=REF)
    np.testing.assert_allclose(W_cal0, W_fixed, rtol=1e-8, atol=1e-12)

    _, W_sub = MVDR_Souden_recursive_mask_subtract(
        X, ms, mn, min_loading=1e-9, alpha=0.97, mu=0.0, lambda_floor=1e-3,
        psd_project=True, save_weights=True, ref_mic_idx=REF)
    _, W_cal1 = MVDR_Souden_recursive_mask_calibrated(
        X, ms, mn, nu=1.0, gamma=0.0, min_loading=1e-9, alpha=0.97, mu=0.0,
        lambda_floor=1e-3, psd_project=True, save_weights=True, ref_mic_idx=REF)
    np.testing.assert_allclose(W_cal1, W_sub, rtol=1e-8, atol=1e-12)


def test_core_calibrado_coincide_con_el_banco(data):
    """
    Consistencia BANCO <-> PRODUCCION con nu y gamma NO triviales (distintos por
    bin, shrinkage activo): los pesos del core que corre en el benchmark tienen
    que ser los mismos que los que el optimizador estuvo puntuando. Sin esto, los
    parametros ajustados no significan nada al enchufarlos.
    """
    from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask_calibrated
    X, ms, mn = data
    rng = np.random.default_rng(3)
    nu = rng.uniform(0.2, 1.6, size=K)
    gam = rng.uniform(0.0, 0.7, size=K)

    coords = np.stack([np.linspace(0, 0.26, M), np.zeros(M), np.zeros(M)], axis=1)
    freqs = np.linspace(0, 8000, K)
    G = diffuse_coherence(coords, freqs)

    ev = eval_frame_indices(T, 6, start_frame=10)
    Pxx, Pnn = snapshot_scms_masked(X, ms, mn, ev, alpha=0.97)
    W_bank, _ = parametric_weights(Pxx, Pnn, G, REF, nu=nu, gamma=gam, mu=0.0,
                                   min_loading=1e-9, lambda_floor=1e-3)

    _, W_core = MVDR_Souden_recursive_mask_calibrated(
        X, ms, mn, nu=nu, gamma=gam, Gamma_diff=G, min_loading=1e-9, alpha=0.97,
        mu=0.0, lambda_floor=1e-3, psd_project=True, save_weights=True,
        ref_mic_idx=REF)

    np.testing.assert_allclose(W_bank, W_core[:, ev, :], rtol=1e-8, atol=1e-12)


def test_core_calibrado_exige_geometria_si_hay_shrinkage(data):
    """gamma > 0 sin Gamma_diff tiene que fallar fuerte, no correr en silencio."""
    from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask_calibrated
    X, ms, mn = data
    with pytest.raises(ValueError, match="Gamma_diff"):
        MVDR_Souden_recursive_mask_calibrated(X, ms, mn, nu=1.0, gamma=0.3,
                                              Gamma_diff=None, ref_mic_idx=REF)


# =====================================================================
# El regimen odds-ratio de la rama de ruido
# =====================================================================
def test_mask_ruido_es_un_odds_ratio_RECORTADO():
    """
    Caracteriza la forma que eligio el ajuste para la rama de ruido, y descarta
    la lectura ingenua.

    mask_n = sigma(a_n * logit(1-m) + b_n) crece como C*((1-m)/m)^{a_n} en el
    grueso de las celdas, PERO SATURA EN 1 para las de ruido mas confiable. Es
    tentador decir "como Phi_NN es invariante a escala, esto ES un odds-ratio y
    b_n solo mueve una constante". ES FALSO: la saturacion es justamente lo que
    hace que funcione. Medido sobre MIRD (rt60=0.61, no visto), con la rama de
    voz en identidad:

        odds-ratio PURO  ((1-m)/m)^2      L = 4.042 dB
        sigmoide a_n=2, b_n=-8            L = 3.585 dB   <-- 0.46 dB mejor

    Solo el 2.5 % de las celdas cae en la zona saturada, pero son las de MAYOR
    peso: sin el recorte, un punado de celdas domina Phi_NN. b_n es un parametro
    real -- fija DONDE recorta -- no una direccion plana.
    """
    from beamforming.mask.scm_calibration import warp_mask
    rng = np.random.default_rng(5)
    m = np.clip(rng.beta(0.7, 2.0, size=(K, T)), 1e-4, 1 - 1e-4)
    X = (rng.standard_normal((K, T, M)) + 1j * rng.standard_normal((K, T, M)))
    ev = eval_frame_indices(T, 5, start_frame=15)
    a_n, b_n = 2.0, -8.0

    m_sig = warp_mask(1.0 - m, a_n, b_n)
    assert m_sig.max() <= 1.0                      # acotada: satura, no diverge
    assert (m_sig > 0.9).any(), "deberia haber celdas en la zona saturada"

    odds = (1.0 - m) / m
    m_pow = odds ** a_n
    m_pow = m_pow / m_pow.max()
    assert m_pow.max() / np.median(m_pow) > 1e3, "el odds puro tiene cola pesada"

    _, Phi_sig = snapshot_scms_masked(X, m, m_sig, ev, alpha=0.98)
    _, Phi_pow = snapshot_scms_masked(X, m, m_pow, ev, alpha=0.98)
    rel = np.abs(Phi_sig - Phi_pow) / (np.abs(Phi_pow) + 1e-12)
    # las dos formas NO dan la misma Phi_NN: el recorte cambia el estimador.
    assert np.median(rel) > 1e-2, (
        "si esto pasara a ser ~0, la sigmoide ya no estaria recortando y la "
        "lectura 'odds-ratio puro' seria correcta")


def test_wrapper_mcal_usa_la_misma_transformacion_que_el_banco():
    """
    El wrapper NM_MVDR_MCAL construye las mascaras con `masks_from_raw`, la
    MISMA funcion que puntuo el optimizador. Se verifica el contrato completo:
    mascara cruda -> warp -> pesos del core calibrado.
    """
    from beamforming.mask.scm_calibration import masks_from_raw
    from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask_calibrated
    rng = np.random.default_rng(9)
    m_raw = np.clip(rng.beta(0.7, 2.0, size=(K, T)), 1e-3, 1 - 1e-3)
    X = (rng.standard_normal((K, T, M)) + 1j * rng.standard_normal((K, T, M)))
    theta = (1.0, 0.0, 2.0, -8.0)

    ms, mn = masks_from_raw(m_raw, *theta)
    ev = eval_frame_indices(T, 5, start_frame=15)
    Pxx, Pnn = snapshot_scms_masked(X, ms, mn, ev, alpha=0.98)
    W_bank, _ = parametric_weights(Pxx, Pnn, None, REF, nu=1.0, gamma=0.0,
                                   min_loading=1e-9, lambda_floor=1e-3)

    _, W_core = MVDR_Souden_recursive_mask_calibrated(
        X, ms, mn, nu=1.0, gamma=0.0, min_loading=1e-9, alpha=0.98, mu=0.0,
        lambda_floor=1e-3, psd_project=True, save_weights=True, ref_mic_idx=REF)

    np.testing.assert_allclose(W_bank, W_core[:, ev, :], rtol=1e-8, atol=1e-12)
