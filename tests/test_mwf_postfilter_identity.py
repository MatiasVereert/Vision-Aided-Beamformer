"""
Regresion del core MWF (beamforming/MWF/wiener_postfilter.py).

La propiedad que lo hace seguro de adoptar: con w_gmin_db=0 dB la etapa Wiener es la
IDENTIDAD, asi que el core tiene que reproducir BIT A BIT
`MVDR_Souden_recursive_mask_specsub_base` (el post-filtro actual). Si algun cambio en
el Wiener rompe eso, dejo de ser una extension estricta del PF y pasa a ser otro
algoritmo. Las protecciones de STOI (gmin_mask / smooth_f / smooth_t) tienen que estar
OFF por defecto: tambien se verifica que no alteren el camino base.
"""

import numpy as np

from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask_specsub_base
from beamforming.MWF.wiener_postfilter import (
    MVDR_Souden_mask_specsub_MWF, wiener_dd_gain,
)


def _fixture(K=129, T=80, M=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(K, T, M)) + 1j * rng.normal(size=(K, T, M))
    mask_s = rng.uniform(0.1, 1.0, size=(K, T))
    return X, mask_s, 1.0 - mask_s, mask_s ** 0.25


def test_gmin_0db_reproduce_el_pf_base():
    X, ms, mn, msoft = _fixture()
    kw = dict(min_loading=1e-9, alpha=0.99, smooth=0.5, ref_mic_idx=4)
    ref = MVDR_Souden_recursive_mask_specsub_base(X, ms, mn, msoft, **kw)
    got = MVDR_Souden_mask_specsub_MWF(X, ms, mn, msoft, w_gmin_db=0.0, **kw)
    assert np.array_equal(got, ref)


def test_protecciones_off_por_defecto():
    """gmin_mask/smooth_f/smooth_t desactivados no deben tocar la ganancia."""
    X, ms, mn, msoft = _fixture(seed=1)
    kw = dict(min_loading=1e-9, alpha=0.99, smooth=0.5, w_gmin_db=-6.0, ref_mic_idx=4)
    base = MVDR_Souden_mask_specsub_MWF(X, ms, mn, msoft, **kw)
    explicit = MVDR_Souden_mask_specsub_MWF(X, ms, mn, msoft, w_gmin_mask=False,
                                            w_smooth_f=0, w_smooth_t=0.0, **kw)
    assert np.array_equal(base, explicit)


def test_piso_de_ganancia_se_respeta():
    """Ninguna ganancia puede caer por debajo del piso pedido."""
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(64, 50)) + 1j * rng.normal(size=(64, 50))
    mn = rng.uniform(0.0, 1.0, size=(64, 50))
    for g_db in (-3.0, -6.0, -12.0):
        _, G = wiener_dd_gain(Y, mask_n=mn, g_min_db=g_db, smooth_t=0.0)
        assert G.min() >= 10.0 ** (g_db / 20.0) - 1e-12
        assert G.max() <= 1.0 + 1e-12


def test_gmin_mask_protege_los_bins_de_habla():
    """Con gmin_mask, un bin con mask_s=1 no puede ser atenuado."""
    rng = np.random.default_rng(3)
    K, T = 32, 40
    Y = rng.normal(size=(K, T)) + 1j * rng.normal(size=(K, T))
    mn = np.ones((K, T)) * 0.5
    ms = np.zeros((K, T))
    ms[:8, :] = 1.0                      # los 8 primeros bins son 100% habla
    _, G = wiener_dd_gain(Y, mask_n=mn, mask_s=ms, gmin_mask=True, g_min_db=-6.0)
    assert np.allclose(G[:8, :], 1.0)    # piso = 1 -> sin atenuacion
    assert G[8:, :].min() < 1.0          # el resto si se atenua
