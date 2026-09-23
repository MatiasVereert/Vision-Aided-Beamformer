"""
ofb_lowfreq_bands.py
====================
LA RESPUESTA AL TARGET POR BANDAS -- el instrumento que PESQ no tiene.

`souden_mvdr.py` documenta el COLAPSO DE ESCALA EN GRAVES de los cores que NO
restan: donde la mascara no encuentra voz, Phi_XX -> Phi_NN, el filtro degenera
a w = u/M y la banda sale -20log10(M) dB (M=12 -> -21.6 dB) con ganancia de
arreglo NULA. P.862 no evalua debajo de ~300 Hz, asi que el harness puede dar
"empate" mientras el sistema tira los graves a la basura.

QUE MIDE
--------
Se corre el lazo sobre la MEZCLA (los pesos se adaptan a la escena real), y
despues esos MISMOS pesos, frame a frame, se aplican a la senal de TARGET SOLO:

    G(k,t) = w(k,t)^H s(k,t)          <- la respuesta al target, sin ruido

y se reporta, por banda, 10log10( sum|G|^2 / sum|s_ref|^2 ): cuantos dB le pone
o le saca el beamformer al target respecto del mic de referencia. Distortionless
= 0 dB. El colapso se ve como una caida de ~-20log10(M) en la banda grave.

USO
---
    conda activate tesis_beam
    python tests/ofb_lowfreq_bands.py --variants psd nosub
"""

import os
import sys
import argparse

import numpy as np
import scipy.signal as sig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for d in (os.path.join(PROJECT_ROOT, "src"), os.path.join(PROJECT_ROOT, "tests")):
    if d not in sys.path:
        sys.path.insert(0, d)

from propagation.mird_loader import MirdDatasetProvider                  # noqa: E402
from ofb_refactor_equivalence import load_mono, FS, MIRD_ROOT, SPACING, \
    MODEL_1, SOURCE_WAV, INTERF_WAV                                      # noqa: E402
from ofb_psd_variants import make_wrapper                                # noqa: E402

BANDS = [(0, 130), (130, 300), (300, 1000), (1000, 3400), (3400, 8000)]


def scene_with_components(provider, rt60, angle_t, angle_i, isir_db, dur,
                          snr_db=60.0, spacing=SPACING):
    """Igual que `build_scene`, pero devuelve TAMBIEN el target multicanal solo."""
    n = int(dur * FS)
    rng = np.random.default_rng(0)

    def rir(angle):
        h = np.asarray(provider.load_rir(rt60, spacing, 1.0, angle), dtype=np.float64)
        fs_nat = provider.get_current_fs()
        return sig.resample_poly(h, FS, fs_nat, axis=0) if fs_nat != FS else h

    h_t, h_i = rir(angle_t), rir(angle_i)
    M = h_t.shape[1]
    s, v = load_mono(SOURCE_WAV, n), load_mono(INTERF_WAV, n, offset=FS)
    tgt = np.stack([sig.fftconvolve(s, h_t[:, m])[:n] for m in range(M)])
    itf = np.stack([sig.fftconvolve(v, h_i[:, m])[:n] for m in range(M)])
    ref = M // 2
    p_t = np.mean(tgt[ref] ** 2)
    itf *= np.sqrt(p_t / (np.mean(itf[ref] ** 2) + 1e-20) * 10.0 ** (-isir_db / 10.0))
    noise = rng.standard_normal((M, n)) * np.sqrt(p_t * 10.0 ** (-snr_db / 10.0))
    cfg = {'fs': FS, 'stft_window': 512, 'stft_overlap': 384,
           'dtln_model_path': MODEL_1, 'ref_mic_idx': ref, 'isir_db': float(isir_db)}
    return tgt + itf + noise, tgt, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=["psd", "nosub"])
    ap.add_argument("--rt60", type=float, nargs="+", default=[0.610])
    ap.add_argument("--isir", type=float, nargs="+", default=[0.0])
    ap.add_argument("--spacing", default=SPACING)
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--smooth", type=float, default=0.2)
    args = ap.parse_args()

    prov = MirdDatasetProvider(root_dir=MIRD_ROOT)
    print(f"\nrespuesta al TARGET respecto del mic de referencia, por banda [dB]")
    print("(0 dB = distortionless; el colapso de escala se ve como una caida de "
          "-20log10(M) en la banda grave)\n")
    hdr = (f"{'rt60':>6} {'iSIR':>5} {'variante':<8}" +
           "".join(f"{f'{lo}-{hi}':>11}" for lo, hi in BANDS))
    print(hdr); print("-" * len(hdr))
    for rt60 in args.rt60:
        for isir in args.isir:
            mic, tgt, cfg = scene_with_components(prov, rt60, 0, 45, isir, args.dur,
                                                  spacing=args.spacing)
            ref = cfg['ref_mic_idx']
            freqs, _, Zt = sig.stft(tgt, fs=FS, window='boxcar', nperseg=512,
                                    noverlap=384, nfft=512)
            S = np.transpose(Zt, (1, 2, 0))              # (K, T, M) target SOLO
            bands = [(lo, hi, (freqs >= lo) & (freqs < hi)) for lo, hi in BANDS]
            for v in args.variants:
                _, W = make_wrapper(v, 4)(smooth=args.smooth,
                                          return_weights=True).process(mic, cfg)
                G = np.einsum("ktm,ktm->kt", W.conj(), S)    # w^H s, frame a frame
                Pg, Pr = np.abs(G) ** 2, np.abs(S[:, :, ref]) ** 2
                row = [10 * np.log10((Pg[m].sum() + 1e-30) / (Pr[m].sum() + 1e-30))
                       for _, _, m in bands]
                print(f"{rt60:6.3f} {isir:5.0f} {v:<8}" +
                      "".join(f"{r:11.2f}" for r in row), flush=True)


if __name__ == "__main__":
    main()
