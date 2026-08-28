"""
coupled_wrappers.py
===================
Beamformer + DTLN ACOPLADOS EN UNA SOLA STFT.

MOTIVACION
----------
En el benchmark actual la cadena ganadora (NM-MVDR o DTLN_MB_MVDR_SOUDEN_BAN,
con DTLN completo a la salida) usa DOS enmarcados distintos:

  mic -> STFT(hamming, 512/128) -> MVDR -> iSTFT -> senal ->
         framing DTLN (RECTANGULAR, 512/128) -> DTLN -> OLA -> salida

El DTLN (tanto `get_dtln_masks` como `apply_dtln_post_tflite_realtime`) NO aplica
ventana: llena `in_buffer` por corrimiento y hace `np.fft.rfft` directo, y
reconstruye por overlap-add de suma simple. O sea: ventana RECTANGULAR, 512/128.

Para implementar la cadena en hardware con UN SOLO bloque STFT/iSTFT hay que
usar el MISMO enmarcado en las dos etapas. Este modulo hace exactamente eso:
la STFT se calcula una vez, el MVDR filtra en ese dominio y el DTLN se ejecuta
SOBRE ESOS MISMOS FRAMES (sin iSTFT intermedia ni re-analisis).

QUE SE PUEDE COMPARAR CON ESTO
------------------------------
  1) Efecto de la ventana en la SALIDA DEL BEAMFORMER (resultado intermedio):
     NM_MVDR(win_type='hamming') vs NM_MVDR(win_type='rect').
  2) Efecto de la ventana en la SALIDA FINAL acoplada:
     BF_DTLN_COUPLED(win_type='hamming') vs BF_DTLN_COUPLED(win_type='rect').
  3) Acoplado vs desacoplado: la columna `dtln_post_*` del benchmark (Node 6)
     es la cascada DESACOPLADA (iSTFT + re-framing rectangular nativo del DTLN)
     del mismo beamformer, asi que sale gratis en la misma corrida.

DETALLES NUMERICOS QUE IMPORTAN
-------------------------------
* Escalado: `scipy.signal.stft` divide el espectro por `win.sum()`. Para darle al
  DTLN la misma magnitud que ve en su camino nativo hay que multiplicar por
  `win.sum()` antes de invocar el modelo.
* Normalizacion de pico: `apply_dtln_post_tflite_realtime` normaliza la senal
  completa por su pico antes de enmarcar y restaura la escala al final. Aca se
  hace lo mismo usando el pico de la salida del beamformer (es lineal, da igual
  normalizar antes o despues de la STFT).
* Overlap-add: el DTLN suma los bloques SIN normalizar (con rectangular, la
  envolvente de suma vale L/H = 4 y la red esta entrenada con ESA convencion).
  Con una ventana distinta la envolvente cambia, asi que aca se divide por
  env_w(n) / env_rect(n). Con ventana rectangular ese cociente es EXACTAMENTE 1
  en todas las muestras (bordes incluidos) -> el camino rect reproduce bit a bit
  la convencion nativa del DTLN.
* Latencia: el lazo nativo del DTLN escribe la salida del bloque `idx` en
  `out[idx*H : idx*H+H]`, lo que introduce un retardo puro de L-H = 384 muestras
  (24 ms). Se respeta esa convencion para que el acoplado sea comparable con el
  desacoplado. (Las metricas del benchmark realinean por correlacion cruzada.)
* Indexado de frames: el frame `t` de `scipy.signal.stft` (boundary='zeros',
  padded=True) coincide EXACTAMENTE con el bloque `t+1` del buffer del DTLN
  (verificado numericamente, error < 1e-15 con boxcar). El primer bloque nativo
  del DTLN (idx=0) queda fuera del barrido de frames de scipy.
"""

import numpy as np
import scipy.signal as sig

from beamforming.mask.dtln_masks import get_dtln_masks, get_dtln_masks_sharpen
from beamforming.mask.souden_mvdr import (
    MVDR_Souden_recursive_mask,
    MVDR_Souden_recursive_mask_BAN,
)

from evaluation.bf_wrappers import resolve_stft_window


def _load_interpreter(path):
    """Interprete TFLite (mismo backend que usa dtln_masks.py)."""
    from ai_edge_litert.interpreter import Interpreter
    itp = Interpreter(model_path=str(path))
    itp.allocate_tensors()
    return itp


def _derive_model2_path(model1_path):
    """model_quant_1.tflite -> model_quant_2.tflite."""
    p = str(model1_path)
    if "_1.tflite" in p:
        return p.replace("_1.tflite", "_2.tflite")
    raise ValueError(
        f"No pude derivar el path del 2do modelo DTLN desde {p!r}. "
        "Pasalo explicito con model2_path=..."
    )


class BF_DTLN_COUPLED:
    """
    MVDR mask-based (Souden) + DTLN completo, ACOPLADOS en la misma STFT.

    Parametros
    ----------
    core : {'souden', 'souden_ban'}
        'souden'     -> MVDR_Souden_recursive_mask   (el core de NM_MVDR)
        'souden_ban' -> MVDR_Souden_recursive_mask_BAN (el de DTLN_MB_MVDR_SOUDEN_BAN)
    win_type : None | 'hamming' | 'hann' | 'rect' | 'sqrt_hann' | array
        Ventana de analisis/sintesis. None -> lee scene_config['stft_win_type'],
        y si tampoco esta -> 'hamming' (default historico de los wrappers).
        'rect' es la unica que hace el acople EXACTO con el enmarcado del DTLN.
    return_stage : {'dtln', 'bf'}
        'dtln' (default) devuelve la salida final de la cadena acoplada.
        'bf' devuelve solo la salida del beamformer (sirve de control: con la
        misma ventana debe coincidir con NM_MVDR / DTLN_MB_MVDR_SOUDEN_BAN).
    dtln_stage : bool
        False saltea el DTLN (equivalente a return_stage='bf'), util para medir
        el costo de la etapa neuronal sin cambiar de clase.
    """

    def __init__(self, core='souden', nperseg=512, noverlap=384, min_loading=1e-6,
                 alpha=0.99, sharpen_exp=4.0, win_type=None,
                 model1_path=None, model2_path=None, return_stage='dtln',
                 mask_from_stft=False):
        if core not in ('souden', 'souden_ban'):
            raise ValueError(f"core={core!r} no soportado ('souden' | 'souden_ban').")
        if return_stage not in ('dtln', 'bf'):
            raise ValueError(f"return_stage={return_stage!r} ('dtln' | 'bf').")

        self.core = core
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha
        self.sharpen_exp = sharpen_exp
        self.win_type = win_type
        self.return_stage = return_stage
        # True -> la mascara tambien sale de la STFT compartida (una sola FFT en
        # todo el sistema, y sin el corrimiento de un frame). False (default) ->
        # se usa get_dtln_masks*, exactamente como NM_MVDR / SOUDEN_BAN.
        self.mask_from_stft = mask_from_stft

        self._model1_path = model1_path
        self._model2_path = model2_path
        self._itp1 = None   # cache: los interpretes se crean una sola vez
        self._itp2 = None
        self._itp_key = None

    # ------------------------------------------------------------------
    # Mascaras DTLN calculadas SOBRE LA STFT COMPARTIDA (acople total)
    # ------------------------------------------------------------------
    def _masks_from_stft(self, X_ref_spec, win_sum, peak, sharpen_exp):
        """
        Corre el modelo 1 del DTLN sobre los frames de la STFT compartida en vez
        de re-enmarcar la senal (lo que hace get_dtln_masks). Con ventana
        rectangular los frames son LOS MISMOS, asi que el resultado es el mismo
        salvo dos detalles que este camino ademas ARREGLA:

          * se ahorra una FFT (el hardware calcula la STFT UNA sola vez);
          * desaparece el corrimiento de un frame entre mascara y espectro que
            tiene el camino por defecto (el frame t de scipy es el bloque t+1 del
            buffer del DTLN, pero los wrappers los emparejan por indice).

        X_ref_spec : (K, T) espectro scipy del canal de referencia.
        Devuelve (mask_s, mask_n) con la MISMA post-procesado que dtln_masks.py
        (stretch de contraste + sharpening con exponente `sharpen_exp`).
        """
        itp1 = self._itp1
        in1, out1 = itp1.get_input_details(), itp1.get_output_details()
        states_1 = np.zeros(in1[1]['shape'], dtype=np.float32)

        K, T = X_ref_spec.shape
        ch_mask = np.zeros((K, T), dtype=np.float32)
        # get_dtln_masks normaliza el canal por su pico antes de enmarcar
        spec = X_ref_spec * (win_sum / (peak if peak > 0 else 1.0))

        for t in range(T):
            mag = np.ascontiguousarray(
                np.reshape(np.abs(spec[:, t]), (1, 1, -1)), dtype=np.float32)
            itp1.set_tensor(in1[1]['index'], states_1)
            itp1.set_tensor(in1[0]['index'], mag)
            itp1.invoke()
            ch_mask[:, t] = np.squeeze(itp1.get_tensor(out1[0]['index']).copy())
            states_1 = itp1.get_tensor(out1[1]['index']).copy()

        # Mismo post-procesado que beamforming/mask/dtln_masks.py
        m = ch_mask
        m = (m - np.min(m)) / (np.max(m) - np.min(m) + 1e-12)
        mask_s = m ** sharpen_exp
        mask_n = (1.0 - m) ** sharpen_exp
        return mask_s, mask_n

    # ------------------------------------------------------------------
    # DTLN sobre frames YA calculados (el acople propiamente dicho)
    # ------------------------------------------------------------------
    def _dtln_on_stft(self, Y_stft, win, hop, n_out, scale_factor):
        """
        Y_stft : (K, T) espectro de la salida del beamformer, con el escalado de
                 scipy YA deshecho y la senal YA normalizada por pico.
        win    : array de la ventana de analisis (L,)
        Devuelve la senal de tiempo (n_out,) reconstruida por OLA.
        """
        L = win.shape[0]
        K, T = Y_stft.shape

        itp1, itp2 = self._itp1, self._itp2
        in1 = itp1.get_input_details()
        out1 = itp1.get_output_details()
        in2 = itp2.get_input_details()
        out2 = itp2.get_output_details()

        # Estados LSTM SIEMPRE en cero al arrancar cada escena
        states_1 = np.zeros(in1[1]['shape'], dtype=np.float32)
        states_2 = np.zeros(in2[1]['shape'], dtype=np.float32)

        out_audio = np.zeros(n_out, dtype=np.float32)
        out_buffer = np.zeros(L, dtype=np.float32)

        # Envolventes de OLA: la de la ventana usada y la rectangular (referencia
        # nativa del DTLN). El cociente normaliza sin tocar el caso rect.
        env_audio = np.zeros(n_out, dtype=np.float64)
        env_rect_audio = np.zeros(n_out, dtype=np.float64)
        env_buffer = np.zeros(L, dtype=np.float64)
        env_rect_buffer = np.zeros(L, dtype=np.float64)
        ones = np.ones(L, dtype=np.float64)

        for t in range(T):
            spec = Y_stft[:, t]
            mag = np.ascontiguousarray(
                np.reshape(np.abs(spec), (1, 1, -1)), dtype=np.float32)
            phase = np.angle(spec)

            # --- Modelo 1: mascara en magnitud ---
            itp1.set_tensor(in1[1]['index'], states_1)
            itp1.set_tensor(in1[0]['index'], mag)
            itp1.invoke()
            out_mask = itp1.get_tensor(out1[0]['index']).copy()
            states_1 = itp1.get_tensor(out1[1]['index']).copy()

            # --- Vuelta al tiempo (bloque) ---
            estimated_complex = mag * out_mask * np.exp(1j * phase)
            estimated_block = np.fft.irfft(estimated_complex, n=L)
            estimated_block = np.ascontiguousarray(
                np.reshape(estimated_block, (1, 1, -1)), dtype=np.float32)

            # --- Modelo 2: nucleo de separacion en el dominio aprendido ---
            itp2.set_tensor(in2[1]['index'], states_2)
            itp2.set_tensor(in2[0]['index'], estimated_block)
            itp2.invoke()
            out_block = itp2.get_tensor(out2[0]['index']).copy()
            states_2 = itp2.get_tensor(out2[1]['index']).copy()

            # --- Overlap-add (convencion nativa del DTLN: suma simple) ---
            out_buffer[:-hop] = out_buffer[hop:]
            out_buffer[-hop:] = 0.0
            out_buffer += np.squeeze(out_block)

            env_buffer[:-hop] = env_buffer[hop:]
            env_buffer[-hop:] = 0.0
            env_buffer += win

            env_rect_buffer[:-hop] = env_rect_buffer[hop:]
            env_rect_buffer[-hop:] = 0.0
            env_rect_buffer += ones

            # El frame t de scipy es el bloque t+1 del buffer nativo del DTLN:
            # se respeta el mismo punto de escritura (retardo puro de L-hop).
            start = (t + 1) * hop
            stop = min(start + hop, n_out)
            if start >= n_out:
                break
            n = stop - start
            out_audio[start:stop] = out_buffer[:n]
            env_audio[start:stop] = env_buffer[:n]
            env_rect_audio[start:stop] = env_rect_buffer[:n]

        # Normalizacion de ventana: 1.0 exacto si win es rectangular
        norm = np.ones_like(env_audio)
        valid = env_rect_audio > 0
        norm[valid] = env_audio[valid] / env_rect_audio[valid]
        norm[np.abs(norm) < 1e-8] = 1.0

        return (out_audio / norm) * scale_factor

    # ------------------------------------------------------------------
    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        fs = scene_config['fs']
        model1_path = self._model1_path or scene_config.get(
            'dtln_model_path', r'dnn_denoise/models/model_quant_1.tflite')
        model2_path = self._model2_path or scene_config.get(
            'dtln_model2_path') or _derive_model2_path(model1_path)

        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_dyn = nperseg_dyn - noverlap_dyn
        if nperseg_dyn != 512 or hop_dyn != 128:
            print(f"[Warning]: Window length ({nperseg_dyn}) and hop length ({hop_dyn}) "
                  f"should ideally match DTLN training (512/128).")

        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))
        sharpen_exp = scene_config.get('dtln_sharpen_exp', self.sharpen_exp)

        # --- Ventana ---
        win_spec = resolve_stft_window(scene_config, self.win_type, nperseg_dyn)
        win = (sig.get_window(win_spec, nperseg_dyn, fftbins=True)
               if isinstance(win_spec, str) else np.asarray(win_spec, dtype=np.float64))

        # --- 1. STFT unica (la que compartirian BF y DTLN en hardware) ---
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window=win_spec,
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn)
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # --- 2. Mascaras DTLN sobre el mic de referencia ---
        if self.mask_from_stft:
            self._ensure_interpreters(model1_path, model2_path)
            peak_ref = float(np.max(np.abs(mic_signals[ref_mic_idx])))
            mask_s, mask_n = self._masks_from_stft(
                X_stft[:, :, ref_mic_idx], float(np.sum(win)), peak_ref,
                sharpen_exp if self.core == 'souden' else 4.0)
        elif self.core == 'souden':
            mask_s, mask_n = get_dtln_masks_sharpen(
                mic_signals, ref_mic_idx, model1_path,
                block_len=nperseg_dyn, block_shift=hop_dyn, sharpen_exp=sharpen_exp)
        else:
            mask_s, mask_n = get_dtln_masks(
                mic_signals, ref_mic_idx, model1_path,
                block_len=nperseg_dyn, block_shift=hop_dyn)

        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        # --- 3. Core del beamformer (identico al de los wrappers de referencia) ---
        if self.core == 'souden':
            Y_stft, weights = MVDR_Souden_recursive_mask(
                X_stft, mask_s, mask_n, min_loading=self.min_loading,
                save_weights=True, alpha=self.alpha, ref_mic_idx=ref_mic_idx)
        else:
            Y_stft, weights = MVDR_Souden_recursive_mask_BAN(
                X_stft, mask_s, mask_n, min_loading=self.min_loading,
                save_weights=True, ref_mic_idx=ref_mic_idx)

        original_length = mic_signals.shape[1]

        # Salida del beamformer en el tiempo: se necesita igual para fijar la
        # escala del DTLN (normalizacion por pico) y es el resultado intermedio.
        _, y_bf = sig.istft(
            Y_stft, fs=fs, window=win_spec,
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn)
        y_bf = y_bf[:original_length]

        if self.return_stage == 'bf':
            return y_bf, weights

        # --- 4. DTLN EN EL MISMO DOMINIO STFT ---
        self._ensure_interpreters(model1_path, model2_path)

        peak = float(np.max(np.abs(y_bf)))
        scale_factor = peak if peak > 1e-6 else 1.0

        # scipy escala por 1/win.sum(): se deshace para entregarle al DTLN la
        # misma magnitud que ve en su camino nativo.
        Y_frames = (Y_stft * float(np.sum(win))) / scale_factor

        y_out = self._dtln_on_stft(
            Y_frames, win=win, hop=hop_dyn,
            n_out=original_length, scale_factor=scale_factor)

        return y_out[:original_length], weights

    # ------------------------------------------------------------------
    def _ensure_interpreters(self, model1_path, model2_path):
        key = (str(model1_path), str(model2_path))
        if self._itp1 is None or self._itp_key != key:
            self._itp1 = _load_interpreter(model1_path)
            self._itp2 = _load_interpreter(model2_path)
            self._itp_key = key


class NM_MVDR_DTLN(BF_DTLN_COUPLED):
    """NM_MVDR + DTLN acoplados (core base de Souden). Alias comodo."""

    def __init__(self, **kwargs):
        kwargs.setdefault('core', 'souden')
        super().__init__(**kwargs)


class SOUDEN_BAN_DTLN(BF_DTLN_COUPLED):
    """DTLN_MB_MVDR_SOUDEN_BAN + DTLN acoplados. Alias comodo."""

    def __init__(self, **kwargs):
        kwargs.setdefault('core', 'souden_ban')
        kwargs.pop('alpha', None)
        super().__init__(**kwargs)
