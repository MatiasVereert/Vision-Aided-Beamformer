#!/usr/bin/env bash
# =============================================================================
# run_wpe_experiment.sh
# =============================================================================
# Runner reutilizable de intrusive_benchmark_real.py para ir variando parametros
# del WPE (online RLS vs block-online HW) sobre grabaciones REALES y guardar los
# WAVs + CSVs de cada corrida en una carpeta propia.
#
# Cada corrida escribe en:  results/<EXP_NAME>/<run>/  con:
#     mixture.wav, ref_mic_raw.wav, senal_ref.wav, senal_ref_wpe.wav,
#     <beamformers>.wav, intrusive_metrics.csv, diagnostics_real.csv
# y el stdout completo en  results/<EXP_NAME>/<run>.log
#
# USO
# ---
#   # corrida unica con los valores de abajo (o overrides por env):
#   ./run_wpe_experiment.sh mi_experimento
#
#   # variar parametros por env (cualquiera de los de la seccion CONFIG):
#   WPE_METHOD=block WPE_TAPS=10 WPE_BLOCK_REG=1e-2 ./run_wpe_experiment.sh block_t10_reg1e-2
#   SNR=0 REF_MIC=6 ./run_wpe_experiment.sh snr0
#
#   # modo MATRIZ: corre con/sin WPE x {online,block} (4 corridas, ref comun wpe):
#   MATRIX=1 ./run_wpe_experiment.sh con_sin_wpe
#
#   # pasar flags crudos extra al benchmark (se agregan al final):
#   ./run_wpe_experiment.sh prueba -- --wpe-block-mode sliding
#
# Requiere:  conda env tesis_beam (o setear CONDA_ENV).
# =============================================================================
set -euo pipefail

# ----------------------------------------------------------------------------
# CONFIG  (todo overrideable por variable de entorno: VAR=... ./run...)
# ----------------------------------------------------------------------------
REPO="${REPO:-/home/matias/Documents/Tesis/Vision-Aided-Beamformer}"
WAVS_DIR="${WAVS_DIR:-/home/matias/pdm_mic_interface/kria_app/capture/wavs_paso5}"
SENAL="${SENAL:-$WAVS_DIR/senal12.wav}"      # WAV multicanal SOLO target (voz)
RUIDO="${RUIDO:-$WAVS_DIR/ruido12.wav}"      # WAV multicanal SOLO ruido
CONDA_ENV="${CONDA_ENV:-tesis_beam}"

# --- Mezcla / evaluacion ---
SNR="${SNR:-5}"                # SNR objetivo del mixture [dB] en el mic de ref. 'nat' = niveles grabados.
REF_MIC="${REF_MIC:-}"         # indice del canal de ref (vacio => M//2). p.ej. 6 para 12 mics.
EVAL_START="${EVAL_START:-5}"  # segundos iniciales a descartar (convergencia)
EVAL_REF="${EVAL_REF:-wpe}"    # 'wpe' (ref comun dereverberada; head-to-head con/sin) | 'domain'

# --- WPE (comun) ---
USE_WPE="${USE_WPE:-1}"        # 1 => aplica WPE front-end; 0 => sin WPE (mezcla cruda)
WPE_METHOD="${WPE_METHOD:-block}"   # online | block
WPE_TAPS="${WPE_TAPS:-7}"
WPE_DELAY="${WPE_DELAY:-3}"
WPE_BITS="${WPE_BITS:-}"       # vacio => float; 24/20/18 => fixed-point (solo method=online)

# --- Block-online (solo si WPE_METHOD=block) ---
WPE_BLOCK_L="${WPE_BLOCK_L:-512}"
WPE_BLOCK_SHIFT="${WPE_BLOCK_SHIFT:-64}"
WPE_BLOCK_ITERS="${WPE_BLOCK_ITERS:-3}"
WPE_BLOCK_REG="${WPE_BLOCK_REG:-1e-6}"
WPE_BLOCK_SOLVER="${WPE_BLOCK_SOLVER:-cholesky}"   # cholesky | lu | cholesky_explicit
WPE_BLOCK_MODE="${WPE_BLOCK_MODE:-resolve}"        # resolve | sliding
WPE_BLOCK_GDELAY="${WPE_BLOCK_GDELAY:-0}"          # latencia pipeline HW [bloques]. 0=batch, 1=piso real

# ----------------------------------------------------------------------------
EXP_NAME="${1:-exp_$(date +%Y%m%d_%H%M%S)}"; shift || true
# todo lo que venga despues de '--' se pasa crudo al benchmark
EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then shift; EXTRA_ARGS=("$@"); fi

RESULTS_DIR="$REPO/tests/documentation_experiments/results/$EXP_NAME"
mkdir -p "$RESULTS_DIR"

# activar conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "$CONDA_ENV" 2>/dev/null || echo "[!] no pude activar conda '$CONDA_ENV' (sigo con el python actual)"

BENCH="$REPO/src/evaluation/intrusive_benchmark_real.py"

# ----------------------------------------------------------------------------
# _run <subdir> <use_wpe 0|1> <method> [extra flags...]
#   arma la linea de comando del benchmark y la ejecuta guardando en subdir.
# ----------------------------------------------------------------------------
_run() {
  local sub="$1"; local use_wpe="$2"; local method="$3"; shift 3
  local out="$RESULTS_DIR/$sub"
  mkdir -p "$out"
  local args=( "$SENAL" "$RUIDO" "$out"
               --snr "$SNR" --eval-start "$EVAL_START" --eval-ref "$EVAL_REF"
               --wpe-method "$method"
               --wpe-taps "$WPE_TAPS" --wpe-delay "$WPE_DELAY"
               --wpe-block-L "$WPE_BLOCK_L" --wpe-block-shift "$WPE_BLOCK_SHIFT"
               --wpe-block-iters "$WPE_BLOCK_ITERS" --wpe-block-reg "$WPE_BLOCK_REG"
               --wpe-block-solver "$WPE_BLOCK_SOLVER" --wpe-block-mode "$WPE_BLOCK_MODE"
               --wpe-block-g-delay "$WPE_BLOCK_GDELAY" )
  [[ -n "$REF_MIC" ]] && args+=( --ref-mic "$REF_MIC" )
  [[ "$use_wpe" == "1" ]] && args+=( --wpe )
  [[ -n "$WPE_BITS" ]] && args+=( --wpe-bits "$WPE_BITS" )
  args+=( "$@" "${EXTRA_ARGS[@]}" )

  echo "=========================================================================="
  echo ">>> [$sub] python intrusive_benchmark_real.py ${args[*]}"
  echo "=========================================================================="
  ( cd "$REPO" && python "$BENCH" "${args[@]}" ) 2>&1 | tee "$RESULTS_DIR/$sub.log"
  echo "<<< [$sub] listo -> $out"
}

# ----------------------------------------------------------------------------
if [[ "${MATRIX:-0}" == "1" ]]; then
  # con/sin WPE x {online, block}, todas con la misma ref comun (--eval-ref wpe).
  # NOTA: en 'wpe' la ref lleva el G de cada metodo -> comparar Δ (con-sin) DENTRO
  # de cada par, no columnas absolutas entre online y block.
  echo "[*] MODO MATRIZ: online_nowpe, online_wpe, block_nowpe, block_wpe -> $RESULTS_DIR"
  _run online_nowpe 0 online
  _run online_wpe   1 online
  _run block_nowpe  0 block
  _run block_wpe    1 block
else
  # corrida unica con la CONFIG de arriba.
  sub="${WPE_METHOD}_$([[ "$USE_WPE" == "1" ]] && echo wpe || echo nowpe)"
  _run "$sub" "$USE_WPE" "$WPE_METHOD"
fi

echo
echo "[*] TODO en: $RESULTS_DIR"
echo "    WAVs por corrida + intrusive_metrics.csv + diagnostics_real.csv (no-intrusivas) + <run>.log"
