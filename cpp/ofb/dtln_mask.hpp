#pragma once 
//std
#include <array>
#include <cmath> 
#include <complex>
#include <stdexcept>
//fft
#include "kiss_fft.h"
#include "kiss_fftr.h" 
//
#include <ofb/config.hpp>
#include <ofb/types.hpp>
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>


namespace ofb{

// La STFT del beamformer y la entrada de la red tienen que ser el mismo numero.
// Hoy coinciden por casualidad (K = kFft/2+1 = 257 = k_bins); si algun dia tocas
// kFft, esto avisa al compilar en vez de fallar en silencio.
static_assert(K == k_bins, "la STFT del beamformer y la entrada del DTLN no coinciden");


// =====================================================================
// Lo que COMPARTEN las dos instancias de la red. Medido:
//   el TfLiteModel          ahorra  ~95 kB  (el interprete NO copia los pesos:
//                                            apunta al buffer del flatbuffer)
//   el weights_cache        ahorra ~530 kB  (XNNPACK REEMPAQUETA los pesos a un
//                                            formato SIMD, ~540 kB POR interprete)
// El segundo es el que importa en el A53, con 1 MB de L2.
// =====================================================================
class DtlnModel {
  public:
    explicit DtlnModel(const char* model_path){
      model_ = TfLiteModelCreateFromFile(model_path);
      if (model_ == nullptr)
        throw std::runtime_error("no se pudo abrir el .tflite");
      cache_ = TfLiteXNNPackDelegateWeightsCacheCreate();
      if (cache_ == nullptr)
        throw std::runtime_error("no se pudo crear el weights cache de XNNPACK");
    }

    ~DtlnModel(){
      // OJO: esto tiene que correr DESPUES de que mueran los DtlnMask, porque
      // sus delegates apuntan al cache. Se consigue solo declarando el DtlnModel
      // ANTES que los DtlnMask en la clase que los contiene: los miembros se
      // destruyen en orden INVERSO al de declaracion.
      TfLiteXNNPackDelegateWeightsCacheDelete(cache_);
      TfLiteModelDelete(model_);
    }

    DtlnModel(const DtlnModel&)            = delete;
    DtlnModel& operator=(const DtlnModel&) = delete;

    TfLiteModel*                        model() const { return model_; }
    TfLiteXNNPackDelegateWeightsCache*  cache() const { return cache_; }

    // Se llama UNA vez, DESPUES de construir todos los DtlnMask que comparten
    // este cache. Antes de eso el cache todavia esta recibiendo pesos.
    void finalize(){
      if (!TfLiteXNNPackDelegateWeightsCacheFinalizeHard(cache_))
        throw std::runtime_error("FinalizeHard del weights cache fallo");
    }

  private:
    TfLiteModel*                       model_ = nullptr;
    TfLiteXNNPackDelegateWeightsCache* cache_ = nullptr;
};


class DtlnMask{
 private:
      // interprete (el modelo NO es miembro: lo comparte el DtlnModel)
      TfLiteInterpreter* interpreter_m0_ = nullptr;
      // delegate
      TfLiteDelegate* xnnpack_delegate_m0_ = nullptr;
      // state
      std::array<float, k_state> state_m0{};
      // Interpreter Tensors Ptrs
      TfLiteTensor*       h_m0_mag_in_    = nullptr;
      TfLiteTensor*       h_m0_state_in_  = nullptr;
      const TfLiteTensor* h_m0_mask_out_  = nullptr;
      const TfLiteTensor* h_m0_state_out_ = nullptr;
      //kissfft config
      kiss_fftr_cfg fftr_cfg_ = nullptr; 

  public: 

    // Recibe el modelo y el cache YA creados: no es dueno de ninguno de los dos,
    // y los dos tienen que sobrevivirlo.
    explicit DtlnMask( const DtlnModel& shared ){

      // create Options
      TfLiteInterpreterOptions* options_0 = TfLiteInterpreterOptionsCreate();

      // delegate model 0, con el cache COMPARTIDO
      TfLiteXNNPackDelegateOptions xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
      xnnpack_options.num_threads   = 1;
      xnnpack_options.weights_cache = shared.cache();
      xnnpack_delegate_m0_ = TfLiteXNNPackDelegateCreate( &xnnpack_options );
      TfLiteInterpreterOptionsAddDelegate( options_0, xnnpack_delegate_m0_);

      // create interpeter and delete options
      interpreter_m0_ = TfLiteInterpreterCreate(shared.model(), options_0);

      TfLiteInterpreterOptionsDelete(options_0);
      if (interpreter_m0_ == nullptr )
        throw std::runtime_error("TfLiteInterpreterCreate fallo");

      // Allocate
      if (TfLiteInterpreterAllocateTensors(interpreter_m0_) != kTfLiteOk )
        throw std::runtime_error("AllocateTensors fallo");

      h_m0_mag_in_    = TfLiteInterpreterGetInputTensor (interpreter_m0_, 0);
      h_m0_state_in_  = TfLiteInterpreterGetInputTensor (interpreter_m0_, 1);
      h_m0_mask_out_  = TfLiteInterpreterGetOutputTensor(interpreter_m0_, 0);
      h_m0_state_out_ = TfLiteInterpreterGetOutputTensor(interpreter_m0_, 1);

      // fft config: solo la DIRECTA. La inversa no hace falta: esta clase
      // produce una mascara, no resintetiza audio.
      fftr_cfg_ = kiss_fftr_alloc( block_len, 0, nullptr, nullptr) ;
      if (fftr_cfg_ == nullptr)
        throw std::runtime_error("kiss_fftr_alloc fallo");
    }

    ~DtlnMask(){
      // el interprete primero: usa el delegate 
      TfLiteInterpreterDelete(interpreter_m0_);
      TfLiteXNNPackDelegateDelete(xnnpack_delegate_m0_);
      kiss_fftr_free(fftr_cfg_);          // kiss_fftr_alloc hace malloc
    }

    // Tiene punteros crudos y destructor: copiarla seria doble liberacion.
    DtlnMask(const DtlnMask&)            = delete;
    DtlnMask& operator=(const DtlnMask&) = delete;

    Frame_mag process_mask( const Frame_real& audio_input);

  };

// Lanza si una llamada de tflite no devolvio kTfLiteOk.
inline void check(TfLiteStatus s, const char* what){
  if (s != kTfLiteOk) throw std::runtime_error(what);
}

inline Frame_mag DtlnMask::process_mask(const Frame_real& input_frame){

  std::array<kiss_fft_cpx, k_bins> bins{};   // tipo nativo de kissfft: sin reinterpret_cast
  Frame_mag  in_mag{};
  Frame_mag  mask{};

  // ---- FFT real: 512 reales -> 257 complejos. Igual que np.fft.rfft,
  //      sin normalizar en esta direccion.
  kiss_fftr(fftr_cfg_, input_frame.data(), bins.data());

  for (int k = 0; k < k_bins; ++k)
    in_mag[k] = std::sqrt(bins[k].r * bins[k].r + bins[k].i * bins[k].i);

  // ---- modelo 0: (magnitud, estado) -> (mascara, estado')
  check(TfLiteTensorCopyFromBuffer(h_m0_mag_in_,   in_mag.data(),   bytes_mag),
        "copia a la entrada de magnitud");
  check(TfLiteTensorCopyFromBuffer(h_m0_state_in_, state_m0.data(), bytes_state),
        "copia del estado a la entrada");

  check(TfLiteInterpreterInvoke(interpreter_m0_), "Invoke");

  check(TfLiteTensorCopyToBuffer(h_m0_mask_out_,  mask.data(),     bytes_mag),
        "lectura de la mascara (m0)");
  check(TfLiteTensorCopyToBuffer(h_m0_state_out_, state_m0.data(), bytes_state),
        "lectura del estado (m0)");   // <-- la recurrencia: pisa el estado viejo

  return mask;
  };

}
