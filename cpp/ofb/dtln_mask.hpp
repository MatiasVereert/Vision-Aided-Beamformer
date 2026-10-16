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


class DtlnMask{
 private:
      // model an intepreter
      TfLiteInterpreter* interpreter_m0_ = nullptr;
      TfLiteModel* model_ = nullptr;
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
      kiss_fftr_cfg ifftr_cfg_ = nullptr; 

  public: 

    DtlnMask( const char* model_path_0 ){
      // Load and create  model
      model_ = TfLiteModelCreateFromFile( model_path_0);
      if (model_ == nullptr )
        throw std::runtime_error("no se pudo abrir el .tflite");

      // create Options
      TfLiteInterpreterOptions* options_0 = TfLiteInterpreterOptionsCreate();

      // delegate model 0
      TfLiteXNNPackDelegateOptions xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
      xnnpack_options.num_threads = 1;
      xnnpack_delegate_m0_ = TfLiteXNNPackDelegateCreate( &xnnpack_options );
      TfLiteInterpreterOptionsAddDelegate( options_0, xnnpack_delegate_m0_);
;

      // create interpeter and delete options
      interpreter_m0_ = TfLiteInterpreterCreate(model_, options_0);

      TfLiteInterpreterOptionsDelete(options_0);
      if (interpreter_m0_ == nullptr )
        throw std::runtime_error("TfLiteInterpreterCreate fallo");

      // el interprete ya no necesita el model: se puede borrar aca
      TfLiteModelDelete(model_);  model_ = nullptr;

      // Allocate
      if (TfLiteInterpreterAllocateTensors(interpreter_m0_) != kTfLiteOk )
        throw std::runtime_error("AllocateTensors fallo");

      h_m0_mag_in_    = TfLiteInterpreterGetInputTensor (interpreter_m0_, 0);
      h_m0_state_in_  = TfLiteInterpreterGetInputTensor (interpreter_m0_, 1);
      h_m0_mask_out_  = TfLiteInterpreterGetOutputTensor(interpreter_m0_, 0);
      h_m0_state_out_ = TfLiteInterpreterGetOutputTensor(interpreter_m0_, 1);

      // fft config 
      fftr_cfg_ =  kiss_fftr_alloc( block_len, 0, nullptr, nullptr) ;
      ifftr_cfg_  = kiss_fftr_alloc( block_len, 1, nullptr, nullptr) ;
    }

    ~DtlnMask(){
      // el interprete primero: usa el delegate 
      TfLiteInterpreterDelete(interpreter_m0_);
      TfLiteXNNPackDelegateDelete(xnnpack_delegate_m0_);
    }

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






