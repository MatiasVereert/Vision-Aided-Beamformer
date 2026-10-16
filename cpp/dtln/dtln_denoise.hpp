#pragma once 
#include <dtln/config.hpp>
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>
#include <array>
#include <cmath> 
#include <complex>
#include <stdexcept>

// Wrap the C headers with extern "C" for C++ compatibility
extern "C" {
    #include "kiss_fft.h"
    #include "kiss_fftr.h" 
}

namespace dtln {

  using  Frame_real = std::array< float, block_len>;
  using  Frame_mag = std::array< float, k_bins>;
  using  Frame_cplx = std::array< std::complex<float>, k_bins>; 


  class DtlnDenoise{
    private:
      // model an intepreter
      TfLiteInterpreter* interpreter_m0_ = nullptr;
      TfLiteInterpreter* interpreter_m1_ = nullptr;

      TfLiteModel* model_m0_ = nullptr;
      TfLiteModel* model_m1_ = nullptr;

      // delegator
      TfLiteDelegate* xnnpack_delegate_m0_ = nullptr;
      TfLiteDelegate* xnnpack_delegate_m1_ = nullptr;

      // input buffers
      std::array<float, block_len> input_buffer_{};
      std::array<float, block_len> output_buffer_{};

      // state
      std::array<float, k_state> state_m0{};
      std::array<float, k_state> state_m1{};

      // Interpreter Tensors Ptrs

      TfLiteTensor*       h_m0_mag_in_    = nullptr;
      TfLiteTensor*       h_m0_state_in_  = nullptr;
      const TfLiteTensor* h_m0_mask_out_  = nullptr;
      const TfLiteTensor* h_m0_state_out_ = nullptr;

      TfLiteTensor*       h_m1_block_in_  = nullptr;
      TfLiteTensor*       h_m1_state_in_  = nullptr;
      const TfLiteTensor* h_m1_block_out_ = nullptr;
      const TfLiteTensor* h_m1_state_out_ = nullptr;

      //kissfft config 
      kiss_fftr_cfg fftr_cfg_ = nullptr; 
      kiss_fftr_cfg ifftr_cfg_ = nullptr; 

    public:

      DtlnDenoise( const char* model_path_0, const char* model_path_1){
        // Load and create  model
        model_m0_ = TfLiteModelCreateFromFile( model_path_0);
        model_m1_ = TfLiteModelCreateFromFile( model_path_1);
        if (model_m0_ == nullptr || model_m1_ == nullptr)
          throw std::runtime_error("no se pudo abrir alguno de los .tflite");

        // create Options
        TfLiteInterpreterOptions* options_0 = TfLiteInterpreterOptionsCreate();
        TfLiteInterpreterOptions* options_1 = TfLiteInterpreterOptionsCreate();

        // delegate model 0
        TfLiteXNNPackDelegateOptions xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
        xnnpack_options.num_threads = 1;
        xnnpack_delegate_m0_ = TfLiteXNNPackDelegateCreate( &xnnpack_options );
        TfLiteInterpreterOptionsAddDelegate( options_0, xnnpack_delegate_m0_);

        xnnpack_delegate_m1_ = TfLiteXNNPackDelegateCreate( &xnnpack_options );
        TfLiteInterpreterOptionsAddDelegate( options_1, xnnpack_delegate_m1_);

        // create interpeter and delete options
        interpreter_m0_ = TfLiteInterpreterCreate(model_m0_, options_0);
        interpreter_m1_ = TfLiteInterpreterCreate(model_m1_, options_1);
        TfLiteInterpreterOptionsDelete(options_0);
        TfLiteInterpreterOptionsDelete(options_1);
        if (interpreter_m0_ == nullptr || interpreter_m1_ == nullptr)
          throw std::runtime_error("TfLiteInterpreterCreate fallo");

        // el interprete ya no necesita el model: se puede borrar aca
        TfLiteModelDelete(model_m0_);  model_m0_ = nullptr;
        TfLiteModelDelete(model_m1_);  model_m1_ = nullptr;

        // Allocate
        if (TfLiteInterpreterAllocateTensors(interpreter_m0_) != kTfLiteOk ||
            TfLiteInterpreterAllocateTensors(interpreter_m1_) != kTfLiteOk)
          throw std::runtime_error("AllocateTensors fallo");

        h_m0_mag_in_    = TfLiteInterpreterGetInputTensor (interpreter_m0_, 0);
        h_m0_state_in_  = TfLiteInterpreterGetInputTensor (interpreter_m0_, 1);
        h_m0_mask_out_  = TfLiteInterpreterGetOutputTensor(interpreter_m0_, 0);
        h_m0_state_out_ = TfLiteInterpreterGetOutputTensor(interpreter_m0_, 1);

        h_m1_block_in_  = TfLiteInterpreterGetInputTensor (interpreter_m1_, 0);
        h_m1_state_in_  = TfLiteInterpreterGetInputTensor (interpreter_m1_, 1);
        h_m1_block_out_ = TfLiteInterpreterGetOutputTensor(interpreter_m1_, 0);
        h_m1_state_out_ = TfLiteInterpreterGetOutputTensor(interpreter_m1_, 1);

 
        // fft config 
        fftr_cfg_ =  kiss_fftr_alloc( block_len, 0, nullptr, nullptr) ;
        ifftr_cfg_  = kiss_fftr_alloc( block_len, 1, nullptr, nullptr) ;
      }

      ~DtlnDenoise(){
        // el interprete primero: usa el delegate 
        TfLiteInterpreterDelete(interpreter_m0_);
        TfLiteInterpreterDelete(interpreter_m1_);
        TfLiteXNNPackDelegateDelete(xnnpack_delegate_m0_);
        TfLiteXNNPackDelegateDelete(xnnpack_delegate_m1_);
      }

      Frame_real process_block( const Frame_real& audio_input);

      // Etapa A: solo el modelo 0. Entra una magnitud ya calculada (257), sale
      // la mascara (257). Encadena state_m0 igual que process_block, asi que
      // no mezclar las dos llamadas sobre el mismo objeto.
      Frame_mag mask_from_magnitude( const Frame_mag& in_mag);
    };
  
    // Lanza si una llamada de tflite no devolvio kTfLiteOk.
  inline void check(TfLiteStatus s, const char* what){
    if (s != kTfLiteOk) throw std::runtime_error(what);
  }

  
 inline Frame_real DtlnDenoise::process_block(const Frame_real& input_frame){

    Frame_real output_frame{};
    std::array<kiss_fft_cpx, k_bins> bins{};   // tipo nativo de kissfft: sin reinterpret_cast
    Frame_mag  in_mag{};
    Frame_mag  mask{};
    Frame_real estimated_block{};

    // ---- FFT real: 512 reales -> 257 complejos. Igual que np.fft.rfft,
    //      sin normalizar en esta direccion.
    kiss_fftr(fftr_cfg_, input_frame.data(), bins.data());

    for (int k = 0; k < k_bins; ++k)
      in_mag[k] = std::sqrt(bins[k].r * bins[k].r + bins[k].i * bins[k].i);

    // ---- modelo 0: (magnitud, estado) -> (mascara, estado')
    check(TfLiteTensorCopyFromBuffer(h_m0_mag_in_,   in_mag.data(),   bytes_mag),
          "copia a la entrada de magnitud (m0)");
    check(TfLiteTensorCopyFromBuffer(h_m0_state_in_, state_m0.data(), bytes_state),
          "copia del estado a la entrada (m0)");

    check(TfLiteInterpreterInvoke(interpreter_m0_), "Invoke del modelo 0");

    check(TfLiteTensorCopyToBuffer(h_m0_mask_out_,  mask.data(),     bytes_mag),
          "lectura de la mascara (m0)");
    check(TfLiteTensorCopyToBuffer(h_m0_state_out_, state_m0.data(), bytes_state),
          "lectura del estado (m0)");   // <-- la recurrencia: pisa el estado viejo

    // ---- aplicar la mascara. En la referencia es
    //        in_mag * mask * exp(1j*phase)
    //      pero in_mag*exp(1j*phase) ES el espectro, asi que alcanza con
    //      multiplicar los bins complejos por la mascara real.
    for (int k = 0; k < k_bins; ++k) {
      bins[k].r *= mask[k];
      bins[k].i *= mask[k];
    }

    // ---- IFFT real: 257 complejos -> 512 reales.
    //      kissfft NO normaliza; np.fft.irfft divide por n. Hay que dividir a mano.
    //      1/512 es exacto en binario, asi que multiplicar == dividir, bit a bit.
    kiss_fftri(ifftr_cfg_, bins.data(), estimated_block.data());

    const float inv_n = 1.0f / static_cast<float>(block_len);
    for (int n = 0; n < block_len; ++n)
      estimated_block[n] *= inv_n;

    // ---- modelo 1: (bloque, estado) -> (bloque', estado')
    check(TfLiteTensorCopyFromBuffer(h_m1_block_in_, estimated_block.data(), bytes_block),
          "copia del bloque a la entrada (m1)");
    check(TfLiteTensorCopyFromBuffer(h_m1_state_in_, state_m1.data(), bytes_state),
          "copia del estado a la entrada (m1)");

    check(TfLiteInterpreterInvoke(interpreter_m1_), "Invoke del modelo 1");

    check(TfLiteTensorCopyToBuffer(h_m1_block_out_, output_frame.data(), bytes_block),
          "lectura del bloque de salida (m1)");
    check(TfLiteTensorCopyToBuffer(h_m1_state_out_, state_m1.data(), bytes_state),
          "lectura del estado (m1)");

    return output_frame;
    }

  inline Frame_mag DtlnDenoise::mask_from_magnitude(const Frame_mag& in_mag){
    Frame_mag mask{};
    check(TfLiteTensorCopyFromBuffer(h_m0_mag_in_,   in_mag.data(),   bytes_mag),
          "copia a la entrada de magnitud (m0)");
    check(TfLiteTensorCopyFromBuffer(h_m0_state_in_, state_m0.data(), bytes_state),
          "copia del estado a la entrada (m0)");
    check(TfLiteInterpreterInvoke(interpreter_m0_), "Invoke del modelo 0");
    check(TfLiteTensorCopyToBuffer(h_m0_mask_out_,  mask.data(),     bytes_mag),
          "lectura de la mascara (m0)");
    check(TfLiteTensorCopyToBuffer(h_m0_state_out_, state_m0.data(), bytes_state),
          "lectura del estado (m0)");
    return mask;
  }

  };

