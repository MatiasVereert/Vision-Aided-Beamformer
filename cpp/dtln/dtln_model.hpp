#pragma once 
#include <vector> 
#include <dtln/config.hpp>
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>
#include <array>

namespace dtln {
  using  cplx_t = std::complex<float>; 
  using  Frame_real = std::array< float, block_len>;
  using  Frame_cplx = std::array< float, block_len>; 


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

      // Direct memory pointers for Model 0
      float* m0_audio_in_ptr_ = nullptr;
      float* m0_state_in_ptr_ = nullptr;
      const float* m0_audio_out_ptr_ = nullptr;
      const float* m0_state_out_ptr_ = nullptr;

      // Direct memory pointers for Model 1
      float* m1_audio_in_ptr_ = nullptr;
      float* m1_state_in_ptr_ = nullptr;
      const float* m1_audio_out_ptr_ = nullptr;
      const float* m1_state_out_ptr_ = nullptr;

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

        // Save interpreter tensor pointers to avoid recall in during proces runtime
        m0_audio_in_ptr_  = TfLiteInterpreterGetInputTensor(interpreter_m0_, 0)->data.f;
        m0_state_in_ptr_  = TfLiteInterpreterGetInputTensor(interpreter_m0_, 1)->data.f;

        m0_audio_out_ptr_ = TfLiteInterpreterGetOutputTensor(interpreter_m0_, 0)->data.f;
        m0_state_out_ptr_ = TfLiteInterpreterGetOutputTensor(interpreter_m0_, 1)->data.f;

        m1_audio_in_ptr_  = TfLiteInterpreterGetInputTensor(interpreter_m1_, 0)->data.f;
        m1_state_in_ptr_  = TfLiteInterpreterGetInputTensor(interpreter_m1_, 1)->data.f;

        m1_audio_out_ptr_ = TfLiteInterpreterGetOutputTensor(interpreter_m1_, 0)->data.f;
        m1_state_out_ptr_ = TfLiteInterpreterGetOutputTensor(interpreter_m1_, 1)->data.f;
      }

      ~DtlnDenoise(){
        // el interprete primero: usa el delegate
        TfLiteInterpreterDelete(interpreter_m0_);
        TfLiteInterpreterDelete(interpreter_m1_);
        TfLiteXNNPackDelegateDelete(xnnpack_delegate_m0_);
        TfLiteXNNPackDelegateDelete(xnnpack_delegate_m1_);
      }

      DtlnDenoise::step( const frame_real);
  };




  DtlnDenoise::step( const frame_real){



  }


}  // namespace dtln