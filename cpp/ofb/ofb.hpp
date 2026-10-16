#pragma once 
#include "kiss_fft.h"
#include "kiss_fftr.h" 
#include <array>
#include <cmath> 
#include <complex>
#include <stdexcept>
#include <config.hpp>
#include <isir_tracker.hpp>
#include <solve.hpp>
#include <types.hpp>
#include <dtln_mask.hpp>


namespace ofb{

  class nm_mvdr_ofb{
    private: 

    // Declarar atributos: 

      // Input Buffer

      // SoudenCoreSoudenCore
      // DtlnModel
      // 
      SoudenCore = nullptr; 

    


    public:
      // constructor
      nm_mvdr_ofb(const char* model_path){
        
        //init SoudenCore

        DtlnMask DtlnMask( model_path); 

        //init DtlnMask

      };

    Frame_real procces_frame( const Frame_real& input_frame );


  };


  inline Frame_real nm_mvdr_ofb::procces_frame( const Frame_real& input_frame ){

    Frame_real output_frame{};

    // multiframe en tiempo - porcces_frame - frame en tiempo

    // fft x canal

    // estimar mascara
    
    // Computar MVDR 

    // Procesar input_frame

    // ifft de salida 

    return output_frame 
  }

}