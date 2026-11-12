#pragma once
#include <ofb/config.hpp>
#include <ofb/types.hpp>
#include <array>
#include <complex>
#include <cmath>

namespace ofb{

  class SoudenCore {
    public: 
      void update(const Frame& X_frame,
                  const Reals& m_s,
                  const Reals& m_n);
      Vec solve_bin( int f ) const;

      // Solo lectura, para la GUI: Phi = Num / Den (sin carga diagonal).
      const std::array<TriMat, K>& num_ss() const { return Num_SS; }
      const std::array<TriMat, K>& num_nn() const { return Num_NN; }
      const Reals& den_ss() const { return Den_SS; }
      const Reals& den_nn() const { return Den_NN; }

    private:
        // (K, M, M) complejas, pero solo el triangulo inferior (ver TriMat):
        // 2 x 257 x 78 x 8 B = 321 kB en vez de 592 kB, que con el weights
        // cache de XNNPACK (~540 kB) se disputan el MB de L2 del A53.
        std::array<TriMat, K> Num_SS{}; //signal cov
        std::array<TriMat, K> Num_NN{}; //noise cov
        Reals Den_SS{0};
        Reals Den_NN{0};
        // Los pesos NO viven aca: un solo Vec es el estado equivocado (guarda un
        // bin cuando hacen falta K) y tener a mano un miembro `w` ademas de la
        // variable local es como se escribe el equivocado. Cuando este el solve
        // de los K bines, va un `Frame W{}` miembro y `W[f] = solve_bin(f)`.
  };

  inline void SoudenCore::update(const Frame& X_frame,
              const Reals& m_s,
              const Reals& m_n){

        //updates all bins per call
        for (int f = 0 ; f<K; f++){
          cplx_t Phi_XX_element{};

          // Solo c <= r: el recorrido por filas cae justo en el orden de tri(),
          // asi que el indice empaquetado es un contador.
          int p = 0;
          for (int r=0; r<M; r++ ){
            for (int c=0; c<=r; c++, p++ ){
              // Inst. Cov Matrix
              Phi_XX_element = X_frame[f][r] * std::conj(X_frame[f][c]);

              Num_SS[f][p] = kAlpha * Num_SS[f][p] + Phi_XX_element * m_s[f];
              Num_NN[f][p] = kAlpha * Num_NN[f][p] + Phi_XX_element * m_n[f];
            }
          }
          Den_SS[f] = kAlpha *  Den_SS[f] + m_s[f];
          Den_NN[f] = kAlpha *  Den_NN[f] + m_n[f];
        }
      }

  inline Mat unpack_load( const TriMat& Num, real_t Den,
                          real_t rel, real_t abs_floor ){
    /**
        Phi = Num / Den, desempaquetada al triangulo INFERIOR de una Mat (el de
        arriba queda en cero: Cholesky_bin solo lee j <= i), con la carga
        diagonal ya sumada.

        Hermitiana EXACTA por construccion: se guarda un solo triangulo, asi que
        no hay dos mitades que el redondeo pueda desviar y el 0.5*(P + P^H) de
        ofb.py no tiene nada que corregir. Solo queda forzar la diagonal real.

        Carga diagonal RELATIVA a la traza, igual que SoudenCore.solve de ofb.py:
        invariante a la escala de la entrada. Es lo que garantiza que las DOS
        Cholesky existan siempre -- la de Phi_NN por definida positiva, y la de
        Phi_SS porque al arranque tiene rango 1 y no lo es hasta acumular M
        frames. Con esto la rama `else` de Cholesky_bin es inalcanzable.
    */
    const real_t inv = 1 / Den;
    Mat P{};
    real_t tr = 0;
    for (int i = 0; i < M; i++){
      for (int j = 0; j < i; j++)
        P[i][j] = inv * Num[tri(i, j)];
      const real_t d = (inv * Num[tri(i, i)]).real();
      P[i][i] = cplx_t(d, 0.0f);
      tr += d;
    }
    const real_t add = rel * (tr / M) + abs_floor;
    for (int i = 0; i < M; i++) P[i][i] += cplx_t(add, 0);
    return P;
  }

  inline Vec column_ref( const TriMat& Num, real_t Den ){
    /**
        Phi e_ref = la columna kRefMic de Num / Den, SIN carga: es el lado
        derecho del numerador. Por encima de la diagonal no esta guardada y sale
        de la fila kRefMic conjugada: Phi[i][ref] = conj(Phi[ref][i]).
    */
    const real_t inv = 1 / Den;
    Vec c{};
    for (int i = 0; i < M; i++){
      if      (i > kRefMic) c[i] = inv * Num[tri(i, kRefMic)];
      else if (i < kRefMic) c[i] = std::conj(inv * Num[tri(kRefMic, i)]);
      else                  c[i] = cplx_t((inv * Num[tri(i, i)]).real(), 0.0f);
    }
    return c;
  }

  inline Mat Cholesky_bin( const Mat& Cov){
    /**
        Solves Cholesky decomposition and returns de lower triangular component.
        Solo lee el triangulo inferior de Cov (j <= i).
    */
    Mat L{};
    real_t minus_d{};
    cplx_t minus{};
    real_t L_diag_inv{}; 

    for(int j=0; j<M; j++){
      // Diagonal L_jj
      for (int k=0; k< j; k++){
        minus_d += std::norm((L[j][k])); 
      }

      if (Cov[j][j].real() > minus_d){
        L[j][j] = std::sqrt( Cov[j][j].real() - minus_d ); 

        L_diag_inv = 1.0f / L[j][j].real(); 
        
      } else {
        L[j][j] = 0;
        L_diag_inv = 0.0f;   // si no, queda el de la columna j-1
        //Advertencia de error ¿return o relleno
        //Despues esto se podria tranformar en un bucle cerrado y auto regularizar
        //Pero probablemente convenga aislar eso de la logica matematica
      }
      minus_d = 0; 

      for(int i =j+1; i<M;i++){
        // Under Diagonal L_ij
        for ( int k =0; k<j; k++){
          minus += L[i][k] * std::conj(L[j][k]); 
        }
        L[i][j] = L_diag_inv * (Cov[i][j] - minus )  ; 
        minus = 0; 
      }
    }
    return L; 
  }

  inline Mat Forward_solve_Mat_bin( const Mat& L_B, const Mat& L_A){
    Mat X_bin{};
    
    for (int j =0; j<M; j++){

      for (int i= j; i<M; i++){
        // la diagonal de una Cholesky es real POR CONSTRUCCION (sale de una
        // raiz real): dividir en complejo acá llama a __divsc3 de libgcc.
        const real_t L_ii_inv = 1.0f / L_B[i][i].real();
        cplx_t acc =0; 
        for (int k = j; k< i; k++){
          acc += L_B[i][k] * X_bin[k][j];
        }
        X_bin[i][j] = L_ii_inv * (L_A[i][j] - acc );
      }
    }
    return X_bin;
  }

  inline real_t frobenius_norm_square( const Mat& X ){
    real_t acc = 0; 
  
    // j<=i: X es triangular inferior, el resto es cero y no hace falta tocarlo.
    for (int i =0; i<M; i++){
      for (int j =0; j<=i; j++){
        acc += std::norm( X[i][j] );   // norm() YA es |z|^2: elevarlo daba |z|^4

      }
    }
    return acc;
  }

  inline Vec Forward_solve_Vec_bin(const Mat& L_B, const Vec& c ){
    Vec y_bin{};

    for (int i= 0; i<M; i++){
      const real_t L_ii_inv = 1.0f / L_B[i][i].real();
      cplx_t acc =0;
      for (int k = 0; k< i; k++){
        acc += L_B[i][k] * y_bin[k];
      }
      y_bin[i] = L_ii_inv * (c[i] - acc );
    }
    return y_bin;
  }

  inline Vec Backward_solve_Vec_bin(const Mat& L_B, const Vec& y) {
      Vec w{};
      for (int i = M - 1; i >= 0; --i) {
          cplx_t acc = 0;
          for (int k = i + 1; k < M; ++k)
              acc += std::conj(L_B[k][i]) * w[k];
          w[i] = (y[i] - acc) * (1.0f / L_B[i][i].real());
      }
      return w;
  }

  inline Vec SoudenCore::solve_bin( int f ) const {
    // inline: definida FUERA de la clase y en un header. Sin esto, el dia que
    // dos .cpp incluyan solve.hpp el linker ve dos definiciones. Y tiene que
    // quedar DESPUES de los helpers: un metodo definido adentro del cuerpo de
    // la clase no ve las funciones libres declaradas mas abajo.

    // El +1e-15 del denominador es el de ofb.py: al arranque Den puede ser
    // ~m_scm^8, que es diminuto.
    const real_t den_ss = Den_SS[f] + 1e-15f;
    const real_t den_nn = Den_NN[f] + 1e-15f;

    // Carga diagonal, como en ofb.py. En float32 kMinLoading = 1e-5: el 1e-9 de
    // Python queda POR DEBAJO del epsilon de float y la carga desaparece en el
    // redondeo.
    // `A` es Phi_SS CARGADA: existe solo para que su Cholesky exista siempre.
    // Phi_SS SIN cargar se sigue necesitando, pero solo su columna kRefMic
    // como lado derecho (column_ref).
    const Mat A           = unpack_load( Num_SS[f], den_ss, kMinLoading, 1e-30f );
    const Mat Phi_NN_load = unpack_load( Num_NN[f], den_nn, kMinLoading, 1e-12f );
    const Vec c           = column_ref ( Num_SS[f], den_ss );

    // Cholesky (Phi_NN_inv Phi_SS = L_B_inv L_B_-H L_A L_A_H
    Mat L_A = Cholesky_bin(A);
    Mat L_B = Cholesky_bin(Phi_NN_load);

    // Denominator
    // Obtain X = L_B_inv L_A solving L_B X = L_A
    // OJO con el orden: los dos argumentos son Mat, asi que invertirlos compila
    // igual y resuelve el sistema equivocado en silencio.
    Mat X = Forward_solve_Mat_bin( L_B, L_A );

    //TR(Phi_NN_inv Phi_SS) = TR(X X_H) = FrobeniusNorm(X)
    real_t Tr = frobenius_norm_square( X );

    // Nominator
    // Solve LB y = c  con c = Phi_ss.e_ref
    Vec y = Forward_solve_Vec_bin(L_B, c );
    // Solve LB_H w_nom = y
    Vec w_nom = Backward_solve_Vec_bin(L_B, y );

    // w = w_nom / Tr( Phi_NN_inv Phi_SS)
    // Tr = lambda_S + M >= M por construccion (ver el docstring de SoudenCore en
    // ofb.py): no puede acercarse a cero, asi que el piso es solo una guarda.
    const real_t Tr_inv = (Tr > 0.0f) ? 1.0f / Tr : 0.0f;

    Vec w_bin{};
    for (int i =0; i<M; i++){
      w_bin[i] = Tr_inv * w_nom[i];
    }
    return w_bin; 
  }

} // end namespace



