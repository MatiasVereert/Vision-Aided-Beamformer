import numpy as np 
import scipy.signal as signal



def get_lambda_frame(Y_buffer_mirror, delta ):
   " Equation (17)"
   F, D, _ = Y_buffer_mirror.shape

   Y_abs_2 = np.abs(Y_buffer_mirror) **2
   Y_abs_2_sum = np.sum(Y_abs_2, axis=1 )
   Y_abs_2_sum_sum = np.sum( Y_abs_2_sum[:, 0:delta +1], axis = 1)
   denominator = (delta +1 ) * D 
   lamda_frame = Y_abs_2_sum_sum / denominator

   return lamda_frame

def get_K( R_inv, Y_obs_frame, lambda_frame, alpha ):
  # matricial product  (DK, DK)(DKx1) -> (DK) broadcast F
  nominator = np.einsum( 'fdk, fk-> fd', R_inv, Y_obs_frame) # Shape (DKx1) 

  # internal product (1, DK)(DKx1) -> (1) broadcast F 
  denominator_term = np.einsum('fd, fd -> f', Y_obs_frame.conj(), nominator)
  denominator = alpha* lambda_frame + denominator_term

  #reshape
  denominator = denominator[:,None]

  K = nominator / denominator 

  return K 

def get_R_inv( R_inv,K, Y_obs_frame, alpha ):
  # (DK,1) (1,DK) (DK, DK) -> (DK, DK)
  denom_term = np.einsum( 'fd,fk,fkc-> fdc', K,Y_obs_frame.conj(), R_inv)
  denom = R_inv - denom_term

  R_inv = denom / alpha

  return R_inv

def get_X_frame_early( Y_frame, Y_obs_frame, G):
   X_hat_frame = Y_frame - np.einsum('fkd, fk-> fd', G.conj(), Y_obs_frame)
   return X_hat_frame
   



def frame_online_WPE(Y_in, taps, delay, delta, alpha = 0.99999):
  F, D, T = Y_in.shape

  # Load Buffer
  # Para cargar Y_obs tienen que pasar Delta + K muestras 
  buffer_len = delay + taps 
  Y_buffer = np.zeros( (F, D, buffer_len), dtype = np.complex128 )

  I_dk = np.eye( (D*taps), dtype = np.complex128)
  R_inv = np.broadcast_to(I_dk, (F, D*taps, D*taps) ).copy()
  G_frame = np.zeros((F, D*taps, D ), dtype = np.complex128  )
  X_hat = np.zeros( (F, D, T), dtype = np.complex128)


  for t in range(T):
    # Y frame 
    Y_frame = Y_in[:,:,t]

    # Shift Left (Last pos remains unchanged)
    Y_buffer[:,:, :-1] = Y_buffer[:,:, 1:]
    Y_buffer[:,:,-1] = Y_frame
    Y_buffer_mirror = Y_buffer[:,:,::-1]

    # Extract Y_obs from buffer
    # Cut from t=delta to t= delta+taps
    Y_obs_frame = Y_buffer_mirror[:,:,delay:]
    Y_obs_frame = Y_obs_frame.transpose(0,2,1)
    Y_obs_frame = Y_obs_frame.reshape(F,taps*D)

    # -------- Filter with past G(t-1) -------------
    # Filter and obtain X_hat
    X_hat_frame = get_X_frame_early( Y_frame, Y_obs_frame, G_frame)

    # Save to output
    X_hat[:,:,t] = X_hat_frame

    # -------- Update G(t)  --------------------

    # Estimate variance
    lamda_frame = get_lambda_frame(Y_buffer_mirror, delta )
    
    # Ensamble K 
    K = get_K(R_inv, Y_obs_frame, lamda_frame, alpha )

    # Update R_inv
    R_inv = get_R_inv( R_inv,K, Y_obs_frame, alpha )

    # Update G
    G_frame = G_frame + np.einsum( 'fk,fd->fkd', K, X_hat_frame.conj())

  return X_hat





  