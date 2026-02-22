import numpy as np 
import scipy as signal
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

def create_tapped_delay_line( X, K, Delta_frames, axis  = 2):
    """
        Generates a tapped delay line using a sliding window view with causal padding.

        Applies zero-padding to the left of the temporal axis equivalent to
        (Delta_frames + K - 1) to preserve system causality. Constructs the observation
        matrix by temporally aligning the delays and truncates the excess of
        generated windows to maintain the original block length (T).

        Parameters
        ----------
        X : numpy.ndarray
            Multidimensional input tensor representing the observation signal.
        K : int
            Prediction filter order (number of temporal taps).
        Delta_frames : int
            Initial prediction delay (\Delta_frames).
        axis : int, optional
            Index of the axis corresponding to the temporal dimension in tensor X.
            Defaults to 2.

        Returns
        -------
        numpy.ndarray
            Tensor containing the memory view of the shifted history. If X has
            dimensions (F, M, T) and axis=2, the returned tensor will have
            dimensions (F, M, K, T). The K axis is ordered such that index 0
            contains the sample with delay Delta_frames, and index K-1 contains the
            oldest sample (Delta_frames + K - 1).
    """
    T = X.shape[axis]
    dim = X.ndim
    pad_with = []

    for i in range(dim):
        if i ==axis:
            pad_with.append( (Delta_frames+ K - 1 , 0))
        else: 
            pad_with.append((0,0))

    #Zero pads to obtain consistent lenght
    Y = np.pad(X, pad_with , 'constant', constant_values=0)

    # delay K samples the input signal, add zeros to the begining 
    Y_delays = sliding_window_view(Y, K , axis =axis ) 

    #Transpose
    Y_delays_T = np.swapaxes(Y_delays, axis, -1  )

    #Mirror axes to order data: Y1 = Y[t-tao], Y2 = Y[t-tao-1] ...
    Y_window_view = np.flip(Y_delays_T, axis= axis)

    #Trim to original lenght
    Y_window_view = Y_window_view[..., :T]
    
    return Y_window_view


def get_reverb_tail(X_windowed, g ):
    """
        Compute de MIMO convolution using the g weights

        X : np.array ()

    """
    y_tilde = np.einsum('fnmk,fmkt->fnt' , g.conj(), X_windowed)

def compute_lambda_scaled_identity(x_tilde ):

    #stability loading
    delta = 1e-9

    M = x_tilde.shape[1]
    T = x_tilde.shape[2]

    lamda = np.mean( np.abs(x_tilde) **2 ,axis=1) + delta

    indentity = np.eye(M)

    Lambda_tilde = np.einsum( "ft,jk ->fjkt ", lamda, indentity) 

    
    
    lamda_inv = 1/ lamda
    Lamda_tilde_inv = np.einsum( "ft,jk ->fjkt ", lamda_inv, indentity)

                
    return Lambda_tilde, Lamda_tilde_inv


def ensamble_Psi(Y, K  ):
    F =Y.shape[0]
    M = Y.shape[1]
    T = Y.shape[2]

    # -1 Obtain a Y tensor with K taps
    # Pad zeros to maintain length
    Y_padded = np.pad( Y, ((0,0), (0,0), (K-1,0)), 'constant', constant_values= 0)

    # Window Slides
    Y_windows = sliding_window_view(Y_padded, K)
    Y_windows_T = Y_windows.transpose(0, 1, 3, 2)

    # Flip Taps axis to obtain incremental indexation
    Y_taps = np.flip(Y_windows_T , axis=2)

    #2 Ensamble Psi Matrix
    identitiy = np.eye(M)
    Psi_tensor = np.einsum("ij, fmkt -> fkimjt", identitiy, Y_taps )
    Psi = Psi_tensor.reshape(F, K*M*M, M, T)

    return Psi

def get_R_and_r( Lambda_tilde_inv, Psi, y, Delta_frames):
    #Delay Psi Delta_frames samples 
    #Zero pad
    pad_limits = [(0,0), (0,0), (0,0), (Delta_frames, 0)]    
    Psi_pad = np.pad( Psi, 
                           pad_limits, 
                           'constant',
                            constant_values=0
                            )
    
    #Chop to mantain original lenght
    Psi_delayed = Psi_pad[:,:,:,:-Delta_frames]

    #Construct R
    R_hat = np.einsum( 'fant, fnmt, fjmt -> faj', Psi_delayed.conj(), Lambda_tilde_inv, Psi_delayed.conj() )
    
    #Construct r
    r_hat = np.einsum( 'fant, fnmt, fmt -> fa', Psi_delayed.conj(), Lambda_tilde_inv, y )

    return R_hat, r_hat

def batch_dereverb( Y, fs, K):
    # Constants
    n_window, n_overlap = 256, 192
    d_loading = 1e-6
    iterations = 7
    Delta_frames = 2
    K = 30

    # Tranform to frecuency domain
    f, t, Y = signal.stft(Y=Y, 
                          fs=fs, 
                          nperseg= n_window, 
                        noverlap=n_overlap, 
                        window='hann', 
                        axis=1)
    
    Y = Y.transpose(1,0,2)
    F = Y.shape[0]
    M = Y.shape[1]

    # 1) -- Inicialice filter G = 0
    g_hat = np.zeros( (F, K, M, M), dtype=complex)

    # Obtaing K tapped and delayed signals
    Y_windowed = create_tapped_delay_line(Y, K = K, Delta_frames = Delta_frames, axis= 2)

    #Ensamble Psi
    Psi = ensamble_Psi(Y, K)

    # Start Iterations
    for i in range(iterations):

        # 2) -- Compute De - Reverberation Y(t) = yl(t) - sum( G* yl ) --
        Y_tilde =  np.einsum('fknm, fmkt -> fnt' , g_hat.conj(), Y_windowed)

        # Obtain dereverbated output
        X_hat = Y - Y_tilde

        # 3) -- Obtain Spatial Correlation Matrix Lamda_hat
        # Compute landa using scaled identity aproximation under the assumtion same energy on each sensor
        Lamda_hat,  Lamda_hat_inv = compute_lambda_scaled_identity(X_hat)

        

        # 4) -- Obtain Spatial Correlation Matrix Lamda_hat
        R_hat, r_hat = get_R_and_r( Lamda_hat_inv, Psi, Y, Delta_frames)

        # 5) -- Obtain Optimized weights
        # Diagonal Loading to avoid Non-invertion.
        
        R_hat = R_hat + d_loading * np.eye( K* M *M ).reshape(1,  K* M *M ,  K* M *M )

        # Solve Linear System 
        g_hat = np.linalg.solve(R_hat, r_hat )
        g_hat = g_hat.reshape(F,  K, M, M)
        
    # Compute last output
    Y_tilde =  np.einsum('fknm, fmkt -> fnt' , g_hat.conj(), Y_windowed)
    X_hat = Y - Y_tilde
    X_hat = X_hat.transpose(1,0,2)

    # Inverse Transform
    _, x_out = signal.istft(X_hat, fs= fs, window='hann', nperseg=n_window, noverlap=n_overlap)

    return  x_out #shape (N, T)


    
if __name__ == "__main__":

    matrix = np.random.normal( size = (3,5)).astype(int)

    print(matrix)
    
    K = 5
    tau = 4

    matrix_windowed = create_tapped_delay_line(matrix, K, tau, 1)

    print(matrix_windowed)
    

  
