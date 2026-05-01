#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

'''
This is the implementation with CGMM based on paper "Online MVDR Beamformer Based on Complex Gaussian Mixture Model with Spatial Prior for Noise Robust ASR"
We use Circularly Symmetric Gaussian Mixture Model (Complex domain with mean=0 and pseudocovariance=0)
# M: channel number
# K: CGMM cluster number (usually is 2)
# T: frame number
'''

class CGMM:
  def __init__(self,Y,K=2,openAssert=False,init_R=None):
    self._openAssert = openAssert
    self._K = K # number of clusters (number of sound sources + 1 background noise)
    self._Y = Y # Y: (M = mic_number or feat_dim, T = frame_num)
    self._M, self._T = Y.shape
    self._init_R = init_R  # Guardamos la inicialización externa
    M, T = self._M, self._T
    # declares the parameters shape and type
    self._Phi = np.zeros([K,T],dtype=complex) # (K,T): variance of signals w.r.t. all time frames for each clusters
    self._R = np.zeros([K,M,M],dtype=complex) # (K,M,M): covariances for each clusters
    self._invR = np.zeros([K,M,M],dtype=complex) # (K,M,M): inverse covariances for each clusters
    self._alpha = np.zeros([K,]) # (K,): mixture weights
    self._posterior = np.zeros([K,T]) # posterior prob.
    self._steerVec = np.zeros([K,]) # steering vector

    self._initParam()

  def _initParam(self):
    M, T, K = self._M, self._T, self._K
    Y = self._Y

    if self._init_R is not None:
        # Usamos la inicialización guiada (por ejemplo, desde visión)
        self._R = np.copy(self._init_R)
    else:
        # Inicialización ciega original
        if K == 2:
            self._R[0,...] = 1e-6 * np.eye(M).astype(complex) # Ruido
            self._R[1,...] = np.matmul(Y, Y.conj().T) / T     # Mezcla total
        else:
            # (Mantener lógica original para K > 2)
            pass
    
    self._invR = np.linalg.inv(self._R)
    
    self._invR = np.linalg.inv(self._R)
    tmpMat = np.einsum('mt,kmn->knt',Y.conj(),self._invR)
    self._Phi = np.einsum('knt,nt->kt',tmpMat,Y)/M # (K,T)
    self._alpha = np.ones([K,])/K

  def getR(self):
    return np.copy(self._R)
  def getPost(self):
    return np.copy(self._posterior)
  def getPhi(self):
    return np.copy(self._Phi)
  def getMixWeights(self):
    return np.copy(self._alpha)

  def _calLogGaussianProb(self,Y,Phi,R,invR):
    """
    Arguments: (for nfft: num_fft, M: num_mics, T: num_frames)
        Y: (M, T), T observations with M-dim
        Phi: (T,), Representing the signal variance for each time t.
        R, invR: (M,M), the spatial (inverse) covariance matrix
    Return:
        logProb: (T,), the log-probabilities
    """
    M, T = Y.shape
    R = (R + np.transpose(np.conj(R))) / 2
    if self._openAssert:
      tmpMat = np.einsum('mt,mn->nt',Y.conj(),invR)
      tmpMat = np.einsum('nt,nt->t',tmpMat,Y)
      assert(np.allclose(np.real(tmpMat),np.real(Phi)*M))
      assert(np.allclose(np.imag(tmpMat),np.imag(Phi)*M))

    # Safely compute the determinant, enforcing a minimum positive value
    # to avoid log(0) or log(negative) issues due to floating point precision
    det = np.linalg.det(R).real
    det = np.maximum(det, 1e-12) 

    # Ensure Phi is strictly positive and real
    Phi_safe = np.maximum(np.real(Phi), 1e-12)

    # Calculate log probability with numerical safeguards
    logProb = -M*np.log(Phi_safe*np.pi) - np.log(det) - M
    
    if self._openAssert:
      assert(np.allclose(np.imag(logProb),0))

    return logProb

  def run(self,itr_num=10):
    """
    Maximal Likelihood (ML) with EM algorithm
        itr_num: iteration number
    Return:
        post: (K,T), posterior probabilities for T observations (K-dim)
    """
    K, M, T, Y = self._K, self._M, self._T, self._Y
    R, invR, Phi, alpha, post = self._R, self._invR, self._Phi, self._alpha, self._posterior
    log_post = np.zeros(post.shape)

    for itr in range(itr_num):

      # ===== E Step
      # log_post, post: (K,T)
      # SAFEGUARD: Evitar log(0) si alpha colapsa
      alpha_safe = np.maximum(alpha, 1e-12)
      log_alpha = np.log(alpha_safe) # (K,)
      
      for k in range(K):
        log_post[k,:] = log_alpha[k] + self._calLogGaussianProb(Y,Phi[k,:],R[k,...],invR[k,...])
      
      # SAFEGUARD: Restar el máximo para evitar overflow/underflow en la exponencial
      log_post = log_post - np.max(log_post, axis=0)
      post = np.exp(log_post)
      
      # Evitar división por 0 al normalizar el posterior
      post_denom = np.maximum(np.sum(post,axis=0), 1e-12)
      post = post/post_denom
      
      if self._openAssert:
        assert(np.allclose(np.sum(post,axis=0),1))
        
      post_sum = np.sum(post,axis=1) # (K,)
      # SAFEGUARD: Evitar división por cero si un cluster no recibe asignaciones
      post_sum_safe = np.maximum(post_sum, 1e-12)

      # ===== M Step
      # Update Phi
      tmpMat = np.einsum('mt,kmn->knt',Y.conj(),invR)
      Phi = np.einsum('knt,nt->kt',tmpMat,Y)/M # (K,T)
      
      # Update R
      # SAFEGUARD: Asegurar que Phi (varianza) sea estrictamente positiva
      Phi_safe = np.maximum(np.real(Phi), 1e-12)
      tmpMat = np.einsum('kt,mt->kmt',(post/Phi_safe),Y) # (K,M,T)
      R = np.einsum('kmt,tn->kmn',tmpMat,Y.T.conj()) # (K,M,M)
      R = np.einsum('kmn,k->kmn',R,1/post_sum_safe)
      
      # SAFEGUARD: Diagonal loading. Añadir un poco de ruido blanco a la diagonal
      # previene que la matriz de covarianza espacial se vuelva singular
      for k in range(K):
          R[k] += 1e-6 * np.eye(M)
          
      invR = np.linalg.inv(R)
      
      # Update alpha.
      alpha = post_sum_safe/T
      # Normalizar alpha para garantizar que sume 1 tras los ajustes
      alpha = alpha / np.sum(alpha) 

    # Compute post after all iterations
    alpha_safe = np.maximum(alpha, 1e-12)
    log_alpha = np.log(alpha_safe) # (K,)
    for k in range(K):
      log_post[k,:] = log_alpha[k] + self._calLogGaussianProb(Y,Phi[k,:],R[k,...],invR[k,...])
    
    log_post = log_post - np.max(log_post, axis=0)
    post = np.exp(log_post)
    post_denom = np.maximum(np.sum(post,axis=0), 1e-12)
    post = post/post_denom
    
    self._R, self._invR, self._Phi, self._alpha, self._posterior = R, invR, Phi, alpha, post
    return post


'''
This is the implementation with spatial prior CGMM based on paper "Online MVDR Beamformer Based on Complex Gaussian Mixture Model with Spatial Prior for Noise Robust ASR"
# M: channel number
# K: CGMM cluster number (usually is 2)
# T: frame number
'''
class PriorCGMM(CGMM):
  def __init__(self,Y,K=2,openAssert=False):
    CGMM.__init__(self,Y,K,openAssert)
    CGMM.run(self,itr_num=3)
    # Init Super-parameters
    # See https://en.wikipedia.org/wiki/Conjugate_prior for conjugate prior
    self._Eta = self._T # Control the ratio of previous v.s. new data
    # We use lambda (see definition in paper) instead of usual super-parameters in inverse-wishart
    # self._posterior is of shape (K,T)
    self._Lambda = np.sum(self._posterior,axis=1) # (K,)
    assert(len(self._Lambda)==K)

  def run(self,Y,itr_num=3):
    """
    Maximal A Posterior (MAP) with EM algorithm
        itr_num: iteration number
    Return:
        post: (K,T), posterior probabilities for T observations (K-dim)
    """
    self._Y = Y # Y: (M, T), set the new data as current Y
    M, T = Y.shape
    assert(M==self._M)
    self._T = T # set the new frame number as current T
    K = self._K
    R, invR, Phi, alpha, post = self._R, self._invR, self._Phi, self._alpha, self._posterior
    log_post = np.zeros(post.shape)

    for itr in range(itr_num):

      # ===== E Step
      # log_post, post: (K,T)
      log_alpha = np.log(alpha) # (K,)
      for k in range(K):
        log_post[k,:] = log_alpha[k] + self._calLogGaussianProb(Y,Phi[k,:],R[k,...],invR[k,...])
      post = np.exp(log_post)
      post = post/np.sum(post,axis=0)
      if self._openAssert:
        assert(np.allclose(np.sum(post,axis=0),1))
      post_sum = np.sum(post,axis=1) # (K,)

      # ===== M Step
      # Update Phi
      tmpMat = np.einsum('mt,kmn->knt',Y.conj(),invR)
      Phi = np.einsum('knt,nt->kt',tmpMat,Y)/M # (K,T)
      # # Update alpha
      # alpha = post_sum/T
      # Update R, MAP udpate
      lambda_next = self._Lambda + post_sum # (K,)
      tmpConst = (self._Eta + M + 1)/2
      numerator = self._Lambda + tmpConst # (K,)
      demonimator = lambda_next + tmpConst # (K,)
      tmpMat = np.einsum('kt,mt->kmt',(post/Phi),Y) # (K,M,T)
      tmpMat = np.einsum('kmt,tn->kmn',tmpMat,Y.T.conj()) # (K,M,M)
      # R: (K,M,M)
      priorInfo = np.einsum('k,kmn->kmn',numerator/demonimator,R) # (K,M,M)
      newInfo = np.einsum('k,kmn->kmn',1/demonimator,tmpMat) # (K,M,M)
      R = priorInfo + newInfo
      invR = np.linalg.inv(R)

    # Compute post after all iterations
    log_alpha = np.log(alpha) # (K,)
    for k in range(K):
      log_post[k,:] = log_alpha[k] + self._calLogGaussianProb(Y,Phi[k,:],R[k,...],invR[k,...])
    post = np.exp(log_post)
    post = post/np.sum(post,axis=0)
    post_sum = np.sum(post,axis=1) # (K,)
    self._R, self._invR, self._Phi, self._alpha, self._posterior = R, invR, Phi, alpha, post

    # update super-parameters
    self._Eta = self._Eta + T
    self._Lambda = self._Lambda + post_sum # (K,)

    return post

