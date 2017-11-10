from __future__ import print_function
from __future__ import division
import numpy as np
import numpy.linalg as la
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
#import time
#import itertools
from abc import abstractmethod
import pywt



class AbstractOperator(object):
    '''To make sure that the derived classes have the right functions'''
    @abstractmethod
    def apply(self, x):
        """Compute Ax"""
        pass
       
    @abstractmethod 
    def inv(self, x):
        """A^-1 x"""
        pass

# reals space: everything 2D
# T-space: 1D

class DCT(AbstractOperator):
    '''Discrete cosine transform'''
    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, image):
        Timage = spfft.dct(spfft.dct(image, norm='ortho', axis=0), norm='ortho', axis=1)
        return Timage.reshape(-1)
    
    def inv(self, Timage):
        Timage = Timage.reshape(self.shape)
        return spfft.idct(spfft.idct(Timage, norm='ortho', axis=0), norm='ortho', axis=1)
    
class WT(AbstractOperator):
    '''wavelet transform... under construction'''
    def __init__(self, shape, wavelet = 'db5', level = 6):
        self.shape = shape
        self.wavelet = wavelet
        self.level = level
    
    def apply(self, image):
        coeffs = pywt.wavedec2(imArray, wavelet=wavelet, level=level)
        #to list of np.arrays
        C = [coeffs[0]]
        for c in coeffs[1:]:
            C=C+list(c)
        self.wavelet_shapes = map(np.shape,C)
        #vectorize elements in C
        #...
        pass

        
    def inv(self,Timage):
        pywt.waverec2( Coeff, wavelet);
        #...
        #back to level format
        Coeff=[C[0]]
        for j in xrange(level):
            Coeff = Coeff+[tuple(C[3*j+1:3*(j+1)+1])]
        #...
        pass
        
def pltPic(X):
    plt.figure(figsize=(9,12))
    plt.imshow(X,interpolation='nearest', cmap=plt.cm.gray)
    plt.show()

def cL(s,x):
    '''returns n-s abs-smallest inices of vector x'''
    ns = len(x)-s
    return np.argpartition(abs(x),ns)[:ns]    

def thresholdingOperator(s,x):
    '''takes vector x, returns hard thresholded vector'''
    x[cL(s,x)] = 0
    return x

def compress(T,s,image):
    '''returns compressed image by keeping the s largest coeffcients in dictionary T'''
    x = T(image)
    x = thresholdingOperator(s,x)
    Cimage = T.inv(x)
    # print error
    print("Relative error: {}".format( la.norm(Cimage-image,'fro')/la.norm(image,'fro') ))
    return Cimage

def getRandMask(N,m):
    '''Random sample of m indices in range(N)'''
    return np.random.choice(N, m, replace=False)

def update(T, s, mask, Xsub, X, mu):
    '''IHT-type update, returns updated matrix Xnew and T-support of Xnew ''' 
    Xm = np.zeros(X.shape)
    Xm.flat[mask] = X.flat[mask]
    
    #calc gradient of squared L2-norm
    grad = 2*(Xm-Xsub)
    norm_grad = la.norm(grad.flat)
    
    #gradient step, transform
    TXnew = T( X-mu*grad )
    
    #partition according to s largest values
    ns = len(TXnew)-s
    part = np.argpartition(abs(TXnew),ns)
    support = part[-s:]
    cSupport = part[:ns]
    
    #threshold
    TXnew[cSupport]=0
    
    Xnew = T.inv(TXnew)
    return (Xnew, norm_grad, support)

def estimate(T, s, mask, Xsub, stepsize = 1, n_steps = 50, X0=None):
    '''IHT-type estimate
    
    :param T: transfrom on pictures, e.g., DCT
    :param s: expected sparsity
    :param mask: np.array of indices of Xsub.flat, i.e., Xsub[mask]==0
    :param X0: original picture to output the relative error'''
    #learning rate
    mu = stepsize #/np.sqrt(np.sum(mask))
    X = Xsub
    norm0 = la.norm(Xsub,'fro')
    
    if isinstance(X0,np.ndarray):
        print("Relative error (support change): {:3.2f}".format( la.norm(X-X0,'fro')/la.norm(X0,'fro') ), end = ', ')
    else:
        print("Support change: ")
    for j in xrange(n_steps):
        #update
        X, norm_grad, support = update(T, s, mask, Xsub, X, mu)
        if j>=1:
            support_diff = np.sum( support == last_support )
            print(' ({})'.format(support_diff),end ='')
        elif isinstance(X0,np.ndarray):
            print('(-)', end='')
        last_support = support
        if isinstance(X0,np.ndarray):
            rel_error = la.norm(X-X0,'fro')/la.norm(X0,'fro')
            if rel_error>10: break
            print(", {:3.2f}".format( rel_error ), end = '')            
        #interrupt if diverging
        elif la.norm(X,'fro')> norm0*np.prod(T.shape)/np.sqrt(s):
            break            
    return X

def rand_ux(N,s):
    ux = np.random.uniform(0,255,N)
    mask = np.random.choice(N, N-s, replace=False) # random sample of indices
    ux[mask] = 0
    return ux

def randomPic(T,s):
    '''generates a random picture, s-sparse in T-space'''
    shape = T.shape
    n = np.prod(shape)
    return T.inv( rand_ux(n,s).reshape(shape) )
