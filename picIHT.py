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

try: 
    from itertools import accumulate 
except:
    import operator
    def accumulate(iterable, func=operator.add):
        'Return running totals'
        # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
        # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
        it = iter(iterable)
        try:
            total = next(it)
        except StopIteration:
            return
        yield total
        for element in it:
            total = func(total, element)
            yield total



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
    def __init__(self, shape, wavelet = 'db6', level = 5, amplify = [1]):
        self.shape = shape
        self.wavelet = wavelet
        self.level = level
        self.cMat_shapes = [] 
        self.amplify = amplify + (level-len(amplify)+1)*[1]
    
    def __call__(self, image):
        coeffs = pywt.wavedec2(image, wavelet=self.wavelet, level=self.level)
        # format: [cAn, (cHn, cVn, cDn), ...,(cH1, cV1, cD1)] , n=level

        #to list of np.arrays
        #multiply with self.amplify[0] to have them more strongly weighted in compressions
        #tbd: implement others
        cMat_list = [self.amplify[0]*coeffs[0]]
        for c in coeffs[1:]:
            cMat_list = cMat_list + list(c)
        #memorize all shapes for inv
        self.cMat_shapes = map(np.shape,cMat_list)
        
        #array vectorization
        vect = lambda array: np.array(array).reshape(-1)
        
        #store coeffcient matrices as vectors in list
        cVec_list = map(vect,cMat_list)

        return np.concatenate(cVec_list)
    
    def inv(self,wavelet_vector):
        '''Inverse WT
            cVec_list: vector containing all wavelet coefficients as vectrized in __call__'''
        
        #check if shapes of the coefficient matrices are known
        if self.cMat_shapes == []:
            print("Call WT first to obtain shapes of coefficient matrices")
            return None
        
        cVec_shapes = map(np.prod,self.cMat_shapes)
        
        split_indices = list(accumulate(cVec_shapes))
        
        cVec_list = np.split(wavelet_vector,split_indices)

        #back to level format
        coeffs=[ np.reshape(cVec_list[0]/self.amplify[0],self.cMat_shapes[0]) ]
        for j in xrange(self.level):
            triple = cVec_list[3*j+1:3*(j+1)+1]
            triple = [np.reshape( triple[i], self.cMat_shapes[1 +3*j +i] ) 
                     for i in xrange(3) ]
            coeffs = coeffs + [tuple(triple)]

        return pywt.waverec2( coeffs, wavelet=self.wavelet )
        
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
