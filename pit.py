from __future__ import print_function
from __future__ import division
import numpy as np
import numpy.linalg as la
import numbers
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
        Timage = spfft.dct(spfft.dct(image, norm='ortho', axis=0, overwrite_x=True), norm='ortho', axis=1, overwrite_x=True)
        return Timage.reshape(-1)
    
    def inv(self, Timage):
        Timage = Timage.reshape(self.shape)
        return spfft.idct(spfft.idct(Timage, norm='ortho', axis=0), norm='ortho', axis=1)
    
class WT(AbstractOperator):
    '''wavelet transform... under construction'''
    def __init__(self, shape, wavelet = 'db6', level = 5, amplify = None):
        self.shape = shape
        self.wavelet = wavelet
        self.level = level
        self.cMat_shapes = [] 
        #build amplification vector of length 3*level
        if amplify is None:
            self.amplify = np.ones(3*self.level+1)
        else:
            self.amplify = amplify
        if isinstance(amplify, numbers.Number):
            self.amplify = np.ones(3*self.level+1)
            self.amplify[0] = amplify       
    
    def __call__(self, image):
        coeffs = pywt.wavedec2(image, wavelet=self.wavelet, level=self.level)
        # format: [cAn, (cHn, cVn, cDn), ...,(cH1, cV1, cD1)] , n=level

        #to list of np.arrays
        #multiply with self.amplify[0] to have them more strongly weighted in compressions
        #tbd: implement others
        cMat_list = [coeffs[0]]
        for c in coeffs[1:]:
            cMat_list = cMat_list + list(c)
        #memorize all shapes for inv
        self.cMat_shapes = list(map(np.shape,cMat_list))
        
        #array vectorization
        vect = lambda array: np.array(array).reshape(-1)
        
        #store coeffcient matrices as vectors in list
        #cVec_list = map(vect,cMat_list)
        
        #apply amplification
        cVec_list = [vect(cMat_list[j])*self.amplify[j] for j in range(3*self.level+1)]
            
        return np.concatenate(cVec_list)
    
    def inv(self,wavelet_vector):
        '''Inverse WT
            cVec_list: vector containing all wavelet coefficients as vectrized in __call__'''
        
        #check if shapes of the coefficient matrices are known
        if self.cMat_shapes == []:
            print("Call WT first to obtain shapes of coefficient matrices")
            return None
        
        cVec_shapes = list(map(np.prod,self.cMat_shapes))
        
        split_indices = list(accumulate(cVec_shapes))
        
        cVec_list = np.split(wavelet_vector,split_indices)
        
        #reverse amplification
        cVec_list = [cVec_list[j]/self.amplify[j] for j in range(3*self.level+1)]

        #back to level format
        coeffs=[ np.reshape(cVec_list[0],self.cMat_shapes[0]) ]
        for j in range(self.level):
            triple = cVec_list[3*j+1:3*(j+1)+1]
            triple = [np.reshape( triple[i], self.cMat_shapes[1 +3*j +i] ) 
                     for i in range(3) ]
            coeffs = coeffs + [tuple(triple)]

        return pywt.waverec2( coeffs, wavelet=self.wavelet )
        
def pltPic(X, size = (9,12) ):
    plt.figure(figsize=size)
    plt.imshow(X,interpolation='nearest', cmap=plt.cm.gray)
    plt.show()

def cL(s,x):
    '''returns n-s abs-smallest indices of vector x'''
    ns = len(x)-s
    return np.argpartition(abs(x),ns)[:ns]    

class hardTO(object):
    '''Hard thresholding operator:
            takes vector x, returns hard thresholded vector'''
    def __init__(self,sparsity):
        '''s: sparsity (integer number)'''
        self.s = sparsity
        
    def __call__(self,x):
        x[cL(self.s,x)] = 0
        return x
    
class softTO(object):
    '''Soft thresholding operator:
        takes vector x, returns hard thresholded vector'''
    def __init__(self,tau):
        '''tau>0: thresholding parameter'''
        self.tau = tau
    def __call__(self,x):
        return pywt.threshold(x, self.tau, mode='soft')
    
def compress(T, TO, image):
    '''returns compressed image by appyling thresholding to coeffcients in dictionary T:
    T: transformation taking image to vector, subclass of AbstractOperator
    thresholding = (H,thresholding_parameter): 
        H(v,thresholding_parameter) gives a vector for a vector v
    image: matrix of black-white values'''
    x = T(image)
    x = TO(x)
    Cimage = T.inv(x)
    # print error
    rel_error = la.norm(Cimage-image,'fro')/la.norm(image,'fro')
    print("Relative error: {}".format( rel_error ))
    return Cimage

def getRandMask(N,m):
    '''Random sample of m indices in range(N)'''
    return np.random.choice(N, m, replace=False)

def update(T, thOp, mask, Xsub, X, mu):
    '''IHT-type update, returns updated matrix Xnew and T-support of Xnew
    T: transform
    TO: thresholding operator
    mask: indices with unknown pixels
    Xsub: image matrix with Xsub[mask] arbitrary
    mu: step size'''
    
    Xm = np.zeros(T.shape)
    Xm.flat[mask] = X.flat[mask]
    
    #calc gradient of squared L2-norm
    grad = 2*(Xm-Xsub)
    norm_grad = la.norm(grad.flat)
    
    #gradient step, transform
    TXnew = T( X-mu*grad )
          
    #threshold
    TXnew = thOp(TXnew)
    
    #calculate support
    support = TXnew==0
    
    return ( T.inv(TXnew), norm_grad, support )


def estimate(T, thOp, mask, Xsub, stepsize = 1, n_steps = 100, X0=None, Xorig = None):
    '''IHT-type estimate
    
    :param T: transfrom on pictures, e.g., DCT
    :param s: expected sparsity
    :param mask: np.array of indices of Xsub.flat, i.e., Xsub[mask]==0
    :param X0: original picture to output the relative error'''
    #learning rate
    mu = stepsize #/np.sqrt(np.sum(mask))
    if X0 is None:
        X = Xsub
    else:
        X = X0
    last_support = T(X)==0
            
    # for checking divergence later
    norm0 = la.norm(Xsub,'fro')
    
    if isinstance(Xorig,np.ndarray):
        print("Relative error (support change): {:3.3f}".format( la.norm(X-Xorig,'fro')/la.norm(Xorig,'fro') ), end = ', ')
    else:
        print("Support change: ")
            
    for j in range(n_steps):
        #update
        X, norm_grad, support = update(T, thOp, mask, Xsub, X, mu)
        #set negative values to zero
        #X = pywt.threshold(X, 0, mode='greater', substitute = 0)
        X = proj2range(X)
        #print output
        if j % 10 == 0:
            #output support diff size
            support_diff = np.sum( support == last_support )
            print(' ({})'.format(len( support)-support_diff ),end ='')
            last_support = support
            # print error if original picture is provided
            if isinstance(Xorig,np.ndarray):
                rel_error = la.norm(X-Xorig,'fro')/la.norm(Xorig,'fro')
                if rel_error>10: break
                print(", {:3.3f}".format( rel_error ), end = '')            
            #interrupt if diverging
            elif la.norm(X,'fro')> 10*norm0*np.sqrt( np.prod(T.shape)/len(mask) ):
                break    
    print(' ')
    return X

def proj2range(X):
    '''Projects array elements to interval [0,255]'''
    X = pywt.threshold(X, 255, mode='less', substitute = 255)
    X = pywt.threshold(X, 0, mode='greater', substitute = 0)
    return X

def rand_ux(N,s):
    ux = np.random.uniform(0,255,N)
    mask = np.random.choice(N, N-s, replace=False) # random sample of indices
    ux[mask] = 0
    return ux

def randomPic(T,s):
    '''generates a random picture, s-sparse in T-space'''
    shape = T.shape
    _ = T(np.zeros(shape))
    n = np.prod(shape)
    return T.inv( rand_ux(n,s).reshape(shape) )
