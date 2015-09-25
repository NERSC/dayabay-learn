# Methods for rotating daya bay data by Seyoon.
import numpy as np

def get_stats_rot(X,posvec=np.tile(range(0,24),8), posvecsq=np.tile(range(0,24),8)**2):
    '''
    X: minibatch of data(192*minibatchsize)
    posvec: (rotated) x-coordinate 
    posvecsq: (rotated) sqared x-coordinate
    '''
    sumprod = np.dot(posvec,X)
    sumprodsq = np.dot(posvecsq,X)
    sums = np.sum(X, axis=0)
    sums[sums==0] = 1 # avoid division by zero
    msq = sumprodsq/sums
    mean = sumprod/sums
    dist = np.abs(mean-11.)
    var = msq-mean**2
    return dist, var
    


def get_rot_amount(X, eps=1e-6):
    '''
    X: minibatch of data(192*minibatchsize)
    '''
    posvec  = np.tile(range(0,24),8).astype(float)
    posvecsq = posvec**2
    bestidx = np.zeros(X.shape[1]).astype(int)
    bestdist, bestvar = get_stats_rot(X, posvec, posvecsq)
    
    for i in xrange(23):
        posvec = np.roll(posvec,1)
        posvecsq = np.roll(posvecsq,1)
        dist, var = get_stats_rot(X,posvec,posvecsq)
        isbest = np.logical_or(var<bestvar-eps, np.logical_and(var<bestvar+eps,dist<bestdist ))
        bestidx[isbest] = i+1
   
    return -bestidx

def get_rot_amount_maxelem(X):
    '''
    X: minibatch of data(192*minibatchsize)
    '''
    return 11-(np.argmax(X, axis=0) % 24)
def get_rot_amount_maxcol(X):
    '''
    X: minibatch of data(192*minibatchsize)
    '''
    shape = X.shape
    tmp = X.reshape(8,24,(shape[1]))
    colsum = tmp.sum(axis=0)
    #print colsum.shape
    maxidx = np.argmax(colsum,axis=0)
    #print maxidx
    return 11-maxidx

def rotate(X, amount):
    '''
    X: minibatch of data(192*minibatchsize)
    amount(1-dim array): amount to rotate, for each datapoint.
    '''
    datapts=X.shape[1]
    for i in xrange(datapts):
        #print get_center_of_mass(X[:,i])
        tmp = X[:,i]
        tmp.shape = [8,24]
        #print tmp
        #print amount[i]
        tmp = np.roll(tmp, amount[i], axis=1)
        #print tmp
        tmp.shape=[192]
        X[:,i] = tmp
        #print get_center_of_mass(X[:,i])
    return X


