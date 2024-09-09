import numpy as np 

from seemps.expectation import MPS, scprod
from seemps.state import product_state

#Probability amplitude computation
################################################################
def amplitude(bitstring,mpsstate,norm_mps=1):
    if norm_mps is None:
        norm_mps = MPS.norm(mpsstate) 
    zero_one = [[1,0],[0,1]]
    bitstate = []
    for bit in bitstring:
        bitstate.append(zero_one[int(bit)])
    bit_mpsstate = product_state(bitstate)
    return scprod(bit_mpsstate,mpsstate) / norm_mps

def probability_amplitude(bitstring,mpsstate,norm_mps=1):
    return abs(amplitude(bitstring,mpsstate,norm_mps=norm_mps))**2 

#Sampling
################################################################
def sampling(mpsstate,direction='left',size=None,print_info=False):
    "https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.031012 Sec.IIC"
    if size is None:
        size=1
    samples = np.zeros((mpsstate.size,size), dtype=int)
    for s in range(size):
        Xpreviousbits = np.array([1])
        prob_previousbits = 1
        if direction=='right': #right-canonical
            l = np.arange(mpsstate.size)
        elif direction=='left': #left-canonical
            l = np.arange(mpsstate.size-1,-1,-1)
        for en,i in enumerate(l):
            A = mpsstate._data[i]
            if direction=='right': 
                X = [np.einsum('i,ik->k', Xpreviousbits, A[:,0,:]),np.einsum('i,ik->k', Xpreviousbits, A[:,1,:])]
            elif direction=='left':
                X = [np.einsum('ki,i->k', A[:,0,:], Xpreviousbits),np.einsum('ki,i->k', A[:,1,:], Xpreviousbits)]   
            prob = [np.vdot(X[0],X[0]).real,np.vdot(X[1],X[1]).real]
            if not en:
                Z = np.sum(prob)
                prob_norm = prob/Z
                prob1_cond = round(prob_norm[1] / prob_previousbits, 12) #We round to avoid floating errors
            else:
                prob1_cond = round(prob[1] / prob_previousbits,12) 
            outcome = np.random.choice(np.array([0,1]),size=1,p=np.array([1-prob1_cond,prob1_cond]))[0]
            samples[i,s] = outcome
            Xpreviousbits = X[outcome]
            prob_previousbits = prob[outcome]
            if print_info:
                print(prob1_cond, outcome, prob_previousbits)
    return samples