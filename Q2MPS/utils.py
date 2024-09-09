import math
import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt 

#General utils
#######################################################
def all_bit_strings(N):
    """Return a matrix of shape (2**N, N) of all bit strings that
    can be constructed using 'N' bits. Each row is a different
    configuration, corresponding to the integers 0, 1, 2 up to (2**N)-1"""
    confs = np.arange(2 ** N, dtype=np.int32)
    return np.array([(confs >> i) & 1 for i in range(N)], dtype=np.uint32)

#SK utils
###############################################################
def createJ_SK(N, sigma=1, plot=False):
    J = np.random.normal(loc=0, scale=sigma, size=(N, N))
    J = np.triu(J, k=1) + np.triu(J, k=1).T
    if plot:
        plt.imshow(J)
        plt.grid()
        plt.colorbar()
        plt.show()
    return J/math.sqrt(N)

def createJ_SK_regulargraph(N, degree, sigma=1, plot=False):
    W = nx.to_numpy_array(nx.random_regular_graph(d=min(degree, N), n=N))
    J = W * np.random.normal(loc=0, scale=sigma, size=(N, N))
    J = np.triu(J, k=1) + np.triu(J, k=1).T
    if plot:
        plt.imshow(J)
        plt.grid()
        plt.colorbar()
        plt.show()
    return J/math.sqrt(degree)

def SKenergy(bits,J):
    spins = 2 * bits.astype(int) - 1
    return ((J @ spins) * spins).sum(0) / 2

def uniformsample_energies(N,J,nsamples):
    bits = np.random.choice(np.array([0,1]), size=(N,nsamples))
    return SKenergy(bits,J)/N 

#Q2algorithm
##########################################################
def q_optimal(E,β,N,τ0=None,tol=10):
    nstates = len(E) #to improve the calculation of tau we use the number of states with non-zero probability amplitude
    if τ0 is None:
        τ = 0
    else:
        τ = τ0 
    τ_hist = []
    q = -(β/2)*E + τ
    q_copy = np.copy(q) #we save the distribution before cutting to 0
    q[np.where((q<0))]=0
    while np.round(np.sum(q),tol)!=1:
        τ = (1 - np.sum(q))/nstates
        q = q_copy + τ
        q_copy = np.copy(q) #we save the distribution before cutting to 0
        q[np.where((q<0))]=0
        τ_hist.append(τ)
    F = np.sum(q*E) - (1/β)*(1-np.sum(q**2))
    return q, sum(τ_hist), F

#Tsallis Statistics
##########################################################
def tau_equation(β,σ,N,τ):
    
    return 2**(N)*(
        (math.exp(-(2*τ**2)/(β*σ)**2) * β * σ )/(2*math.sqrt(2*math.pi)) +
        0.5 * τ * (1 + math.erf((math.sqrt(2)*τ)/(β*σ))) ) -1
    
def avE(β,σ,N,τ):
    
    return 2**(N)*-(1/4)*β*σ**2 * (
        1+math.erf(math.sqrt(2)*τ/(β*σ)) )

def entropy(β,σ,N,τ):
    #1-S    
    return 2**(N)*(
        (math.exp(-2*τ**2/(β*σ)**2)*β*σ*τ)/(2*math.sqrt(2*math.pi))
        + (1/8)*((β**2) * (σ**2) + 4*τ**2) * (1+math.erf(math.sqrt(2)*τ/(β*σ))) 
    )

def tau_equation_emin(β,σ,N,emin,τ):
    
    return 2**(N)*(
        ( -( math.exp(-(emin)**2/(2*(σ)**2)) - math.exp(-(2*τ**2)/(β*σ)**2) ) * β * σ )/(2*math.sqrt(2*math.pi)) +
        0.5 * τ * (-math.erf(emin/(math.sqrt(2)*σ)) + math.erf((math.sqrt(2)*τ)/(β*σ))) ) -1

def avE_emin(β,σ,N,emin,τ):
    
    return 2**(N)*(1/4)*σ * (
        math.exp(-(emin)**2/(2*(σ)**2)) * math.sqrt(2/math.pi) * (-emin*β+2*τ) + 
        β*σ*math.erf(emin/(math.sqrt(2)*σ)) -  β*σ*math.erf(math.sqrt(2)*τ/(β*σ)) )

def entropy_emin(β,σ,N,emin,τ):
    #1-S    
    return 2**(N)*(
        ( β*σ* (math.exp(-(emin)**2/(2*(σ)**2)) * (emin*β-4*τ) + 2*math.exp(-(2*τ**2)/(β*σ)**2)*τ)) / (4*math.sqrt(2*math.pi)) +
        + (1/8)*((β**2) * (σ**2) + 4*τ**2) * (-math.erf(emin/(math.sqrt(2)*σ))+math.erf(math.sqrt(2)*τ/(β*σ))) 
        )

def approx_ratio_emin(β,σ,N,emin,τ):

    return 1 - ((avE_emin(β,σ,N,emin,τ) - emin) / abs(emin))
