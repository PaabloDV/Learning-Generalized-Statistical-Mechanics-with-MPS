import numpy as np
from opt_einsum import contract

class MPS:

    Sz = np.array([[1,0],[0,-1]])

    def __init__(self, N, d_bond, d_phys=2, A_mpsstate=None, x=None):
        """Create an MPS with 'N' sites, bond dimension 'd_bond', and physical dimension 'd_phys' from an array 'x' of parameters 
        or an array of tensors. Either 'x' or 'A_mpsstate' must be specified.
        If A_mpsstate is specified, it is an array or list of the tensors of the MPS sites."""
    
        self.N = N 
        self.d_bond = d_bond 
        self.d_phys = d_phys
        
        if A_mpsstate is None:
            self.x = x
            self.A_mpsstate = self.calculateMPS() 
        else:
            self.A_mpsstate = A_mpsstate

    def calculateMPS(self, shapes=None):
        "Build MPS's tensors from array of parameters"
        if shapes is None:
            tensors = [self.x[:self.d_bond*self.d_phys].reshape(1,self.d_phys,self.d_bond)]
            for n in range(0,self.N-2):
                tensors.append(self.x[(self.d_bond*self.d_phys)+(self.d_bond*self.d_phys*self.d_bond)*n:(self.d_bond*self.d_phys)+(self.d_bond*self.d_phys*self.d_bond)*(n+1)].reshape(self.d_bond,self.d_phys,self.d_bond))
            tensors.append(self.x[-(self.d_bond*self.d_phys):].reshape(self.d_bond,self.d_phys,1))
        else:
            tensors = [] 
            k = 0 #number of parameters already included
            for shape in shapes:
                x_aux = self.x[k:k+(shape[0]*shape[1]*shape[2])] 
                tensors.append(x_aux.reshape(shape[0],shape[1],shape[2]))
                k+=shape[0]*shape[1]*shape[2] 
        return tensors
    
    #Environments (iterative contractions)
    ################################################################
    def forward_contraction_q(self):
        C = np.ones((1,1)) 
        Cs = [C]
        for n in range(self.N-1):
            C = contract('xy,xsa,ysb -> ab', C,self.A_mpsstate[n],self.A_mpsstate[n])
            Cs.append(C)
        return Cs 

    def backward_contraction_q(self):
        C = np.ones((1,1)) 
        Cs = [C]
        for n in range(self.N-1,0,-1):
            C = contract('xy,asx,bsy -> ab', C,self.A_mpsstate[n],self.A_mpsstate[n])
            Cs.append(C)
        return Cs[::-1]

    def forward_contraction_q2(self):
        C = np.ones((1,1,1,1)) 
        Cs = [C]
        for n in range(self.N-1):
            C = contract('xyzw,xsa,ysb,zsc,wsd -> abcd', C,self.A_mpsstate[n],self.A_mpsstate[n],self.A_mpsstate[n],self.A_mpsstate[n],optimize='optimal')
            Cs.append(C)
        return Cs 

    def backward_contraction_q2(self):
        C = np.ones((1,1,1,1)) 
        Cs = [C]
        for n in range(self.N-1,0,-1):
            C = contract('xyzw,asx,bsy,csz,dsw -> abcd', C,self.A_mpsstate[n],self.A_mpsstate[n],self.A_mpsstate[n],self.A_mpsstate[n],optimize='optimal')
            Cs.append(C)
        return Cs[::-1]

    def forward_contraction_sk(self,J):
        Cs = []
        for i in range(self.N):
            for j in range(i+1,self.N):
                if J[i,j]!=0:
                    C = np.ones((1,1))
                    Cs_aux = [C]
                    for n in range(0,i):
                        Cs_aux.append(contract('xy,xsa,ysb -> ab', Cs_aux[-1],self.A_mpsstate[n],self.A_mpsstate[n]))
                    Cs_aux.append(contract('xy,xsa,sk,ykb -> ab', Cs_aux[-1],self.A_mpsstate[i],self.Sz,self.A_mpsstate[i])) 
                    for n in range(i+1,j):
                        Cs_aux.append(contract('xy,xsa,ysb -> ab', Cs_aux[-1],self.A_mpsstate[n],self.A_mpsstate[n]))
                    Cs_aux.append(contract('xy,xsa,sk,ykb -> ab', Cs_aux[-1],self.A_mpsstate[j],self.Sz,self.A_mpsstate[j]))
                    for n in range(j+1,self.N):
                        Cs_aux.append(contract('xy,xsa,ysb -> ab', Cs_aux[-1],self.A_mpsstate[n],self.A_mpsstate[n]))
                    Cs.append(Cs_aux)
        return Cs 

    def backward_contraction_sk(self,J):
        Cs = []
        for i in range(self.N):
            for j in range(i+1,self.N):
                if J[i,j]!=0:
                    C = np.ones((1,1))
                    Cs_aux = [C]
                    for n in range(self.N-1,j,-1):
                        Cs_aux.append(contract('xy,asx,bsy -> ab', Cs_aux[-1],self.A_mpsstate[n],self.A_mpsstate[n]))
                    Cs_aux.append(contract('xy,asx,sk,bky -> ab', Cs_aux[-1],self.A_mpsstate[j],self.Sz,self.A_mpsstate[j])) 
                    for n in range(j-1,i,-1):
                        Cs_aux.append(contract('xy,asx,bsy -> ab', Cs_aux[-1],self.A_mpsstate[n],self.A_mpsstate[n]))
                    Cs_aux.append(contract('xy,asx,sk,bky -> ab', Cs_aux[-1],self.A_mpsstate[i],self.Sz,self.A_mpsstate[i]))
                    for n in range(i-1,-1,-1):
                        Cs_aux.append(contract('xy,asx,bsy -> ab', Cs_aux[-1],self.A_mpsstate[n],self.A_mpsstate[n]))
                    Cs.append(Cs_aux[::-1])
        return Cs
    
    #Left and right environments approach (for the iterative Q2 algorithm)
    ################################################################

    def leftenvironment_q(self,k):
        "Perform and forward contractions from site 0 to site 'k'"
        C = np.ones((1,1)) 
        Cs = [C]
        for n in range(k):
            C = contract('xy,xsa,ysb -> ab', C,self.A_mpsstate[n],self.A_mpsstate[n])
            Cs.append(C)
        return Cs
    
    def rightenvironment_q(self,k):
        "Perform and backward contractions from site self.N-1 to site 'k'"
        C = np.ones((1,1)) 
        Cs = [C]
        for n in range(self.N-1,k,-1):
            C = contract('xy,asx,bsy -> ab', C,self.A_mpsstate[n],self.A_mpsstate[n])
            Cs.append(C)
        return Cs
    
    def leftenvironment_q2(self,k):
        "Perform and forward contractions from site 0 to site 'k'"
        C = np.ones((1,1,1,1)) 
        Cs = [C]
        for n in range(k):
            C = contract('xyzw,xsa,ysb,zsc,wsd -> abcd', C,self.A_mpsstate[n],self.A_mpsstate[n],self.A_mpsstate[n],self.A_mpsstate[n],optimize='optimal')
            Cs.append(C)
        return Cs 
    
    def rightenvironment_q2(self,k):
        "Perform and backward contractions from site self.N-1 to site 'k'"
        C = np.ones((1,1,1,1)) 
        Cs = [C]
        for n in range(self.N-1,k,-1):
            C = contract('xyzw,asx,bsy,csz,dsw -> abcd', C,self.A_mpsstate[n],self.A_mpsstate[n],self.A_mpsstate[n],self.A_mpsstate[n],optimize='optimal')
            Cs.append(C)
        return Cs
    
    def leftenvironment_sk(self,J,k):
        "Perform and forward contractions from site 0 to site 'k'"
        C = np.ones((1,1))
        Cs = [[C for i in range(self.N) for j in range(i+1,self.N) if J[i,j]!=0]]
        for n in range(k):
            Csn = []
            m=0
            for i in range(self.N):
                for j in range(i+1,self.N):
                    if J[i,j]!=0:
                        if n==i or n==j:
                            C_aux = contract('xy,xsa,sk,ykb -> ab', Cs[-1][m],self.A_mpsstate[n],self.Sz,self.A_mpsstate[n])
                        else:
                            C_aux = contract('xy,xsa,ysb -> ab', Cs[-1][m],self.A_mpsstate[n],self.A_mpsstate[n])
                        m+=1
                        Csn.append(C_aux)
            Cs.append(Csn)
        return Cs 
    
    def rightenvironment_sk(self,J,k):
        "Perform and backward contractions from site self.N-1 to site 'k'"
        C = np.ones((1,1))
        Cs = [[C for i in range(self.N) for j in range(i+1,self.N) if J[i,j]!=0]]
        for n in range(self.N-1,k,-1):
            Csn = []
            m=0
            for i in range(self.N):
                for j in range(i+1,self.N):
                    if J[i,j]!=0:
                        if n==i or n==j:
                            C_aux = contract('xy,asx,sk,bky -> ab', Cs[-1][m],self.A_mpsstate[n],self.Sz,self.A_mpsstate[n])
                        else:
                            C_aux = contract('xy,asx,bsy -> ab', Cs[-1][m],self.A_mpsstate[n],self.A_mpsstate[n])
                        m+=1
                        Csn.append(C_aux)
            Cs.append(Csn)
        return Cs
    
    
    def rightenvironment_sk_alternative(self,J,k):
        "Perform and backward contractions from site self.N-1 to site 'k'"
        Cs = []
        for i in range(self.N):
            for j in range(i+1,self.N):
                if J[i,j]!=0:
                    C = np.ones((1,1))
                    Cs_aux = [C]
                    for n in range(self.N-1,max(j,k),-1):
                        Cs_aux.append(contract('xy,asx,bsy -> ab', Cs_aux[-1],self.A_mpsstate[n],self.A_mpsstate[n]))
                    if j>k:
                        Cs_aux.append(contract('xy,asx,sk,bky -> ab', Cs_aux[-1],self.A_mpsstate[j],self.Sz,self.A_mpsstate[j])) 
                        for n in range(j-1,max(i,k),-1):
                            Cs_aux.append(contract('xy,asx,bsy -> ab', Cs_aux[-1],self.A_mpsstate[n],self.A_mpsstate[n]))
                        if i>k:
                            Cs_aux.append(contract('xy,asx,sk,bky -> ab', Cs_aux[-1],self.A_mpsstate[i],self.Sz,self.A_mpsstate[i]))
                            for n in range(i-1,k,-1):
                                Cs_aux.append(contract('xy,asx,bsy -> ab', Cs_aux[-1],self.A_mpsstate[n],self.A_mpsstate[n]))
                            Cs.append(Cs_aux[::-1])
                        else:
                            Cs.append(Cs_aux[::-1])
                    else:
                        Cs.append(Cs_aux[::-1]) 
        return Cs
    
    def update_leftenvironment_q(self,k,Ck,direction='right'):
        """Update the forward contractions from site 'k-1' to 'k' if direction=='right' or from 'k+1' to 'k' if direction=='left'.
            Ck is a list of tensors."""
        if direction=='right':
            Ck.append(contract('xy,xsa,ysb -> ab', Ck[-1],self.A_mpsstate[k-1],self.A_mpsstate[k-1]))
            return Ck
        elif direction=='left':
            return Ck[:-1]
        
    def update_rightenvironment_q(self,k,Ck,direction='right'):
        """Update the backward contractions from site 'k-1' to 'k' if direction=='right' or from 'k+1' to 'k' if direction=='left'.
        Ck is a list of tensors."""
        if direction=='left':
            Ck.append(contract('xy,asx,bsy -> ab', Ck[-1],self.A_mpsstate[k+1],self.A_mpsstate[k+1]))
            return Ck
        elif direction=='right':
            return Ck[:-1]
        
    def update_leftenvironment_q2(self,k,Ck,direction='right'):
        """Update the forward contractions from site 'k-1' to 'k' if direction=='right' or from 'k+1' to 'k' if direction=='left'.
            Ck is a list of tensors."""
        if direction=='right':
            Ck.append(contract('xyzw,xsa,ysb,zsc,wsd -> abcd', Ck[-1],self.A_mpsstate[k-1],self.A_mpsstate[k-1],self.A_mpsstate[k-1],self.A_mpsstate[k-1],optimize='optimal'))
            return Ck
        elif direction=='left':
            return Ck[:-1]
        
    def update_rightenvironment_q2(self,k,Ck,direction='right'):
        """Update the backward contractions from site 'k-1' to 'k' if direction=='right' or from 'k+1' to 'k' if direction=='left'.
            Ck is a list of tensors."""
        if direction=='left':
            Ck.append(contract('xyzw,asx,bsy,csz,dsw -> abcd', Ck[-1],self.A_mpsstate[k+1],self.A_mpsstate[k+1],self.A_mpsstate[k+1],self.A_mpsstate[k+1],optimize='optimal'))
            return Ck
        elif direction=='right':
            return Ck[:-1]
        
    def update_leftenvironment_sk(self,J,k,Ck,direction='right'):
        """Update the forward contractions from site 'k-1' to 'k' if direction=='right' or from 'k+1' to 'k' if direction=='left'.
            Ck is a list of lists of tensors."""
        if direction=='right':
            newCs = []
            m=0
            for i in range(self.N):
                for j in range(i+1,self.N):
                    if J[i,j]!=0:
                        if direction=='right':
                            if (k-1)==i or (k-1)==j:
                                newCs.append(contract('xy,xsa,sk,ykb -> ab', Ck[-1][m],self.A_mpsstate[k-1],self.Sz,self.A_mpsstate[k-1]))
                            else:
                                newCs.append(contract('xy,xsa,ysb -> ab', Ck[-1][m],self.A_mpsstate[k-1],self.A_mpsstate[k-1]))
                            m+=1
            Ck.append(newCs)
            return Ck
        elif direction=='left':
            return Ck[:-1] 
        
    def update_rightenvironment_sk(self,J,k,Ck,direction='right'):
        """Update the forward contractions from site 'k-1' to 'k' if direction=='right' or from 'k+1' to 'k' if direction=='left'.
            Ck is a list of lists of tensors."""
        if direction=='left':
            newCs = []
            m=0
            for i in range(self.N):
                for j in range(i+1,self.N):
                    if J[i,j]!=0:
                            if (k+1)==i or (k+1)==j:
                                newCs.append(contract('xy,asx,sk,bky -> ab', Ck[-1][m],self.A_mpsstate[k+1],self.Sz,self.A_mpsstate[k+1]))
                            else:
                                newCs.append(contract('xy,asx,bsy -> ab', Ck[-1][m],self.A_mpsstate[k+1],self.A_mpsstate[k+1]))
                            m+=1
            Ck.append(newCs)
            return Ck
        elif direction=='right':
            return Ck[:-1] 