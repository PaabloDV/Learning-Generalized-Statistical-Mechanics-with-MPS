import numpy as np 
import scipy
import time 
from tqdm import tqdm
from opt_einsum import contract
#from functools import partial

from Q2MPS.MPS_utils import probability_amplitude, sampling
from Q2MPS.utils import all_bit_strings, SKenergy

from seemps.state import CanonicalMPS, Strategy
from seemps.state import MPS as seemMPS 

from Q2MPS.MPS_class import MPS 

class IterativeQ2algorithm:

    def __init__(self, J, d_bond, d_phys=2):
        """New MPS algorithm to solve combinatorial optimization problems:
            min(E) with E[s] = \sum_{ij} s[i] J[i,j] s[j] ; s={-1,+1} J[i,j]=J[j,i]
            we optimize one site of an MPS iteratively."""
        self.J = J
        self.N = len(J)
        self.d_bond = d_bond
        self.d_phys = d_phys

    ###Observables
    ###########################
    
    def av_e(self,A,n,left_environment_sk,right_environment_sk):
        """Calculate the average energy <A_mpsstate|E|A_mpsstate>"""
        ave = 0
        k=0
        for i in range(self.N):
            for j in range(i+1,self.N):
                if self.J[i,j]!=0:
                    if n==i or n==j:
                        ave+= self.J[i,j]*contract('xy,xsa,sk,ykb,ab -> ', left_environment_sk[-1][k],A,MPS.Sz,A,right_environment_sk[-1][k])
                    else:
                        ave+= self.J[i,j]*contract('xy,xsa,ysb,ab -> ', left_environment_sk[-1][k],A,A,right_environment_sk[-1][k]) 
                    k+=1
        return ave

    def wholegrad_ave(self,A,n,left_environment_sk,right_environment_sk):
        """Calculate the gradient of the average energy ∇(<A_mpsstate|E|A_mpsstate>)"""
        A_aux = np.ones(A.shape) 
        grad = np.zeros(A.shape)
        k=0
        for i in range(self.N):
            for j in range(i+1,self.N):
                if self.J[i,j]!=0:
                    if i==n or j==n:
                        C = contract('xy,xsa,sk,ykb -> aykb', left_environment_sk[-1][k],A,MPS.Sz,A_aux)
                    else:
                        C = contract('xy,xsa,ysb -> aysb', left_environment_sk[-1][k],A,A_aux)
                    grad += 2*self.J[i,j]*contract('aykb,ab -> ykb', C,right_environment_sk[-1][k])
                    k+=1
        return grad.reshape(-1)

    @staticmethod
    def av_q(A,left_environment_q2,right_environment_q2):
        """Calculate the average Q <A_mpsstate|Q|A_mpsstate> where 
            Q=\sum_{x} |P(x)><P(x)| with P(x) = |A_mpsstate(x)|^2/<A_mpsstate|A_mpsstate>"""
        return contract('xyzw,xsa,ysb,zsc,wsd,abcd ->  ',left_environment_q2[-1],A,A,A,A,right_environment_q2[-1],optimize='optimal')

    def wholegrad_avq(self,A,left_environment_q2,right_environment_q2):
        """Calculate the gradient of the average Q ∇(<A_mpsstate|Q|A_mpsstate>)"""
        A_aux = np.ones(A.shape) 
        grad = np.zeros(A.shape)
        C = contract('xyzw,xsa,ysb,zsc,wsd -> abcwsd',left_environment_q2[-1],A,A,A,A_aux,optimize='optimal')
        grad += 4*contract('abcwsd,abcd -> wsd', C,right_environment_q2[-1],optimize='optimal')
        return grad.reshape(-1)

    @staticmethod
    def norm(A,left_environment_q,right_environment_q):
        """Calculate the MPS norm square <A_mpsstate|A_mpsstate>"""
        return contract('xy,xsa,ysb,ab -> ',left_environment_q[-1],A,A,right_environment_q[-1])
        #return np.einsum('xy,xsa,ysb,ab -> ',left_environment_q[-1],A,A,right_environment_q[-1])

    def wholegrad_norm(self,A,left_environment_q,right_environment_q):
        """Calculate the gradient of the MPS norm ∇(<A_mpsstate|A_mpsstate>)"""
        A_aux = np.ones(A.shape) 
        grad = np.zeros(A.shape)
        C = np.einsum('xy,xsa,ysb -> aysb',left_environment_q[-1],A,A_aux)
        grad += 2*contract('aysb,ab -> ysb', C,right_environment_q[-1])
        return grad.reshape(-1)

    def observables(self,An,n,left_environment_q,right_environment_q,left_environment_q2,right_environment_q2,left_environment_sk,right_environment_sk):
        """We calculate all the observables involved in the algorithm given the tensor of the site n and their environments."""
        ##Observables for the loss function
        average_energy = self.av_e(An,n,left_environment_sk,right_environment_sk)
        average_q = self.av_q(An,left_environment_q2,right_environment_q2)
        norm_mps = self.norm(An,left_environment_q,right_environment_q)
        ##Observables for the gradient of the loss function
        grad_average_energy = self.wholegrad_ave(An,n,left_environment_sk,right_environment_sk)
        grad_average_q = self.wholegrad_avq(An,left_environment_q2,right_environment_q2)
        grad_norm = self.wholegrad_norm(An,left_environment_q,right_environment_q)

        return average_energy,average_q,norm_mps, grad_average_energy,grad_average_q,grad_norm 
    
    ###Loss function and derivative
    ##################################
    def F_n(self,β, average_energy,average_q,norm_mps):
        """Calculate the new free energy:
            F = (<A_mpsstate|E|A_mpsstate>/(N*<A_mpsstate|A_mpsstate>)) + (1/β)*(<A_mpsstate|Q|A_mpsstate>/<A_mpsstate|A_mpsstate>^2)
            We divide the energy by the size of the system 'N'.""" 
        if β=='inf':
            return average_energy/(self.N*norm_mps)
        else:
            return (average_energy/(self.N*norm_mps)) + (1/β)* (average_q/norm_mps**2)

    def grad_Fn(self,β, average_energy,average_q,norm_mps, grad_average_energy,grad_average_q,grad_norm):
        """Calculate the gradient of the new free energy ∇(F)"""
        if β=='inf':
            return ((grad_average_energy*norm_mps-grad_norm*average_energy)/(self.N*norm_mps**2)) 
        else:
            return ((grad_average_energy*norm_mps-grad_norm*average_energy)/(self.N*norm_mps**2)) + (1/β)*((grad_average_q*norm_mps-2*grad_norm*average_q)/norm_mps**3)

    def cost_and_derivative(self,x, site_shape, β, n, leftenv_q,rightenv_q,leftenv_q2,rightenv_q2,leftenv_sk,rightenv_sk):
        """Calculate the loss function and the derivative given parameters 'x', MPS site 'n', and inverse temperature 'β'."""
        An = x.reshape(site_shape)
        average_energy,average_q,norm_mps, grad_average_energy,grad_average_q,grad_norm = self.observables(An, n, leftenv_q,rightenv_q,leftenv_q2,rightenv_q2,leftenv_sk,rightenv_sk)
        return self.F_n(β, average_energy,average_q,norm_mps), self.grad_Fn(β, average_energy,average_q,norm_mps, grad_average_energy,grad_average_q,grad_norm)
    

    ###Training
    ##################################

    def classical_solution(self,method='Simulated Annealing',**kwargs):
        """Calculate the minimum energy and ground state bits with other algorithms."""
        if method=='Simulated Annealing':
            pass
            #Not included in this version
            #mine, minbits = SA_solver(self.J, **kwargs) 
        elif method=='Exact':
            if self.N>22:
                raise Exception("The size of the problem is too large for the exact calculation of the minimum.")
            bits = all_bit_strings(self.N)
            E = SKenergy(bits,self.J)/self.N 
            mine = np.min(E)
            minbits = bits[:,np.where(np.round(E,10)==np.round(mine,10))][:,0,0]
        return mine, minbits
    
    @staticmethod
    def state_probability(statebits,mps,norm_mps=1):
        """Return the probability amplitude of a specific classical state 'statebits' in the MPS state 'mps'."""
        return probability_amplitude(statebits,mps.A_mpsstate,norm_mps=norm_mps)

    def initial_parameters(self,method='optimal'):
        """Create initial parameters for the annealing and optimization proccess."""
        if method=='random':
            #The parameters are randomly initizalized
            result_param = np.random.rand((self.N-2)*(self.d_bond**2)*self.d_phys + 2*self.d_bond*self.d_phys)*0.5
        elif method=='optimal':
            #The parameters are initizalized so that the MPS is close to the full-superposition state
            result_param = np.zeros((self.N-2)*(self.d_bond**2)*self.d_phys + 2*self.d_bond*self.d_phys)
            for n in range(self.d_phys):
                result_param[n*self.d_bond]=1/np.sqrt(2)
                for k in range(0,self.N-2):
                    result_param[n*self.d_bond+(self.d_bond*self.d_phys)+k*(self.d_bond*self.d_phys*self.d_bond)]=1/np.sqrt(2)
                result_param[-n-(self.d_bond-1)*self.d_phys-1]=1/np.sqrt(2)
            #We add some noise to make the algorithm work
            result_param += np.random.rand(len(result_param))*0.75*10**(-1)   
        else:
            raise Exception("Not known method for initial parameters.")

        return result_param
    
    def optimization_iteration(self, 
                               n, #site 
                               canonical_mps, #MPS in MPS_class object
                               seemps_canonicalmps, #MPS in seemps object
                               β, #inverse temperature
                               leftenv_q, rightenv_q, leftenv_q2, rightenv_q2, leftenv_sk, rightenv_sk, #environments
                               method, #optimizer
                               maxiter, #maximum number of iterations of the optimization method
                               ftol): #tolerance for the optimization method
        
        #Variables to optimize in this iteration
        site_shape = canonical_mps.A_mpsstate[n].shape
        xopt = canonical_mps.A_mpsstate[n].reshape(-1)
        #Perform optimization
        result = scipy.optimize.minimize(self.cost_and_derivative, xopt, jac=True, args=(site_shape, β, n, leftenv_q, rightenv_q, leftenv_q2, rightenv_q2, leftenv_sk, rightenv_sk)
                    , method=method, options={'maxiter': maxiter,'ftol':ftol})
        #We retrieve the optimization result
        newAn = result.x.reshape(site_shape)
        seemps_canonicalmps._data[n] = newAn
        canonical_mps = MPS(self.N, self.d_bond, A_mpsstate=seemps_canonicalmps._data) 

        return newAn, seemps_canonicalmps, canonical_mps    

    def sample_MPS(self, history, seemps_canonicalmps):
        seemps_canonicalmps = CanonicalMPS(seemps_canonicalmps._data) #To sample the MPS must be left-handed canonical
        samples = sampling(seemps_canonicalmps, direction='right',size=10000)
        E = SKenergy(samples,self.J)/self.N
        e_res = np.min(E)
        
        history['eres'] = e_res 
        history['sampledE'] = E 

        return history 

    def update_history(self, canonical_mps, history, n, leftenv_q,rightenv_q,leftenv_q2,rightenv_q2,leftenv_sk,rightenv_sk):
        average_energy,average_q,norm_mps, grad_average_energy,grad_average_q,grad_norm = self.observables(canonical_mps.A_mpsstate[n], n, leftenv_q,rightenv_q,leftenv_q2,rightenv_q2,leftenv_sk,rightenv_sk)
        β = history['beta'][-1]
        mine = history['minE']
        minbits = history['minbits']
        normJ = history['normJ']

        loss = self.F_n(β, average_energy,average_q,norm_mps)
        grad = self.grad_Fn(β, average_energy,average_q,norm_mps, grad_average_energy,grad_average_q,grad_norm)
        ave = average_energy/(self.N*norm_mps) 
        avq = average_q/(norm_mps**2)
        if mine==None:
            ener_relative_error = None 
            gs_prob = None
            deltaE = None
        else:
            ener_relative_error = (ave-mine)/abs(mine) 
            deltaE = (ave-mine)/normJ 
            #To calculate the ground state probability we take into account that in MAXCUT the string {s1,s2,s3,...} has the same energy as (-1)*{s1,s2,s3,...} 
            # for si={-1,+1}
            gs_prob = self.state_probability(minbits,canonical_mps,norm_mps=np.sqrt(norm_mps)) + self.state_probability(-1*minbits+1,canonical_mps,norm_mps=np.sqrt(norm_mps)) 

        history['loss'] += [loss]
        history['gradient norm'] += [np.linalg.norm(grad)]
        history['energy relative error'] += [ener_relative_error]
        history['average q'] += [avq]
        history['average e'] += [ave]
        history['deltaE'] += [deltaE]
        history['GS prob'] += [gs_prob]
        history['MPS Norm'] += [norm_mps]

        return history 

    def optimize(self,β,x0_method='optimal',x0_given=None,given_solution=None, maxiter=100, ftol=10**(-12), maxsweeps=100000, method='L-BFGS-B', tol=10**(-4),
                 do_sampling=False, avqtol=0.99, sample_at_end=False,print_info=True):
        """ Optimize the parameters of the MPS so that the new free energy with 'β' is minimum.
        WARNING: this function might not include updates made to the annealing function. """
    
        #Known problem solution
        if given_solution==None:
            mine = None
            minbits = None
        else:
            mine,minbits = given_solution
    
        normJ = np.linalg.norm(self.J) 

        #Dict to save results. We are not saving the parameters to save space
        history = {'loss':[], 'energy relative error':[], 'average q':[], 'average e':[], 'GS prob':[], 'MPS Norm':[], 'gradient norm':[], 'deltaE':[], 'eres':0, 'sampledE':[], 
               'parameters':[], 'minE':mine, 'minbits':minbits, 'beta':[β], 'N':self.N, 'J':self.J, 'normJ':normJ , 'd_bond':self.d_bond, 'd_phys':self.d_phys, 'maxiter':maxiter,
                'ftol':ftol, 'extime':[], 'sweep':[], 'swchange':[]}
    
        #Initial parameters
        if x0_given is None:
            x0 = self.initial_parameters(method=x0_method)
        else:
            x0 = x0_given

        #We initialize the MPS in the  Canonical form centered in n=0
        mps = MPS(self.N, self.d_bond, x=x0)
        seemps_canonicalmps = CanonicalMPS(mps.A_mpsstate,center=0, strategy=Strategy(method=0), normalize=True) #Strategy(method=0) to not truncate
        canonical_mps = MPS(self.N, self.d_bond, A_mpsstate=seemps_canonicalmps._data)
        #We initialize the environments
        leftenv_q = canonical_mps.leftenvironment_q(0)
        rightenv_q = canonical_mps.rightenvironment_q(0)
        leftenv_q2 = canonical_mps.leftenvironment_q2(0)
        rightenv_q2 = canonical_mps.rightenvironment_q2(0)
        leftenv_sk = canonical_mps.leftenvironment_sk(self.J,0)
        rightenv_sk = canonical_mps.rightenvironment_sk(self.J,0)
        #Training    
        avq = 0  
        sw = 0  
        losssw = 0
        while avq<=avqtol and sw<maxsweeps:
            if sw:
                if np.abs(losssw-history['loss'][-1])<tol: #We stop optimization
                    history['swchange'] += [sw-1] #We save in which sweep the beta changed for post-processing 
                    break
                losssw = history['loss'][-1]
            starttime = time.time()
            for n in tqdm(range(self.N), leave=False):
                direction = 'right'
                if n:
                    seemps_canonicalmps_reduced = CanonicalMPS(canonical_mps.A_mpsstate[n-1:n+1],center=1, strategy=Strategy(method=0), normalize=True)
                    #print(seemps_canonicalmps._data[n+1]-canonical_mps.A_mpsstate[n+1])
                    canonical_mps.A_mpsstate[n-1] = seemps_canonicalmps_reduced._data[0]
                    canonical_mps.A_mpsstate[n] = seemps_canonicalmps_reduced._data[1]
                    seemps_canonicalmps = seemMPS(canonical_mps.A_mpsstate)
                    #canonical_mps = MPS(self.N, self.d_bond, A_mpsstate=seemps_canonicalmps._data)
                    leftenv_q = canonical_mps.update_leftenvironment_q(n,leftenv_q,direction=direction)
                    rightenv_q = canonical_mps.update_rightenvironment_q(n,rightenv_q,direction=direction)
                    leftenv_q2 = canonical_mps.update_leftenvironment_q2(n,leftenv_q2,direction=direction)
                    rightenv_q2 = canonical_mps.update_rightenvironment_q2(n,rightenv_q2,direction=direction)
                    leftenv_sk = canonical_mps.update_leftenvironment_sk(self.J,n,leftenv_sk,direction=direction)
                    rightenv_sk = canonical_mps.update_rightenvironment_sk(self.J,n,rightenv_sk,direction=direction)

                newAn, seemps_canonicalmps, canonical_mps = self.optimization_iteration(n, canonical_mps, seemps_canonicalmps, β,
                                                                                        leftenv_q, rightenv_q, leftenv_q2, rightenv_q2, leftenv_sk, rightenv_sk,
                                                                                        method, maxiter, ftol)

                average_q = self.av_q(newAn,leftenv_q2,rightenv_q2)
                norm_mps = self.norm(newAn,leftenv_q,rightenv_q)
                avq = average_q/(norm_mps**2)
                if avq>avqtol:
                    #If we superate the tol on avq we sample the MPS
                    if do_sampling:
                        history = self.sample_MPS(history, seemps_canonicalmps) 
                    break

            if avq<=avqtol: 
                for n in tqdm(range(self.N-1,-1,-1), leave=False):
                    direction = 'left'
                    if n!=(self.N-1):
                        seemps_canonicalmps_reduced = CanonicalMPS(canonical_mps.A_mpsstate[n:n+2],center=0, strategy=Strategy(method=0), normalize=True)
                        canonical_mps.A_mpsstate[n] = seemps_canonicalmps_reduced._data[0]
                        canonical_mps.A_mpsstate[n+1] = seemps_canonicalmps_reduced._data[1]
                        seemps_canonicalmps = seemMPS(canonical_mps.A_mpsstate)
                        leftenv_q = canonical_mps.update_leftenvironment_q(n,leftenv_q,direction=direction)
                        rightenv_q = canonical_mps.update_rightenvironment_q(n,rightenv_q,direction=direction)
                        leftenv_q2 = canonical_mps.update_leftenvironment_q2(n,leftenv_q2,direction=direction)
                        rightenv_q2 = canonical_mps.update_rightenvironment_q2(n,rightenv_q2,direction=direction)
                        leftenv_sk = canonical_mps.update_leftenvironment_sk(self.J,n,leftenv_sk,direction=direction)
                        rightenv_sk = canonical_mps.update_rightenvironment_sk(self.J,n,rightenv_sk,direction=direction)

                    newAn, seemps_canonicalmps, canonical_mps = self.optimization_iteration(n, canonical_mps, seemps_canonicalmps, β,
                                                                                            leftenv_q, rightenv_q, leftenv_q2, rightenv_q2, leftenv_sk, rightenv_sk,
                                                                                            method, maxiter, ftol)

                    average_q = self.av_q(newAn,leftenv_q2,rightenv_q2)
                    norm_mps = self.norm(newAn,leftenv_q,rightenv_q)
                    avq = average_q/(norm_mps**2)
                    if avq>avqtol:
                        #If we superate the tol on avq we sample the MPS
                        if do_sampling:
                            history = self.sample_MPS(history, seemps_canonicalmps) 
                        break


            extime = time.time() - starttime
            history = self.update_history(canonical_mps, history, n, leftenv_q,rightenv_q,leftenv_q2,rightenv_q2,leftenv_sk,rightenv_sk)
            history['sweep'] += [sw]
            history['extime'] += [extime]
            sw+=1
            if print_info:
                print(f"Sweep:{sw}, Norm:{history['MPS Norm'][-1]} GSprob: {history['GS prob'][-1]} Avq: {history['average q'][-1]} ErelError: {history['energy relative error'][-1]}")

        if sample_at_end and history['eres']==0:
            #We sample at the end if sample_at_end==True and we did not sample before 
            history = self.sample_MPS(history, seemps_canonicalmps)
 
        print(f"Finished after {sw} sweeps with beta:{β} Norm:{history['MPS Norm'][-1]} GSprob: {history['GS prob'][-1]} Avq: {history['average q'][-1]} Eres: {history['eres']} Relative error: {(history['eres']-history['minE'])/abs(history['minE'])} Sweeps-MPSsite: {sw}-{n}")

        return history, canonical_mps 
    
    def annealing(self,β0,βf, method='L-BFGS-B',x0_method='optimal',x0_given=None, given_solution=None, maxiter=10, ftol=10**(-12), β_scheme='log', βs=None, tol=10**(-4)
                  ,steps=50, do_sampling=False, avqtol=0.01,swstep=2,print_info=True):
        """Optimize the parameters of the MPS by increasing the 'β' so that the average energy is minimum at the end of the proccess."""
    
        #Known problem solution
        if given_solution==None:
            mine = None
            minbits = None
        else:
            mine, minbits = given_solution

        normJ = np.linalg.norm(self.J) 

        #Dict to save results. We are not saving the parameters to save space
        history = {'loss':[], 'energy relative error':[], 'average q':[], 'average e':[], 'GS prob':[], 'MPS Norm':[], 'gradient norm':[], 'deltaE':[], 'eres':0, 'sampledE':[], 
               'parameters':[], 'minE':mine, 'minbits':minbits, 'beta':[], 'N':self.N, 'J':self.J, 'normJ':normJ , 'd_bond':self.d_bond, 'd_phys':self.d_phys, 'maxiter':maxiter,
                'ftol':ftol, 'extime':[], 'sweep':[], 'swchange':[]}

        #Initial parameters
        if x0_given is None:
            x0 = self.initial_parameters(method=x0_method)
        else:
            x0 = x0_given

        #We initialize the MPS in the  Canonical form centered in n=0
        mps = MPS(self.N, self.d_bond, x=x0)
        seemps_canonicalmps = CanonicalMPS(mps.A_mpsstate,center=0, strategy=Strategy(method=0), normalize=True) #strategy=Strategy(method=0) to not truncate
        canonical_mps = MPS(self.N, self.d_bond, A_mpsstate=seemps_canonicalmps._data)
        #We initialize the environments
        leftenv_q = canonical_mps.leftenvironment_q(0)
        rightenv_q = canonical_mps.rightenvironment_q(0)
        leftenv_q2 = canonical_mps.leftenvironment_q2(0)
        rightenv_q2 = canonical_mps.rightenvironment_q2(0)
        leftenv_sk = canonical_mps.leftenvironment_sk(self.J,0)
        rightenv_sk = canonical_mps.rightenvironment_sk(self.J,0)
        #Annealing scheme
        if β_scheme=='log': 
            βs = np.logspace(β0, βf, steps)
        elif β_scheme=='linear':
            βs = np.linspace(β0, βf, steps)   
        elif β_scheme=='original':
            βs = βs
            steps = len(βs)
        else:
            raise Exception("Unknown beta_scheme. Available schemes: 'original','log' and 'linear'.")
        #Training  
        i = 0 
        avq = 0  
        sw = 0  
        losssw = 0
        while avq<=avqtol: #and sw<(steps*swstep):
            if sw:
                if np.abs(losssw-history['loss'][-1])<tol:#sw%swstep==0: #If we change beta, we print the current results 
                    if print_info:
                        print(f"Finished with beta:{βs[i]} Norm:{history['MPS Norm'][-1]} GSprob: {history['GS prob'][-1]} Avq: {history['average q'][-1]} ErelError: {history['energy relative error'][-1]} Sweep: {sw}")
                    i+=1
                    history['swchange'] += [sw-1] #We save in which sweep the beta changed for post-processing 
                losssw=history['loss'][-1] 
            if i>=steps:
                break

            starttime = time.time()
            for n in tqdm(range(self.N), leave=False):
                direction = 'right'
                if n:
                    seemps_canonicalmps_reduced = CanonicalMPS(canonical_mps.A_mpsstate[n-1:n+1],center=1, strategy=Strategy(method=0), normalize=True)
                    canonical_mps.A_mpsstate[n-1] = seemps_canonicalmps_reduced._data[0]
                    canonical_mps.A_mpsstate[n] = seemps_canonicalmps_reduced._data[1]
                    seemps_canonicalmps = seemMPS(canonical_mps.A_mpsstate)
                    leftenv_q = canonical_mps.update_leftenvironment_q(n,leftenv_q,direction=direction)
                    rightenv_q = canonical_mps.update_rightenvironment_q(n,rightenv_q,direction=direction)
                    leftenv_q2 = canonical_mps.update_leftenvironment_q2(n,leftenv_q2,direction=direction)
                    rightenv_q2 = canonical_mps.update_rightenvironment_q2(n,rightenv_q2,direction=direction)
                    leftenv_sk = canonical_mps.update_leftenvironment_sk(self.J,n,leftenv_sk,direction=direction)
                    rightenv_sk = canonical_mps.update_rightenvironment_sk(self.J,n,rightenv_sk,direction=direction)
                
                newAn, seemps_canonicalmps, canonical_mps = self.optimization_iteration(n, canonical_mps, seemps_canonicalmps, βs[i],
                                                                                        leftenv_q, rightenv_q, leftenv_q2, rightenv_q2, leftenv_sk, rightenv_sk,
                                                                                        method, maxiter, ftol)
                    
                    
                average_q = self.av_q(newAn,leftenv_q2,rightenv_q2)
                norm_mps = self.norm(newAn,leftenv_q,rightenv_q)
                avq = average_q/(norm_mps**2)
                if avq>avqtol:
                    #If we superate the tol on avq we sample the MPS
                    if do_sampling:
                        history = self.sample_MPS(history, seemps_canonicalmps) 
                    break

            #We initialize again the environments
            if avq<=avqtol:
                for n in tqdm(range(self.N-1,-1,-1), leave=False):
                    direction = 'left'
                    if n!=(self.N-1):
                        seemps_canonicalmps_reduced = CanonicalMPS(canonical_mps.A_mpsstate[n:n+2],center=0, strategy=Strategy(method=0), normalize=True)
                        canonical_mps.A_mpsstate[n] = seemps_canonicalmps_reduced._data[0]
                        canonical_mps.A_mpsstate[n+1] = seemps_canonicalmps_reduced._data[1]
                        seemps_canonicalmps = seemMPS(canonical_mps.A_mpsstate)
                        leftenv_q = canonical_mps.update_leftenvironment_q(n,leftenv_q,direction=direction)
                        rightenv_q = canonical_mps.update_rightenvironment_q(n,rightenv_q,direction=direction)
                        leftenv_q2 = canonical_mps.update_leftenvironment_q2(n,leftenv_q2,direction=direction)
                        rightenv_q2 = canonical_mps.update_rightenvironment_q2(n,rightenv_q2,direction=direction)
                        leftenv_sk = canonical_mps.update_leftenvironment_sk(self.J,n,leftenv_sk,direction=direction)
                        rightenv_sk = canonical_mps.update_rightenvironment_sk(self.J,n,rightenv_sk,direction=direction)

                    newAn, seemps_canonicalmps, canonical_mps = self.optimization_iteration(n, canonical_mps, seemps_canonicalmps, βs[i],
                                                                                            leftenv_q, rightenv_q, leftenv_q2, rightenv_q2, leftenv_sk, rightenv_sk,
                                                                                            method, maxiter, ftol)
                        
                        
                    average_q = self.av_q(newAn,leftenv_q2,rightenv_q2)
                    norm_mps = self.norm(newAn,leftenv_q,rightenv_q)
                    avq = average_q/(norm_mps**2)
                    if avq>avqtol:
                        #If we superate the tol on avq we sample the MPS
                        if do_sampling:
                            history = self.sample_MPS(history, seemps_canonicalmps) 
                        break
                
            extime = time.time() - starttime
            history['beta'] += [βs[i]]
            history = self.update_history(canonical_mps, history, n, leftenv_q,rightenv_q,leftenv_q2,rightenv_q2,leftenv_sk,rightenv_sk)
            avq = history['average q'][-1]
            history['sweep'] += [sw]
            history['extime'] += [extime]
            sw+=1

        if avq<=avqtol and do_sampling:
            #We sample at the end if do_sampling==True and we did not sample before
            history = self.sample_MPS(history, seemps_canonicalmps)

        print(f"Optimitation stopped after {sw} sweeps with beta:{βs[i-1]} Norm:{history['MPS Norm'][-1]} GSprob: {history['GS prob'][-1]} Avq: {history['average q'][-1]} Eres: {history['eres']} Relative error: {history['energy relative error'][-1]}")

        return history, canonical_mps

        

        
