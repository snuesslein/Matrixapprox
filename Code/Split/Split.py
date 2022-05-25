#!/usr/bin/env python



import numpy as np
import matplotlib.pyplot as plt
from tvsclib.stage import Stage
from tvsclib.strict_system import StrictSystem
from tvsclib.mixed_system import MixedSystem
import tvsclib.utils as utils
import tvsclib.math as math





def split_stage(stage,i_in,i_out):
    A=stage.A_matrix
    B=stage.B_matrix
    C=stage.C_matrix
    D=stage.D_matrix
    (d_stateo,d_statei)=A.shape
    U,s,Vt = np.linalg.svd(np.block([[C[i_out:,:],D[i_out:,:i_in]],[A,B[:,:i_in]]]),full_matrices=False)
    
    s_sqrt = np.sqrt(s)
    stage_alpha=Stage(s_sqrt.reshape(-1,1)*Vt[:,:d_statei],                 s_sqrt.reshape(-1,1)*Vt[:,d_statei:],                 C[:i_out,:],                 D[:i_out,:i_in])
    stage_beta=Stage(U[U.shape[0]-d_stateo:,:]*s_sqrt.reshape(1,-1),               B[:,i_in:],               U[:U.shape[0]-d_stateo,:]*s_sqrt.reshape(1,-1),               D[i_out:,i_in:])
    
    return stage_alpha,stage_beta
    
def split_stage_anti(stage,i_in,i_out,D):
    A=stage.A_matrix
    B=stage.B_matrix
    C=stage.C_matrix
    #D=stage.D_matrix
    (d_stateo,d_statei)=A.shape
    U,s,Vt = np.linalg.svd(np.block([[C[:i_out,:],D[:i_out,i_in:]],[A,B[:,i_in:]]]),full_matrices=False)
    
    s_sqrt = np.sqrt(s)
    stage_alpha=Stage(s_sqrt.reshape(-1,1)*Vt[:,:d_statei],                 s_sqrt.reshape(-1,1)*Vt[:,d_statei:],                 C[i_out:,:],                 np.zeros_like(D[:i_out,:i_in]))
    stage_beta=Stage(U[U.shape[0]-d_stateo:,:]*s_sqrt.reshape(1,-1),               B[:,:i_in],               U[:U.shape[0]-d_stateo,:]*s_sqrt.reshape(1,-1),               np.zeros_like(D[i_out:,i_in:]))
    
    return stage_alpha,stage_beta
    
def split(system,k,i_in,i_out):
    stage_alpha,stage_beta = split_stage(system.stages[k],i_in,i_out)
    system.stages[k]=stage_alpha
    system.stages.insert(k+1,stage_beta)
    
def split_anti(system,k,i_in,i_out,D):
    stage_alpha,stage_beta = split_stage_anti(system.stages[k],i_in,i_out,D)
    system.stages[k]=stage_beta
    system.stages.insert(k+1,stage_alpha)
    
def split_mixed(system,k,i_in,i_out):
    split_anti(system.anticausal_system,k,i_in,i_out,system.causal_system.stages[k].D_matrix)
    split(system.causal_system,k,i_in,i_out)
    




def initial(T):
    """
    
    returns a tupel with the initial lists
    for As,Bs,Cs,Ds
    
    """
    
    return StrictSystem(stages=[Stage(np.zeros((0,0)),np.zeros((0,T.shape[1])),np.zeros((T.shape[0],0)),T)],causal=True)

def initial_mixed(T):
    """
    
    returns a tupel with the initial lists
    for As,Bs,Cs,Ds
    
    """
    
    return MixedSystem(\
            causal_system=StrictSystem(stages=[Stage(np.zeros((0,0)),np.zeros((0,T.shape[1])),np.zeros((T.shape[0],0)),T)],causal=True),\
        anticausal_system=StrictSystem(stages=[Stage(np.zeros((0,0)),np.zeros((0,T.shape[1])),np.zeros((T.shape[0],0)),np.zeros_like(T))],causal=False)\
                      )

class A_matrix:
    def __init__(self,s_out,A,s_in):
        self.A = A
        self.s_out = s_out
        self.s_in = s_in
    
    @property
    def input_normal(self):
        return self.A*self.s_in.reshape(1,-1)

    @property
    def output_normal(self):
        return self.s_out.reshape(-1,1)*self.A

    @property
    def balanced(self):
        return np.sqrt(self.s_out.reshape(-1,1))*self.A*np.sqrt(self.s_in.reshape(1,-1))
 
    @property
    def output_input_normal(self):
        """
        Thsi can be used to get a stage where the previous R_k is input normal 
        and the next O_{k+1} is output normal
        """
        return self.s_out.reshape(-1,1)*self.A*self.s_in.reshape(1,-1)
    
    def copy(self):
        return (A_matrix(self.s_out.copy(),self.A.copy(),self.s_in.copy()))
    
class B_matrix:
    def __init__(self,s_out,B):
        self.B = B
        self.s_out = s_out
    
    @property
    def input_normal(self):
        return self.B

    @property
    def output_normal(self):
        return self.s_out.reshape(-1,1)*self.B

    @property
    def blanced(self):
        return np.sqrt(self.s_out.reshape(-1,1))*self.B

    def copy(self):
        return (B_matrix(self.s_out.copy(),self.B.copy()))
    
class C_matrix:
    def __init__(self,C,s_in):
        self.C = C
        self.s_in = s_in
    
    @property
    def input_normal(self):
        return self.C*self.s_in.reshape(1,-1)

    @property
    def output_normal(self):
        return self.C

    @property
    def balanced(self):
        return self.C*np.sqrt(self.s_in.reshape(1,-1))
    
    def copy(self):
        return (C_matrix(self.C.copy(),self.s_in.copy()))




def split_stage_sigmas(stage,i_in,i_out):
    #we need thsi stage such that previous is input normal and later is output normal
    s_in = stage.A_matrix.s_in
    s_out= stage.A_matrix.s_out
    A=stage.A_matrix.output_input_normal
    B=stage.B_matrix.output_normal
    C=stage.C_matrix.input_normal
    D=stage.D_matrix

    (d_stateo,d_statei)=A.shape
    U,s,Vt = np.linalg.svd(np.block([[C[i_out:,:],D[i_out:,:i_in]],[A,B[:,:i_in]]]),full_matrices=False)
    

    stage_alpha=Stage(
                 A_matrix(s,Vt[:,:d_statei]/s_in.reshape(1,-1),s_in),\
                 B_matrix(s,Vt[:,d_statei:]),\
                 C_matrix(stage.C_matrix.C[:i_out,:],s_in),\
                 D[:i_out,:i_in])
    stage_beta=Stage(
               A_matrix(s_out,U[U.shape[0]-d_stateo:,:]/s_out.reshape(-1,1),s),\
               B_matrix(s_out,stage.B_matrix.B[:,i_in:]),\
               C_matrix(U[:U.shape[0]-d_stateo,:],s),\
               D[i_out:,i_in:])
    
    return stage_alpha,stage_beta

def split_stage_sigmas_anti(stage,i_in,i_out,D):
    #we need thsi stage such that previous is input normal and later is output normal
    s_in = stage.A_matrix.s_in
    s_out= stage.A_matrix.s_out
    A=stage.A_matrix.output_input_normal
    B=stage.B_matrix.output_normal
    C=stage.C_matrix.input_normal
    #D=stage.D_matrix

    (d_stateo,d_statei)=A.shape
    U,s,Vt = np.linalg.svd(np.block([[C[:i_out,:],D[:i_out,i_in:]],[A,B[:,i_in:]]]),full_matrices=False)
    

    stage_alpha=Stage(
                 A_matrix(s,Vt[:,:d_statei]/s_in.reshape(1,-1),s_in),\
                 B_matrix(s,Vt[:,d_statei:]),\
                 C_matrix(stage.C_matrix.C[i_out:,:],s_in),\
                 np.zeros_like(D[:i_out,:i_in]))
    stage_beta=Stage(
               A_matrix(s_out,U[U.shape[0]-d_stateo:,:]/s_out.reshape(-1,1),s),\
               B_matrix(s_out,stage.B_matrix.B[:,:i_in]),\
               C_matrix(U[:U.shape[0]-d_stateo,:],s),\
               np.zeros_like(D[i_out:,i_in:]))
    return stage_alpha,stage_beta

def split_sigmas(system,k,i_in,i_out):
    stage_alpha,stage_beta = split_stage_sigmas(system.stages[k],i_in,i_out)
    system.stages[k]=stage_alpha
    system.stages.insert(k+1,stage_beta)
    
def split_sigmas_anti(system,k,i_in,i_out,D):
    stage_alpha,stage_beta = split_stage_sigmas_anti(system.stages[k],i_in,i_out,D)
    system.stages[k]=stage_beta
    system.stages.insert(k+1,stage_alpha)
    
def split_sigmas_mixed(system,k,i_in,i_out):
    split_sigmas_anti(system.anticausal_system,k,i_in,i_out,system.causal_system.stages[k].D_matrix)
    split_sigmas(system.causal_system,k,i_in,i_out)


# In[ ]:


def initial_sigmas(T):
    """
    
    returns a tupel with the initial lists
    for As,Bs,Cs,Ds
    
    """
    
    return StrictSystem(stages=[Stage(A_matrix(np.zeros(0),np.zeros((0,0)),np.zeros(0)),                                      B_matrix(np.zeros(0),np.zeros((0,T.shape[1]))),                                      C_matrix(np.zeros((T.shape[0],0)),np.zeros(0)),                                      T)],causal=True)

def initial_sigmas_mixed(T):
    """
    
    returns a tupel with the initial lists
    for As,Bs,Cs,Ds
    
    """
    return MixedSystem(                causal_system=StrictSystem(stages=[Stage(A_matrix(np.zeros(0),np.zeros((0,0)),np.zeros(0)),                                      B_matrix(np.zeros(0),np.zeros((0,T.shape[1]))),                                      C_matrix(np.zeros((T.shape[0],0)),np.zeros(0)),                                      T)],causal=True),
                anticausal_system=StrictSystem(stages=[Stage(A_matrix(np.zeros(0),np.zeros((0,0)),np.zeros(0)),\
                                      B_matrix(np.zeros(0),np.zeros((0,T.shape[1]))),\
                                      C_matrix(np.zeros((T.shape[0],0)),np.zeros(0)),\
                                      np.zeros_like(T))],causal=False))

def get_system(system):
    stages = []
    for stage in system.stages:
        stages.append(Stage(stage.A_matrix.input_normal,                            stage.B_matrix.input_normal,                            stage.C_matrix.input_normal,                            stage.D_matrix))
    return StrictSystem(stages=stages,causal=system.causal)

def get_system_mixed(system):
    
    return MixedSystem(causal_system=get_system(system.causal_system),anticausal_system=get_system(system.anticausal_system))



