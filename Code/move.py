from tvsclib.strict_system import StrictSystem
from tvsclib.stage import Stage
from tvsclib.mixed_system import MixedSystem
import numpy as np
import scipy.linalg as linalg

from tvsclib.transformations.output_normal import OutputNormal
from tvsclib.transformations.input_normal import InputNormal
from tvsclib.transformations.reduction import Reduction


def transform_rl(stages_causal,stages_anticausal,cost):
    k = len(stages_causal)
    sigmas_causal = []
    sigmas_anticausal = []
    for i in range(1,len(stages_causal)):#loop over inices of causal states

        #move left:
        b = stages_causal[i-1].B_matrix[:,-1:]
        U,s_l,Vt= np.linalg.svd(np.hstack([stages_causal[i-1].A_matrix,stages_causal[i-1].B_matrix[:,:-1]]),full_matrices=False)
        Us=U*s_l

        stages_l = [
            Stage(Vt[:,:stages_causal[i-1].A_matrix.shape[1]],Vt[:,stages_causal[i-1].A_matrix.shape[1]:],\
                    stages_causal[i-1].C_matrix,stages_causal[i-1].D_matrix[:,:-1]),
            Stage(stages_causal[i].A_matrix@Us,np.hstack((stages_causal[i].A_matrix@b,stages_causal[i].B_matrix)),\
                    stages_causal[i].C_matrix@Us,np.hstack((stages_causal[i].C_matrix@b,stages_causal[i].D_matrix)))
        ]


        #no move-> only make R_k input normal
        U,s,Vt= np.linalg.svd(np.hstack([stages_causal[i-1].A_matrix,stages_causal[i-1].B_matrix]),full_matrices=False)
        Us=U*s

        stages_n=[
            Stage(Vt[:,:stages_causal[i-1].A_matrix.shape[1]],Vt[:,stages_causal[i-1].A_matrix.shape[1]:],\
                    stages_causal[i-1].C_matrix,stages_causal[i-1].D_matrix),
            Stage(stages_causal[i].A_matrix@Us,stages_causal[i].B_matrix,\
                    stages_causal[i].C_matrix@Us,stages_causal[i].D_matrix)
        ]

        #move right -> base on non move

        b = stages_n[1].B_matrix[:,:1]
        d = stages_n[1].D_matrix[:,:1]
        #d_add = np.zeros((stages_n[0].D_matrix.shape[0],1))
        d_add = stages_anticausal[i-1].C_matrix@stages_anticausal[i].B_matrix[:,:1]


        U,s_r,Vt= np.linalg.svd(np.block([[stages_n[1].A_matrix,b],
                                          [stages_n[1].C_matrix,d]]),full_matrices=False)
        Us=U*s_r
        stages_r=[
             #Here the A and B are more complicated as we have to stack them
            Stage(Vt@(np.vstack([stages_n[0].A_matrix,np.zeros((1,stages_n[0].A_matrix.shape[1]))])),
                  Vt@(np.block([[stages_n[0].B_matrix,np.zeros((stages_n[0].B_matrix.shape[0],1))],
                               [np.zeros(stages_n[0].B_matrix.shape[1]),np.eye(1)]])),
                  stages_n[0].C_matrix,np.hstack([stages_n[0].D_matrix,d_add])),

           Stage(Us[:stages_n[1].A_matrix.shape[0],:],stages_n[1].B_matrix[:,1:],\
                 Us[stages_n[1].A_matrix.shape[0]:,:],stages_n[1].D_matrix[:,1:])
    ]



    # Now calculate the anticausla part:
    #move left

        b = stages_anticausal[i-1].B_matrix[:,-1:]
        d = stages_causal[i-1].D_matrix[:,-1:]
        d_add = np.zeros((stages_anticausal[i].D_matrix.shape[0],1))
        #d_add = stages_causal[i].C_matrix@stages_causal[i-1].B_matrix[:,-1:]

        U,s_al,Vt= np.linalg.svd(np.block([[b,stages_anticausal[i-1].A_matrix],
                                           [d,stages_anticausal[i-1].C_matrix]]),full_matrices=False)
        sVt=s_al.reshape(-1,1)*Vt
        stages_anti_l=[
            Stage(U[:stages_anticausal[i-1].A_matrix.shape[0],:],stages_anticausal[i-1].B_matrix[:,:-1],\
                  U[stages_anticausal[i-1].A_matrix.shape[0]:,:],stages_anticausal[i-1].D_matrix[:,:-1]),
             #Here the A and B are more complicated as we have to stack them
            Stage(sVt@(np.vstack([np.zeros((1,stages_anticausal[i].A_matrix.shape[1])),stages_anticausal[i].A_matrix])),
                  sVt@(np.block([[np.eye(1),np.zeros(stages_anticausal[i].B_matrix.shape[1])],
                                 [np.zeros((stages_anticausal[i].B_matrix.shape[0],1)),stages_anticausal[i].B_matrix]])),
                  stages_anticausal[i].C_matrix,np.hstack([d_add,stages_anticausal[i].D_matrix]))
        ]



        #no move-> only make R_k input normal
        U,s_a,Vt= np.linalg.svd(np.vstack([stages_anticausal[i-1].A_matrix,stages_anticausal[i-1].C_matrix]),full_matrices=False)
        sVt=s_a.reshape(-1,1)*Vt

        stages_anti_n=[
            Stage(U[:stages_anticausal[i-1].A_matrix.shape[0],:],stages_anticausal[i-1].B_matrix,\
                  U[stages_anticausal[i-1].A_matrix.shape[0]:,:],stages_anticausal[i-1].D_matrix),
            Stage(sVt@stages_anticausal[i].A_matrix,sVt@stages_anticausal[i].B_matrix,\
                  stages_anticausal[i].C_matrix,stages_anticausal[i].D_matrix)
        ]



        #move right: -> base on non move
        b = stages_anti_n[1].B_matrix[:,:1]
        U,s_ar,Vt= np.linalg.svd(np.hstack([stages_anti_n[1].A_matrix,stages_anti_n[1].B_matrix[:,1:]]),full_matrices=False)
        sVt=s_ar.reshape(-1,1)*Vt

        stages_anti_r = [
            Stage(stages_anti_n[0].A_matrix@U,np.hstack((stages_anti_n[0].B_matrix,stages_anti_n[0].A_matrix@b)),\
                  stages_anti_n[0].C_matrix@U,np.hstack((np.zeros((stages_anti_n[0].D_matrix.shape[0],1)),stages_anti_n[0].D_matrix))),
            #D is here 0, instead we have d_add at the causal system. Insted we could use stages_anti_n[0].C_matrix@b
            Stage(sVt[:,:stages_anti_n[1].A_matrix.shape[1]],sVt[:,stages_anti_n[1].A_matrix.shape[1]:],\
                    stages_anti_n[1].C_matrix,stages_anti_n[1].D_matrix[:,:-1])
        ]

        costs = np.array([cost(s_l,s_al),cost(s,s_a),cost(s_r,s_ar)])
        print("costs_lnr:",costs)
        direction = np.argmin(costs)

        if direction == 0:
            stages_causal[i-1]= stages_l[0]
            stages_causal[i]= stages_l[1]
            stages_anticausal[i-1]= stages_anti_l[0]
            stages_anticausal[i]= stages_anti_l[1]
            sigmas_causal.append(s_l)
            sigmas_anticausal.append(s_al)
        if direction == 1:
            stages_causal[i-1]= stages_n[0]
            stages_causal[i]= stages_n[1]
            stages_anticausal[i-1]= stages_anti_n[0]
            stages_anticausal[i]= stages_anti_n[1]
            sigmas_causal.append(s)
            sigmas_anticausal.append(s_a)
        if direction == 2:
            stages_causal[i-1]= stages_r[0]
            stages_causal[i]= stages_r[1]
            stages_anticausal[i-1]= stages_anti_r[0]
            stages_anticausal[i]= stages_anti_r[1]
            sigmas_causal.append(s_r)
            sigmas_anticausal.append(s_ar)
    return sigmas_causal,sigmas_anticausal

def transform_ud(stages_causal,stages_anticausal,cost):
    k = len(stages_causal)
    sigmas_causal = []
    sigmas_anticausal = []
    for i in range(k-1, 0,-1):

        #move down:
        c = stages_causal[i].C_matrix[:1,:]
        U,s_d,Vt= np.linalg.svd(np.vstack([stages_causal[i].A_matrix,stages_causal[i].C_matrix[1:,:]]),full_matrices=False)
        sVt=s_d.reshape(-1,1)*Vt

        stages_d = [
            Stage(sVt@stages_causal[i-1].A_matrix,sVt@stages_causal[i-1].B_matrix,\
                np.vstack([stages_causal[i-1].C_matrix,c@stages_causal[i-1].A_matrix]),
                  np.vstack([stages_causal[i-1].D_matrix,c@stages_causal[i-1].B_matrix])),
            Stage(U[:stages_causal[i].A_matrix.shape[0],:],stages_causal[i].B_matrix,\
                  U[stages_causal[i].A_matrix.shape[0]:,:],stages_causal[i].D_matrix[1:,:])
        ]



        #no move-> only make O_k normal
        U,s,Vt= np.linalg.svd(np.vstack([stages_causal[i].A_matrix,stages_causal[i].C_matrix]),full_matrices=False)
        sVt=s.reshape(-1,1)*Vt

        stages_n=[
            Stage(sVt@stages_causal[i-1].A_matrix,sVt@stages_causal[i-1].B_matrix,\
                stages_causal[i-1].C_matrix,stages_causal[i-1].D_matrix),
            Stage(U[:stages_causal[i].A_matrix.shape[0],:],stages_causal[i].B_matrix,\
                  U[stages_causal[i].A_matrix.shape[0]:,:],stages_causal[i].D_matrix)
        ]


        c = stages_n[0].C_matrix[-1:,:]
        d = stages_n[0].D_matrix[-1:,:]
        #d_add = np.zeros((1,stages_n[1].D_matrix.shape[1]))
        d_add = stages_anticausal[i-1].C_matrix[-1:,:]@stages_anticausal[i].B_matrix


        U,s_u,Vt= np.linalg.svd(np.block([[c,d],
                                          [stages_n[0].A_matrix,stages_n[0].B_matrix]]),full_matrices=False)
        sVt=s_u.reshape(-1,1)*Vt
        stages_u=[
           Stage(sVt[:,:stages_n[0].A_matrix.shape[1]],sVt[:,stages_n[0].A_matrix.shape[1]:],\
                 stages_n[0].C_matrix[:-1,:],stages_n[0].D_matrix[:-1,:]),
             #Here the A and C are more complicated as we have to stack them
            Stage(np.hstack([np.zeros((stages_n[1].A_matrix.shape[0],1)),stages_n[1].A_matrix])@U,
                  stages_n[1].B_matrix,\
                  np.block([[np.eye(1),np.zeros((1,stages_n[1].C_matrix.shape[1]))],
                            [np.zeros((stages_n[1].C_matrix.shape[0],1)),stages_n[1].C_matrix]])@U,
                  np.vstack([d_add,stages_n[1].D_matrix]))
        ]

    # Now calculate the anticausal part:

        c = stages_anticausal[i].C_matrix[:1,:]
        d = stages_causal[i].D_matrix[:1,:]
        d_add = np.zeros((1,stages_anticausal[i-1].D_matrix.shape[1]))
        #d_add = stages_causal[i].C_matrix@stages_causal[i-1].B_matrix[:,-1:]

        U,s_ad,Vt= np.linalg.svd(np.block([[stages_anticausal[i].A_matrix,stages_anticausal[i].B_matrix],
                                           [c,d]]),full_matrices=False)
        Us=U*s_ad
        stages_anti_d=[
             #Here the A and B are more complicated as we have to stack them
            Stage((np.hstack([stages_anticausal[i-1].A_matrix,np.zeros((stages_anticausal[i-1].A_matrix.shape[0],1))]))@Us,
                  stages_anticausal[i-1].B_matrix,
                  np.block([[stages_anticausal[i-1].C_matrix,np.zeros((stages_anticausal[i-1].C_matrix.shape[0],1))],
                            [np.zeros((1,stages_anticausal[i-1].C_matrix.shape[1])),np.eye(1)]])@Us,
                  np.vstack([stages_anticausal[i-1].D_matrix,d_add])),
            Stage(Vt[:,:stages_anticausal[i].A_matrix.shape[1]],Vt[:,stages_anticausal[i].A_matrix.shape[1]:],\
                  stages_anticausal[i].C_matrix[1:,:],stages_anticausal[i].D_matrix[1:,:])
        ]


        #no move-> only make R_k input normal
        U,s_a,Vt= np.linalg.svd(np.hstack([stages_anticausal[i].A_matrix,stages_anticausal[i].B_matrix]),full_matrices=False)
        Us=U*s_a

        stages_anti_n=[
            Stage(stages_anticausal[i-1].A_matrix@Us,stages_anticausal[i-1].B_matrix,\
                  stages_anticausal[i-1].C_matrix@Us,stages_anticausal[i-1].D_matrix),
            Stage(Vt[:,:stages_anticausal[i].A_matrix.shape[1]],Vt[:,stages_anticausal[i].A_matrix.shape[1]:],\
                  stages_anticausal[i].C_matrix,stages_anticausal[i].D_matrix)
        ]


        #move right: -> base on non move
        c = stages_anti_n[0].C_matrix[-1:,:]
        U,s_au,Vt= np.linalg.svd(np.vstack([stages_anti_n[0].A_matrix,stages_anti_n[0].C_matrix[:-1,:]]),full_matrices=False)
        Us=U*s_au

        stages_anti_u = [
            #D is here 0, instead we have d_add at the causal system. Insted we could use stages_anti_n[0].C_matrix@b
            Stage(Us[:stages_anti_n[0].A_matrix.shape[0],:],stages_anti_n[0].B_matrix,\
                  Us[stages_anti_n[0].A_matrix.shape[0]:,:],stages_anti_n[0].D_matrix[:-1,:]),
            Stage(Vt@stages_anti_n[1].A_matrix,Vt@stages_anti_n[1].B_matrix,\
                  np.vstack((c@stages_anti_n[1].A_matrix,stages_anti_n[1].C_matrix)),\
                  np.vstack((np.zeros((1,stages_anti_n[1].D_matrix.shape[1])),stages_anti_n[1].D_matrix)))
        ]

        costs = np.array([cost(s_d,s_ad),cost(s,s_a),cost(s_u,s_au)])
        print("costs_dnu:",costs)
        direction = np.argmin(costs)

        if direction == 0:
            stages_causal[i-1]= stages_d[0]
            stages_causal[i]= stages_d[1]
            stages_anticausal[i-1]= stages_anti_d[0]
            stages_anticausal[i]= stages_anti_d[1]
            sigmas_causal.append(s_d)
            sigmas_anticausal.append(s_ad)
        if direction == 1:
            stages_causal[i-1]= stages_n[0]
            stages_causal[i]= stages_n[1]
            stages_anticausal[i-1]= stages_anti_n[0]
            stages_anticausal[i]= stages_anti_n[1]
            sigmas_causal.append(s)
            sigmas_anticausal.append(s_a)
        if direction == 2:
            stages_causal[i-1]= stages_u[0]
            stages_causal[i]= stages_u[1]
            stages_anticausal[i-1]= stages_anti_u[0]
            stages_anticausal[i]= stages_anti_u[1]
            sigmas_causal.append(s_u)
            sigmas_anticausal.append(s_au)
    sigmas_causal.reverse()
    sigmas_anticausal.reverse()
    return sigmas_causal,sigmas_anticausal

def move(system,N,cost):
    """

    parameters:

    N: number of iterations
    cost: function that calculates a cost term for the sigmas

    """
    sys_move_causal = system.causal_system.copy()
    sys_move_anticausal = system.anticausal_system.copy()

    sys_move_causal = InputNormal().apply(sys_move_causal)
    sys_move_anticausal = OutputNormal().apply(sys_move_anticausal)

    sys_move =MixedSystem(causal_system=sys_move_causal,anticausal_system=sys_move_anticausal)

    input_dims=np.zeros((len(sys_move.causal_system.stages),N+1))
    output_dims=np.zeros((len(sys_move.causal_system.stages),N+1))
    input_dims[:,0] = sys_move.dims_in
    output_dims[:,0] = sys_move.dims_out
    for n in range(N):
        sigmas_causal,sigmas_anticausal=transform_ud(sys_move.causal_system.stages,
                                                 sys_move.anticausal_system.stages,cost)

        sigmas_causal,sigmas_anticausal=transform_rl(sys_move.causal_system.stages,
                                                 sys_move.anticausal_system.stages,cost)

        input_dims[:,n+1] = sys_move.dims_in
        output_dims[:,n+1] = sys_move.dims_out

    return sys_move,input_dims,output_dims
