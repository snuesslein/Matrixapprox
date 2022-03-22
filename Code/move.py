from tvsclib.strict_system import StrictSystem
from tvsclib.stage import Stage
from tvsclib.mixed_system import MixedSystem
import numpy as np
import scipy.linalg as linalg

from tvsclib.transformations.output_normal import OutputNormal
from tvsclib.transformations.input_normal import InputNormal
from tvsclib.transformations.reduction import Reduction
import tvsclib.utils as utils

def transform_rl(stages_causal,stages_anticausal,cost,m=1,dir_preset = -1,epsilon=1e-15):
    k = len(stages_causal)
    sigmas_causal = []
    sigmas_anticausal = []
    for i in range(1,len(stages_causal)):#loop over inices of causal states

        #move left:
        b = stages_causal[i-1].B_matrix[:,-m:]
        U,s_l,Vt= np.linalg.svd(np.hstack([stages_causal[i-1].A_matrix,stages_causal[i-1].B_matrix[:,:-m]]),full_matrices=False)
        n = np.count_nonzero(s_l>epsilon)
        Us=U[:,:n]*s_l[:n]


        stages_l = [
            Stage(Vt[:n,:stages_causal[i-1].A_matrix.shape[1]],Vt[:n,stages_causal[i-1].A_matrix.shape[1]:],\
                    stages_causal[i-1].C_matrix,stages_causal[i-1].D_matrix[:,:-m]),
            Stage(stages_causal[i].A_matrix@Us,np.hstack((stages_causal[i].A_matrix@b,stages_causal[i].B_matrix)),\
                    stages_causal[i].C_matrix@Us,np.hstack((stages_causal[i].C_matrix@b,stages_causal[i].D_matrix)))
        ]


        #no move-> only make R_k input normal
        U,s,Vt= np.linalg.svd(np.hstack([stages_causal[i-1].A_matrix,stages_causal[i-1].B_matrix]),full_matrices=False)
        n = np.count_nonzero(s>epsilon)
        Us=U[:,:n]*s[:n]

        stages_n=[
            Stage(Vt[:n,:stages_causal[i-1].A_matrix.shape[1]],Vt[:n,stages_causal[i-1].A_matrix.shape[1]:],\
                    stages_causal[i-1].C_matrix,stages_causal[i-1].D_matrix),
            Stage(stages_causal[i].A_matrix@Us,stages_causal[i].B_matrix,\
                    stages_causal[i].C_matrix@Us,stages_causal[i].D_matrix)
        ]

        #move right -> base on non move

        b = stages_n[1].B_matrix[:,:m]
        d = stages_n[1].D_matrix[:,:m]
        #d_add = np.zeros((stages_n[0].D_matrix.shape[0],1))
        d_add = stages_anticausal[i-1].C_matrix@stages_anticausal[i].B_matrix[:,:m]


        U,s_r,Vt= np.linalg.svd(np.block([[stages_n[1].A_matrix,b],
                                          [stages_n[1].C_matrix,d]]),full_matrices=False)
        n = np.count_nonzero(s_r>epsilon)
        Us=U[:,:n]*s_r[:n]
        stages_r=[
             #Here the A and B are more complicated as we have to stack them
            Stage(Vt[:n,:]@(np.vstack([stages_n[0].A_matrix,np.zeros((m,stages_n[0].A_matrix.shape[1]))])),
                  Vt[:n,:]@(np.block([[stages_n[0].B_matrix,np.zeros((stages_n[0].B_matrix.shape[0],m))],
                               [np.zeros((m,stages_n[0].B_matrix.shape[1])),np.eye(m)]])),
                  stages_n[0].C_matrix,np.hstack([stages_n[0].D_matrix,d_add])),

           Stage(Us[:stages_n[1].A_matrix.shape[0],:],stages_n[1].B_matrix[:,m:],\
                 Us[stages_n[1].A_matrix.shape[0]:,:],stages_n[1].D_matrix[:,m:])
    ]



    # Now calculate the anticausla part:
    #move left

        b = stages_anticausal[i-1].B_matrix[:,-m:]
        d = stages_causal[i-1].D_matrix[:,-m:]
        d_add = np.zeros((stages_anticausal[i].D_matrix.shape[0],m))
        #d_add = stages_causal[i].C_matrix@stages_causal[i-1].B_matrix[:,-1:]

        U,s_al,Vt= np.linalg.svd(np.block([[b,stages_anticausal[i-1].A_matrix],
                                           [d,stages_anticausal[i-1].C_matrix]]),full_matrices=False)
        n = np.count_nonzero(s_al>epsilon)
        sVt=s_al[:n].reshape(-1,1)*Vt[:n,:]

        stages_anti_l=[
            Stage(U[:stages_anticausal[i-1].A_matrix.shape[0],:n],stages_anticausal[i-1].B_matrix[:,:-m],\
                  U[stages_anticausal[i-1].A_matrix.shape[0]:,:n],stages_anticausal[i-1].D_matrix[:,:-m]),
             #Here the A and B are more complicated as we have to stack them
            Stage(sVt@(np.vstack([np.zeros((m,stages_anticausal[i].A_matrix.shape[1])),stages_anticausal[i].A_matrix])),
                  sVt@(np.block([[np.eye(m),np.zeros((m,stages_anticausal[i].B_matrix.shape[1]))],
                                 [np.zeros((stages_anticausal[i].B_matrix.shape[0],m)),stages_anticausal[i].B_matrix]])),
                  stages_anticausal[i].C_matrix,np.hstack([d_add,stages_anticausal[i].D_matrix]))
        ]



        #no move-> only make R_k input normal
        U,s_a,Vt= np.linalg.svd(np.vstack([stages_anticausal[i-1].A_matrix,stages_anticausal[i-1].C_matrix]),full_matrices=False)
        n = np.count_nonzero(s_a>epsilon)
        sVt=s_a[:n].reshape(-1,1)*Vt[:n,:]

        stages_anti_n=[
            Stage(U[:stages_anticausal[i-1].A_matrix.shape[0],:n],stages_anticausal[i-1].B_matrix,\
                  U[stages_anticausal[i-1].A_matrix.shape[0]:,:n],stages_anticausal[i-1].D_matrix),
            Stage(sVt@stages_anticausal[i].A_matrix,sVt@stages_anticausal[i].B_matrix,\
                  stages_anticausal[i].C_matrix,stages_anticausal[i].D_matrix)
        ]



        #move right: -> base on non move
        b = stages_anti_n[1].B_matrix[:,:m]
        U,s_ar,Vt= np.linalg.svd(np.hstack([stages_anti_n[1].A_matrix,stages_anti_n[1].B_matrix[:,m:]]),full_matrices=False)
        n = np.count_nonzero(s_ar>epsilon)
        sVt=s_ar[:n].reshape(-1,1)*Vt[:n,:]

        stages_anti_r = [
            Stage(stages_anti_n[0].A_matrix@U[:,:n],np.hstack((stages_anti_n[0].B_matrix,stages_anti_n[0].A_matrix@b)),\
                  stages_anti_n[0].C_matrix@U[:,:n],np.hstack((np.zeros((stages_anti_n[0].D_matrix.shape[0],m)),stages_anti_n[0].D_matrix))),
            #D is here 0, instead we have d_add at the causal system. Insted we could use stages_anti_n[0].C_matrix@b
            Stage(sVt[:,:stages_anti_n[1].A_matrix.shape[1]],sVt[:,stages_anti_n[1].A_matrix.shape[1]:],\
                    stages_anti_n[1].C_matrix,stages_anti_n[1].D_matrix[:,:-m])
        ]

        dims_in = [stage.D_matrix.shape[1] for stage in stages_causal]
        dims_out = [stage.D_matrix.shape[0] for stage in stages_causal]
        d_in = np.sum(dims_in[:i])
        d_out = np.sum(dims_out[i:])

        d_in_a = np.sum(dims_in[i:])
        d_out_a = np.sum(dims_out[:i])
        costs = np.array([cost(s,(d_out,d_in),s_a,(d_out_a,d_in_a)),\
                          cost(s_l,(d_out,d_in-1),s_al,(d_out_a,d_in_a+1)),\
                          cost(s_r,(d_out,d_in+1),s_ar,(d_out_a,d_in_a-1))])
        #print("costs_lnr:",costs)
        if dir_preset==-1:
            direction = np.argmin(costs)
        else:
            direction = dir_preset
            print("Move",direction)

        if direction == 1:
            stages_causal[i-1]= stages_l[0]
            stages_causal[i]= stages_l[1]
            stages_anticausal[i-1]= stages_anti_l[0]
            stages_anticausal[i]= stages_anti_l[1]
            sigmas_causal.append(s_l)
            sigmas_anticausal.append(s_al)
        if direction == 0:
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

def transform_ud(stages_causal,stages_anticausal,cost,m=1,dir_preset = -1,epsilon=1e-15):
    k = len(stages_causal)
    sigmas_causal = []
    sigmas_anticausal = []
    for i in range(k-1, 0,-1):

        #move down:
        c = stages_causal[i].C_matrix[:m,:]
        U,s_d,Vt= np.linalg.svd(np.vstack([stages_causal[i].A_matrix,stages_causal[i].C_matrix[m:,:]]),full_matrices=False)
        n = np.count_nonzero(s_d>epsilon)
        sVt=s_d[:n].reshape(-1,1)*Vt[:n,:]

        stages_d = [
            Stage(sVt@stages_causal[i-1].A_matrix,sVt@stages_causal[i-1].B_matrix,\
                np.vstack([stages_causal[i-1].C_matrix,c@stages_causal[i-1].A_matrix]),
                  np.vstack([stages_causal[i-1].D_matrix,c@stages_causal[i-1].B_matrix])),
            Stage(U[:stages_causal[i].A_matrix.shape[0],:n],stages_causal[i].B_matrix,\
                  U[stages_causal[i].A_matrix.shape[0]:,:n],stages_causal[i].D_matrix[m:,:])
        ]



        #no move-> only make O_k normal
        U,s,Vt= np.linalg.svd(np.vstack([stages_causal[i].A_matrix,stages_causal[i].C_matrix]),full_matrices=False)
        n = np.count_nonzero(s>epsilon)
        sVt=s[:n].reshape(-1,1)*Vt[:n,:]

        stages_n=[
            Stage(sVt@stages_causal[i-1].A_matrix,sVt@stages_causal[i-1].B_matrix,\
                stages_causal[i-1].C_matrix,stages_causal[i-1].D_matrix),
            Stage(U[:stages_causal[i].A_matrix.shape[0],:n],stages_causal[i].B_matrix,\
                  U[stages_causal[i].A_matrix.shape[0]:,:n],stages_causal[i].D_matrix)
        ]


        c = stages_n[0].C_matrix[-m:,:]
        d = stages_n[0].D_matrix[-m:,:]
        #d_add = np.zeros((1,stages_n[1].D_matrix.shape[1]))
        d_add = stages_anticausal[i-1].C_matrix[-m:,:]@stages_anticausal[i].B_matrix


        U,s_u,Vt= np.linalg.svd(np.block([[c,d],
                                          [stages_n[0].A_matrix,stages_n[0].B_matrix]]),full_matrices=False)
        n = np.count_nonzero(s_u>epsilon)
        sVt=s_u[:n].reshape(-1,1)*Vt[:n,:]
        stages_u=[
           Stage(sVt[:,:stages_n[0].A_matrix.shape[1]],sVt[:,stages_n[0].A_matrix.shape[1]:],\
                 stages_n[0].C_matrix[:-m,:],stages_n[0].D_matrix[:-m,:]),
             #Here the A and C are more complicated as we have to stack them
            Stage(np.hstack([np.zeros((stages_n[1].A_matrix.shape[0],m)),stages_n[1].A_matrix])@U[:,:n],
                  stages_n[1].B_matrix,\
                  np.block([[np.eye(m),np.zeros((m,stages_n[1].C_matrix.shape[1]))],
                            [np.zeros((stages_n[1].C_matrix.shape[0],m)),stages_n[1].C_matrix]])@U[:,:n],
                  np.vstack([d_add,stages_n[1].D_matrix]))
        ]

    # Now calculate the anticausal part:

        c = stages_anticausal[i].C_matrix[:m,:]
        d = stages_causal[i].D_matrix[:m,:]
        d_add = np.zeros((m,stages_anticausal[i-1].D_matrix.shape[1]))
        #d_add = stages_causal[i].C_matrix@stages_causal[i-1].B_matrix[:,-1:]

        U,s_ad,Vt= np.linalg.svd(np.block([[stages_anticausal[i].A_matrix,stages_anticausal[i].B_matrix],
                                           [c,d]]),full_matrices=False)
        n = np.count_nonzero(s_ad>epsilon)
        Us=U[:,:n]*s_ad[:n]
        stages_anti_d=[
             #Here the A and B are more complicated as we have to stack them
            Stage((np.hstack([stages_anticausal[i-1].A_matrix,np.zeros((stages_anticausal[i-1].A_matrix.shape[0],m))]))@Us,
                  stages_anticausal[i-1].B_matrix,
                  np.block([[stages_anticausal[i-1].C_matrix,np.zeros((stages_anticausal[i-1].C_matrix.shape[0],m))],
                            [np.zeros((m,stages_anticausal[i-1].C_matrix.shape[1])),np.eye(m)]])@Us,
                  np.vstack([stages_anticausal[i-1].D_matrix,d_add])),
            Stage(Vt[:n,:stages_anticausal[i].A_matrix.shape[1]],Vt[:n,stages_anticausal[i].A_matrix.shape[1]:],\
                  stages_anticausal[i].C_matrix[m:,:],stages_anticausal[i].D_matrix[m:,:])
        ]


        #no move-> only make R_k input normal
        U,s_a,Vt= np.linalg.svd(np.hstack([stages_anticausal[i].A_matrix,stages_anticausal[i].B_matrix]),full_matrices=False)
        n = np.count_nonzero(s_a>epsilon)
        Us=U[:,:n]*s_a[:n]

        stages_anti_n=[
            Stage(stages_anticausal[i-1].A_matrix@Us,stages_anticausal[i-1].B_matrix,\
                  stages_anticausal[i-1].C_matrix@Us,stages_anticausal[i-1].D_matrix),
            Stage(Vt[:n,:stages_anticausal[i].A_matrix.shape[1]],Vt[:n,stages_anticausal[i].A_matrix.shape[1]:],\
                  stages_anticausal[i].C_matrix,stages_anticausal[i].D_matrix)
        ]


        #move right: -> base on non move
        c = stages_anti_n[0].C_matrix[-m:,:]
        U,s_au,Vt= np.linalg.svd(np.vstack([stages_anti_n[0].A_matrix,stages_anti_n[0].C_matrix[:-m,:]]),full_matrices=False)
        n = np.count_nonzero(s_au>epsilon)
        Us=U[:,:n]*s_au[:n]

        stages_anti_u = [
            #D is here 0, instead we have d_add at the causal system. Insted we could use stages_anti_n[0].C_matrix@b
            Stage(Us[:stages_anti_n[0].A_matrix.shape[0],:],stages_anti_n[0].B_matrix,\
                  Us[stages_anti_n[0].A_matrix.shape[0]:,:],stages_anti_n[0].D_matrix[:-m,:]),
            Stage(Vt[:n,:]@stages_anti_n[1].A_matrix,Vt[:n,:]@stages_anti_n[1].B_matrix,\
                  np.vstack((c@stages_anti_n[1].A_matrix,stages_anti_n[1].C_matrix)),\
                  np.vstack((np.zeros((m,stages_anti_n[1].D_matrix.shape[1])),stages_anti_n[1].D_matrix)))
        ]

        dims_in = [stage.D_matrix.shape[1] for stage in stages_causal]
        dims_out = [stage.D_matrix.shape[0] for stage in stages_causal]
        d_in = np.sum(dims_in[:i])
        d_out = np.sum(dims_out[i:])

        d_in_a = np.sum(dims_in[i:])
        d_out_a = np.sum(dims_out[:i])
        costs = np.array([cost(s,(d_out,d_in),s_a,(d_out_a,d_in_a)),\
                          cost(s_d,(d_out-1,d_in),s_ad,(d_out_a+1,d_in_a)),\
                          cost(s_u,(d_out+1,d_in),s_au,(d_out_a-1,d_in_a))])
        #print("costs_dnu:",costs)
        if dir_preset==-1:
            direction = np.argmin(costs)
        else:
            direction = dir_preset
            print("Move",direction)

        if direction == 1:
            stages_causal[i-1]= stages_d[0]
            stages_causal[i]= stages_d[1]
            stages_anticausal[i-1]= stages_anti_d[0]
            stages_anticausal[i]= stages_anti_d[1]
            sigmas_causal.append(s_d)
            sigmas_anticausal.append(s_ad)
        if direction == 0:
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

def move(system,m,cost,n_in = 1,n_out=1):
    """

    parameters:

    m: number of iterations
    cost: function that calculates a cost term for the sigmas

    """
    sys_move_causal = system.causal_system.copy()
    sys_move_anticausal = system.anticausal_system.copy()

    sys_move_causal = InputNormal().apply(sys_move_causal)
    sys_move_anticausal = OutputNormal().apply(sys_move_anticausal)

    sys_move =MixedSystem(causal_system=sys_move_causal,anticausal_system=sys_move_anticausal)

    input_dims=np.zeros((len(sys_move.causal_system.stages),m+1))
    output_dims=np.zeros((len(sys_move.causal_system.stages),m+1))
    input_dims[:,0] = sys_move.dims_in
    output_dims[:,0] = sys_move.dims_out
    for m in range(m):
        sigmas_causal,sigmas_anticausal=transform_ud(sys_move.causal_system.stages,
                                                 sys_move.anticausal_system.stages,cost,m=n_out)

        sigmas_causal,sigmas_anticausal=transform_rl(sys_move.causal_system.stages,
                                                 sys_move.anticausal_system.stages,cost,m=n_in)

        input_dims[:,m+1] = sys_move.dims_in
        output_dims[:,m+1] = sys_move.dims_out

    return sys_move,input_dims,output_dims


def test_moves(system,m,epsilon=1e-15):
    sys_move_causal = system.causal_system.copy()
    sys_move_anticausal = system.anticausal_system.copy()

    sys_move_causal = InputNormal().apply(sys_move_causal)
    sys_move_anticausal = OutputNormal().apply(sys_move_anticausal)

    sys_move =MixedSystem(causal_system=sys_move_causal,anticausal_system=sys_move_anticausal)

    input_dims = sys_move.dims_in
    output_dims = sys_move.dims_out
    cost = lambda x,y,z,a: 1
    print("testing move d")
    sys_move_d = sys_move.copy()
    sigmas_causal,sigmas_anticausal=transform_ud(sys_move_d.causal_system.stages,
                                                 sys_move_d.anticausal_system.stages,cost,m=m,dir_preset=1,epsilon=epsilon)
    utils.check_dims(sys_move_d)
    print(np.max(np.abs(sys_move.to_matrix()-sys_move_d.to_matrix())))

    print("testing move u")
    sys_move_u = sys_move.copy()
    sigmas_causal,sigmas_anticausal=transform_ud(sys_move_u.causal_system.stages,
                                                 sys_move_u.anticausal_system.stages,cost,m=m,dir_preset=2,epsilon=epsilon)
    utils.check_dims(sys_move_u)
    print(np.max(np.abs(sys_move.to_matrix()-sys_move_u.to_matrix())))

    print("testing move none")
    sys_move_n = sys_move.copy()
    sigmas_causal,sigmas_anticausal=transform_ud(sys_move_n.causal_system.stages,
                                                 sys_move_n.anticausal_system.stages,cost,m=m,dir_preset=0,epsilon=epsilon)
    utils.check_dims(sys_move_n)
    print(np.max(np.abs(sys_move.to_matrix()-sys_move_n.to_matrix())))

    print("testing_move l")
    sys_move_l = sys_move_n.copy()
    sigmas_causal,sigmas_anticausal=transform_rl(sys_move_l.causal_system.stages,
                                                 sys_move_l.anticausal_system.stages,cost,m=m,dir_preset=1,epsilon=epsilon)
    utils.check_dims(sys_move_l)
    print(np.max(np.abs(sys_move_n.to_matrix()-sys_move_l.to_matrix())))

    print("testing_move r")
    sys_move_r = sys_move_n.copy()
    sigmas_causal,sigmas_anticausal=transform_rl(sys_move_r.causal_system.stages,
                                                 sys_move_r.anticausal_system.stages,cost,m=m,dir_preset=2,epsilon=epsilon)
    utils.check_dims(sys_move_r)
    print(np.max(np.abs(sys_move_n.to_matrix()-sys_move_r.to_matrix())))

    print("testing_move nn")
    sys_move_nn = sys_move_n.copy()
    sigmas_causal,sigmas_anticausal=transform_rl(sys_move_nn.causal_system.stages,
                                                 sys_move_nn.anticausal_system.stages,cost,m=m,dir_preset=0,epsilon=epsilon)
    utils.check_dims(sys_move_nn)
    print(np.max(np.abs(sys_move_n.to_matrix()-sys_move_nn.to_matrix())))
