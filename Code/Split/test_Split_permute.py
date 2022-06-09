import numpy as np
from tvsclib.stage import Stage
from tvsclib.strict_system import StrictSystem
from tvsclib.mixed_system import MixedSystem
import tvsclib.utils as utils
import tvsclib.math as math

import Split_permute as SplitP
from tvsclib.canonical_form import CanonicalForm

import scipy.stats

def testSplit():
    T = np.random.rand(16,16)

    dims_in =  np.array([4, 4, 4, 4])*6
    dims_out = np.array([4, 4, 4, 4])*6

    #dims_in =  np.array([9, 7, 7, 9])*3
    #dims_out = np.array([7, 9, 9, 7])*3


    n = 2
    #create orthogonal vectors and normalize them to the size of the matix (i.e. norm(block)/size(block) = const
    Us =np.vstack([scipy.stats.ortho_group.rvs(dims_out[i])[:,:3*n]*dims_out[i] for i in range(len(dims_in))])
    Vts=np.hstack([scipy.stats.ortho_group.rvs(dims_in[i])[:3*n,:]*dims_in[i] for i in range(len(dims_in))])

    s = np.linspace(1,0.75,n)

    lower = Us[:,:n]@np.diag(s)@Vts[:n,:]
    diag = Us[:,n:2*n]@np.diag(s)@Vts[n:2*n,:]
    upper = Us[:,2*n:3*n]@np.diag(s)@Vts[2*n:3*n,:]
    matrix = np.zeros_like(diag)
    a=0;b=0
    for i in range(len(dims_in)):
        matrix[a:a+dims_out[i],:b]            =lower[a:a+dims_out[i],:b]
        matrix[a:a+dims_out[i],b:b+dims_in[i]]=diag[a:a+dims_out[i],b:b+dims_in[i]]
        matrix[a:a+dims_out[i],b+dims_in[i]:] =upper[a:a+dims_out[i],b+dims_in[i]:]
        a+=dims_out[i];b+=dims_in[i]

    P_in_ref = np.random.permutation(np.arange(matrix.shape[1]))
    P_out_ref= np.random.permutation(np.arange(matrix.shape[0]))

    T = matrix[P_out_ref][:,P_in_ref]


    #test identification
    sys,Ps_col,Ps_row,reports = SplitP.identification_split_permute(T,3,strategy="fro",opts={"gamma":1e6})

    T_per = T[Ps_row[-1]][:,Ps_col[-1]]
    assert np.allclose(T_per,sys.to_matrix()), "identified system not correct"

testSplit()
print("Splitting tested")
