from tvsclib.strict_system import StrictSystem
from tvsclib.stage import Stage
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.mixed_system import MixedSystem
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import tvsclib.utils as utils

from tvsclib.transformations.output_normal import OutputNormal
from tvsclib.transformations.input_normal import InputNormal
from tvsclib.transformations.reduction import Reduction

import move


#matrix = np.arange(0,12).reshape((-1,1))@np.arange(0,12).reshape((1,-1))
N = 3
in_start = 11
out_start = 11
matrix = np.random.rand(N*out_start,N*in_start)
dims_in =  np.array([in_start]*N)
dims_out = np.array([out_start]*N)
T = ToeplitzOperator(matrix, dims_in, dims_out)
S = SystemIdentificationSVD(T,epsilon=1e-12)

system_rand = MixedSystem(S)

move.test_moves(system_rand,4,epsilon=1e-15)

print("test reductiuon m-------------------------------------")
move.test_moves(system_rand,11,epsilon=1e-15)



in_start = 11
out_start = 11
matrix = np.random.rand(N*out_start,N*in_start)
dims_in =  np.array([0]+[in_start]*N)
dims_out = np.array([0]+[out_start]*N)
T = ToeplitzOperator(matrix, dims_in, dims_out)
S = SystemIdentificationSVD(T,epsilon=1e-12)
system_rand = MixedSystem(S)

print("test start with 0 dim--------------------------------")
move.test_moves(system_rand,4,epsilon=1e-15)

in_start = 11
out_start = 11
matrix = np.random.rand(N*out_start,N*in_start)
dims_in =  np.array([in_start]*N+[0])
dims_out = np.array([out_start]*N+[0])
T = ToeplitzOperator(matrix, dims_in, dims_out)
S = SystemIdentificationSVD(T,epsilon=1e-12)
system_rand = MixedSystem(S)

print("test start with 0 dim--------------------------------")
move.test_moves(system_rand,4,epsilon=1e-15)
