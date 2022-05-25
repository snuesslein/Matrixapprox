import numpy as np
from tvsclib.stage import Stage
from tvsclib.strict_system import StrictSystem
from tvsclib.mixed_system import MixedSystem
import tvsclib.utils as utils
import tvsclib.math as math

import Split
from tvsclib.canonical_form import CanonicalForm

def testSplit():
    T = np.random.rand(16,16)

    sys = Split.initial_mixed(T)
    Split.split_mixed(sys,0,8,8)
    Split.split_mixed(sys,0,4,4)
    Split.split_mixed(sys,2,4,4)

    correct,rep = utils.check_dims(sys,text_output=False,return_report=True)
    assert correct, rep

    assert np.allclose(T,sys.to_matrix()), "T incorrect"


    #Test with sigmas
    sys = Split.initial_sigmas_mixed(T)
    Split.split_sigmas_mixed(sys,0,8,8)
    Split.split_sigmas_mixed(sys,0,4,4)
    Split.split_sigmas_mixed(sys,2,4,4)

    correct,rep = utils.check_dims(sys,text_output=False,return_report=True)
    assert correct, rep
    assert np.allclose(T,sys.to_matrix()), "T incorrect"


    #Test normal forms
    sys_output =Split.get_system_mixed(sys,canonical_form=CanonicalForm.OUTPUT)
    assert np.allclose(T,sys_output.to_matrix()), "T incorrect for output normal"
    assert sys_output.causal_system.is_output_normal(), "Not output normal"

    sys_input =Split.get_system_mixed(sys,canonical_form=CanonicalForm.INPUT)
    assert np.allclose(T,sys_input.to_matrix()), "T incorrect for input normal"
    assert sys_input.causal_system.is_input_normal(), "not input normal"


    sys_bal =Split.get_system_mixed(sys,canonical_form=CanonicalForm.BALANCED)
    assert np.allclose(T,sys_bal.to_matrix()), "T incorrect for balanced"
    assert sys_bal.causal_system.is_balanced(), "not balanced"

    sigmas_causal =[stage.s_in for stage in sys.causal_system.stages][1:]
    sigmas_anticausal =[stage.s_in for stage in sys.anticausal_system.stages][:-1]
    #check sigmas
    (sigmas_causal_refer,sigmas_anticausal_refer) = math.extract_sigmas(T, sys.dims_in,sys.dims_out)
    for i in range(len(sigmas_causal_refer)):
        sig_causal = np.zeros_like(sigmas_causal_refer[i])
        sig_anticausal = np.zeros_like(sigmas_anticausal_refer[i])
        sig_causal[:len(sigmas_causal[i])]=sigmas_causal[i]
        sig_anticausal[:len(sigmas_anticausal[i])]=sigmas_anticausal[i]
        assert np.allclose(sig_causal,sigmas_causal_refer[i]),\
                "Causal sigmas do not match"+str(i)+str(sigmas_causal_c[i])+str(sigmas_causal_refer[i])
        assert np.allclose(sig_anticausal,sigmas_anticausal_refer[i]),\
                "Anticausal sigmas do not match"+str(i)+str(sigmas_anticausal_c[i])+str(sigmas_anticausal_refer[i])

testSplit()
print("Splitting tested")
