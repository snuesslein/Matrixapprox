import numpy as np
import matplotlib.pyplot as plt


def invert_permutation(P):
    P_inv = np.zeros_like(P)
    for i in range(len(P)):
        P_inv[P[i]]=i
    return P_inv

def invert_permutations(Ps):
    Ps_inv = np.zeros_like(Ps)
    for (j,i), x in np.ndenumerate(Ps):
        Ps_inv[j,Ps[j,i]]=i
    return Ps_inv


def connection_plot(P1_inv,P2_inv,start=0,end=1,colors = None,flipxy=False,linewidth=5):
    #cosine
    #v = np.linspace(0,1)
    #y = (np.cos(np.pi*v)*0.5)+0.5
    #x = v*(end-start)+start

    #bezier
    t = np.linspace(0,1)
    a =-t**3+3*t**2-3*t+1
    b =3*t**3-6*t**2+3*t
    c =-3*t**3+3*t**2
    d =t**3

    x = 0*a +0.5*b +0.5*c +1*d
    y = 1*a +  1*b +  0*c +0*d
    x = x*(end-start)+start

    f_straight=0.1


    if colors is None:
        cmap = plt.cm.get_cmap('viridis')
        colors = cmap(np.linspace(0,1,len(P1_inv)))
    elif type(colors)==str:
        colors = [colors]*len(P1_inv)
    for p1,p2,c in zip(P1_inv,P2_inv,colors):
        if flipxy:
            plt.plot(y*(p1-p2)+p2,x,color = c,linewidth=linewidth)
        else:
            plt.plot(x,y*(p1-p2)+p2,color = c,linewidth=linewidth)

def multiple_connection_plot(Ps_inv,start=0,end=1,**kwargs):
    N = Ps_inv.shape[0]
    d = (end-start)/(N-1)
    for n in range(N-1):
        connection_plot(Ps_inv[n],Ps_inv[n+1],start=start+d*n,end=start+d*(n+1),**kwargs)
        
        
