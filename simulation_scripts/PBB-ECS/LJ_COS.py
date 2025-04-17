#!/usr/bin/python3
#$: chmod 755 yourfile.py
#$: dos2unix yourfile.py
#$: ./yourfile.py
# This is a python script that will generate a LAMMPS molecule file for use in
# Polymer Brush
import math
import numpy as np
# import matplotlib.pyplot as plt

def types_from_pair(pair):
    types = [1 if "W" in l else 2 if "B" in l else 3 if "T" in l else 4 for l in pair]
    return types


def LJ_COS(e,s,e_ab,N,inner,outer,pair):
    print(e,s,e_ab,N,inner,outer,pair)
    steps = ((outer - inner) / N)
    r = np.arange(inner,outer,steps,dtype=float)
    U = np.zeros(r.shape)
    c = pow(2,(1/3))
    # a,b = 3.173072867831619 -0.8562286454415565
    a = 3.173072867831619
    b = -0.8562286454415565
    print(a,b)
    i=0
    rc0 = (s * 1.122462048)
    rc1 = (s * 1.5)
    for i in range(r.shape[0]):
        if r[i] <= rc0:
            sr = s/r[i]
            sr12 = pow(sr,12)
            sr6 = pow(sr,6)
            print(sr12,sr6)
            U[i] = 4 * e * (sr12 - sr6 + 0.25) - e_ab

        elif rc0 < r[i] <= rc1:
            U[i] = ((0.5) * e_ab * (math.cos(a * ((r[i]/s)*(r[i]/s)) + b) - 1))

        else:
            U[i] = 0
    F = - np.gradient(U,r)
    fname = "LJ-COS.txt"
    types = types_from_pair(pair)
    with open(fname,'a') as f:
        f.write('# Pair potential lj/cut for atom types {0} {1}: i,r,energy,force\n\n'.format(types[0],types[1]))
        f.write("{}\n".format(pair))
        f.write('N {0} R {1} {2}\n\n' .format(N,inner,outer))
        for i in range(len(r)):
            f.write('{0} {1} {2} {3}\n' .format(i+1,r[i],U[i],F[i]))
    return r,U

if __name__ == '__main__':
    print('LJ_COS used')
    # #
    # fig,Uax = plt.subplots()
    # r,U = LJ_COS(e=1.0,s=1.0,e_ab=2.0,N=1000.0,inner=0.3,outer=2.5,pair="BS")
    # Uax.plot(r,U)
    # Uax.set_xlim(0,3)
    # Uax.set_ylim(-2.5,40)
    # plt.show()
    # BSMolf(10,3,3,"test.txt")
    # with open('test.txt','r') as f:
    #     for n,line in enumerate(f):
    #         print(line)
