#!/usr/bin/python3
#$: chmod 755 yourfile.py
#$: dos2unix yourfile.py
#$: ./yourfile.py
# This is a python script that will generate a LAMMPS molecule file for use in
# Polymer Brush
import math

def BSMolf(N,N_s,M_s,fname):
    # N is the Number of atoms to create along the main chain
    # N_s is the Number of atoms per side chain
    # G is the gap between side chains (0=no gaps, 1=alternating)
    # The sequence of side chains starts with the gap

    #M_s = int(math.floor(N/(G+1)))
    G = int(math.floor(N/M_s)) - 1
    N2 = int(math.ceil(N/2))
    Nt2 = int(math.floor((N + (M_s * N_s)/2)))
    z_ids = [((G+1) + i*(G+1)) for i in range(M_s)]
    z_folded = [z_ids[i] - 1 if z_ids[i] <= N2 else (N - z_ids[i])+1 for i in range(M_s)]
    endBranch = [True if z_ids[-1] == N else False][0]
    # Write LAMMPS data file
    isBB = [True if N_s != 0 else False][0]

    atomN = int(N + (M_s * N_s))
    bondN = int((N - 1) + (M_s * N_s))

    if isBB:
        if endBranch:
            angleN = int((N - 2) + ((M_s-1) * (N_s + 1)) + N_s)
        else:
            angleN = int((N - 2) + (M_s * (N_s + 1)))
    else:
        angleN = int((N - 2))

    with open(fname,'w') as fdata:
        # First line is a description
        fdata.write('Bead-Spring Polymer molecule\n\n')

        #--- Header ---#
        #Specify number of atoms and atom types
        fdata.write('{} atoms\n' .format(atomN))
        #Specify the number of bonds
        fdata.write('{} bonds\n' .format(bondN))
        fdata.write('{} angles\n'.format(angleN))

        #--- Body ---#

        # Coords assignment
        # Write the line format for the Coords:
        # atom-ID x y z

        fdata.write('Coords\n\n')
        # Backbone Atoms
        for i in range(N2):
            fdata.write('{} {} {} {}\n' .format(i+1,0,0,i))
        for i in range(N2,N):
            fdata.write('{} {} {} {}\n' .format(i+1,0,1,N-i))

        # Sidechain Atoms
        branchBases = []
        k = 0
        if isBB:
            for i in range(N,atomN,N_s):
                branchBases.append(i+1)
                for j in range(N_s):
                    if z_ids[k] <= N2:
                        fdata.write('{} {} {} {}\n' .format(i+j+1,j+1,0,z_folded[k]))
                    else:
                        fdata.write('{} {} {} {}\n' .format(i+j+1,j+1,1,z_folded[k]))
                k += 1

        # Type assignment
        # atom-ID type
        fdata.write('Types\n\n')

        # Backbone Atoms
        fdata.write('{} {}\n' .format(1,1))
        for i in range(N-2):
            fdata.write('{} {}\n' .format(i+2,1))
        fdata.write('{} {}\n' .format(N,1))

        # Sidechain Atoms
        if isBB:
            for i in range(N,atomN):
                fdata.write('{} {}\n' .format(i+1,1))

        # Bonds section
        fdata.write('Bonds\n\n')
        # Write the line format for the bonds:
        # bond-ID type atom1 atom2
        # Backbone Bonds
        for i in range(N-1):
            fdata.write('{} 1 {} {}\n' .format(i+1,i+1,i+2))
        # Sidechain Bonds
        a_s = G+1
        tribonds = []
        if isBB:
            for i in range(N-1,bondN,N_s):
                for j in range(N_s):
                    if j==0:
                        fdata.write('{} 1 {} {}\n' .format(i+j+1,a_s,i+j+2))
                    else:
                        fdata.write('{} 1 {} {}\n' .format(i+j+1,i+j+1,i+j+2))
                tribonds.append(a_s)
                a_s = a_s + (G+1)

        fdata.write('Angles\n\n')
        #Backbone Angles
        for i in range(N-2):
            fdata.write('{} 1 {} {} {}\n' .format(i+1,i+1,i+2,i+3))
        if isBB:
            k = 0
            for i in range(N-2,angleN,N_s+1):

                if endBranch and k == len(z_ids)-1:
                    for j in range(N_s):
                        if j == 0:
                            fdata.write('{} 2 {} {} {}\n' .format(i+j+1,z_ids[k]-1,z_ids[k],branchBases[k]))
                        elif j == 1:
                            fdata.write('{} 1 {} {} {}\n'.format(i+j+1, z_ids[k], branchBases[k], branchBases[k] + 1))
                        else:
                            fdata.write('{} 1 {} {} {}\n' .format(i+j+1, branchBases[k]+j-2,branchBases[k]+j-1, branchBases[k]+j))
                else:
                    for j in range(N_s+1):
                        if j == 0:
                            fdata.write('{} 2 {} {} {}\n' .format(i+j+1,z_ids[k]-1,z_ids[k],branchBases[k]))
                        elif j == 1:
                            fdata.write('{} 2 {} {} {}\n' .format(i+j+1,branchBases[k],z_ids[k],z_ids[k]+1))
                        elif j == 2:
                            fdata.write('{} 1 {} {} {}\n'.format(i+j+1, z_ids[k], branchBases[k], branchBases[k] + 1))
                        else:
                            fdata.write('{} 1 {} {} {}\n' .format(i+j+1, branchBases[k]+j-3,branchBases[k]+j-2,branchBases[k]+j-1))
                k += 1
        # Special Bond Counts assignment
        fdata.write('Special Bond Counts\n\n')
        # line syntax: ID N1 N2 N3 where N1 = # of 1-2 bonds & N2 = # of 1-3 bonds & N3 = # of 1-4 bonds
        # Create an array of special bond values for all the atoms
        A = [2] * N
        A[0] = 1
        A[-1] = 1
        B = [0] * N
        if isBB:
            for i in range(1,N+1):
                if i%(G+1) == 0:
                    B[i-1] = 1
        #BB = A+B
        BB = [x + y for x, y in zip(A,B)]
        BBS = BB
        if isBB:
            S = [2] * N_s
            S[-1] = 1
            for i in range(M_s):
                BBS.extend(S)

        # Write the line format for the Special Bonds:
        # ID N1 N2 N3
        for i in range(int(N + (M_s * N_s))):
            fdata.write('{} {} 0 0\n' .format(i+1,BBS[i]))

        if isBB:
            # Create an array of all the atom ids of the first side chain atoms
            SCatom = range(N+1,int(N + (M_s * N_s))+1,N_s)
            # Special Bonds assignment
            fdata.write('Special Bonds\n\n')
            # Write the line format for the Coords:
            # ID a b c d
            j=0
            k=0
            if BBS[0]==1:
                fdata.write('{} {}\n' .format(1,2))
            elif BBS[0]==2:
                fdata.write('{} {} {}\n' .format(1,2,N+1))
            for i in range(1,int(N + (M_s * N_s))):
                if BBS[i] == 1:
                    fdata.write('{} {}\n' .format(i+1,i))
                elif BBS[i] == 2:
                    if i+1 in SCatom: # if they are the start of the side chains
                        fdata.write('{} {} {}\n' .format(i+1,tribonds[j],i+2))
                        j = j + 1
                    else:
                        fdata.write('{} {} {}\n' .format(i+1,i,i+2))
                elif BBS[i] == 3: #3 bonds
                    fdata.write('{} {} {} {}\n' .format(i+1,i,i+2,SCatom[k]))
        else:
            # Special Bonds assignment
            fdata.write('Special Bonds\n\n')
            # Write the line format for the Coords:
            # ID a b c d
            fdata.write('{} {}\n' .format(1,2))
            for i in range(N-2):
                fdata.write('{} {} {}\n' .format(i+2,i+1,i+3))
            fdata.write('{} {}\n' .format(N,N-1))

        return None

if __name__ == '__main__':
    print('BSMolf used')
    # BSMolf(10,3,3,"test.txt")
    # with open('test.txt','r') as f:
    #     for n,line in enumerate(f):
    #         print(line)
