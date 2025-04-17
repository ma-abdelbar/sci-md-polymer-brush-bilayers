# This is a python script that reads LAMMPS output files for LPBB-ECS
# And then writes them to .csv files
#!/usr/bin/python3
#$: chmod 755 yourfile.py
#$: dos2unix yourfile.py
#$: ./yourfile.py
import numpy as np
import pandas as pd
import os


def find_lines(filename):
    with open(filename,'r') as f:
        #find Nshear inside log.lammps
        start = []
        Nshear = 0
        for n, line in enumerate(f):
            if ('run         ${Nshear}' in line[:21]):
                    data = np.genfromtxt(filename, dtype=int, skip_header= n+1, max_rows=1)
                    Nshear = data[1]
            if ('Step TotEng' in line[:11]):
                start.append(n)
    return start, Nshear

def find_Ns(filename):
    N_data = np.genfromtxt(filename, dtype=int, skip_header=24, max_rows=7)
    Nequil = N_data[0,3]
    Npre = N_data[1,3]
    Ncomp = N_data[2,3]
    Nthermo = N_data[6,3]
    return Nequil,Npre,Ncomp,Nthermo

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f, 1):
            pass
    return i

def read_chunk(filename):
    with open(filename,'r') as f:
        c_info = np.genfromtxt(filename,comments='#', dtype=float, max_rows=1)
        Nchunks = int(c_info[1])
        lines = file_len(filename)
        for i, line in enumerate(f):
            if i == 0:
                data =  np.genfromtxt(filename, dtype=float, comments='#', usecols= (1,3), skip_header=int(4+(i+(i/Nchunks))) ,max_rows=Nchunks)
            elif (i % Nchunks == 0 and i < (lines-Nchunks-1)  and i > 0):
                get = np.genfromtxt(filename, dtype=float, comments='#', usecols= (3), skip_header=int(4+(i+(i/Nchunks))) ,max_rows=Nchunks)
                data = np.column_stack((data, get))
    return data

def read_mop(filename):
    data= np.genfromtxt(filename, comments='#', usecols=(1,2,3,4), dtype=float, skip_header=4)
    return data


if __name__ == '__main__':
    Nequil,Npre, Ncomp, Nthermo = find_Ns('main.in')

    print(Nequil,Npre,Ncomp,Nthermo)
    logname = 'log.lammps'

    starts, Nshear = find_lines(logname)   # starts will either be of length 4 or 1
    print(starts,Nshear)


    if len(starts) != 1:  # Then this is the first loop and skip = 0 in ecs.in
        # Read in the equilibrium thermo data from log file
        equil_data = np.genfromtxt(logname,comments=None, dtype=float, skip_header=(starts[0]+2), max_rows=(int(Nequil/Nthermo)+1))
        equil_df = pd.DataFrame(equil_data)
        equil_df.to_csv('equil.csv')



        if len(starts) == 4:
            #comp_data = read_thermo('log.lammps', starts[1], Ncomp, Nthermo)
            comp_data = np.genfromtxt(logname,comments=None, dtype=float, skip_header=(starts[2]+1), max_rows=(int(Ncomp/Nthermo)+1))
            comp_df = pd.DataFrame(comp_data)
            comp_df.to_csv('comp.csv')
        else:
            Nmove = Ncomp * 0.1
            Nfix = Ncomp - Nmove
            move_data = np.genfromtxt(logname,comments=None, dtype=float, skip_header=(starts[2]+1), max_rows=(int((Nmove)/Nthermo)+1))
            fix_data = np.genfromtxt(logname,comments=None, dtype=float, skip_header=(starts[3]+1), max_rows=(int((Nfix)/Nthermo)+1))
            comp_data = np.concatenate((move_data,fix_data[1:,:]),axis=0)
            comp_df = pd.DataFrame(comp_data)
            comp_df.to_csv('comp.csv')

        if len(starts) == 4: # Then this is P style
            #comp_data = read_thermo('log.lammps', starts[1], Ncomp, Nthermo)
            shear_data = np.genfromtxt(logname,comments=None, dtype=float, skip_header=(starts[3]+1), max_rows=(int(Nshear/Nthermo)+1))
            shear_df = pd.DataFrame(shear_data)
            shear_df.to_csv('shear.csv')
        else: # Then this is D Style
            shear_data = np.genfromtxt(logname, comments=None, dtype=float, skip_header=(starts[4] + 1),max_rows=(int(Nshear / Nthermo) + 1))
            shear_df = pd.DataFrame(shear_data)
            shear_df.to_csv('shear.csv')


        # Read in the number density profile of bottom beads
        bbdens_data = read_chunk('bbeads_edz')
        bbdens_df = pd.DataFrame(bbdens_data)
        bbdens_df.to_csv('bbdpe.csv')
        # Read in the number density profile for the top beads
        tbdens_data = read_chunk('tbeads_edz')
        tbdens_df = pd.DataFrame(tbdens_data)
        tbdens_df.to_csv('tbdpe.csv')
        # Read in the number density of all the beads combined
        abdens_data = read_chunk('abeads_edz')
        abdens_df = pd.DataFrame(abdens_data)
        abdens_df.to_csv('abdpe.csv')
        # Read in the number density profile of bottom beads
        bbdens_data = read_chunk('bbeads_cdz')
        bbdens_df = pd.DataFrame(bbdens_data)
        bbdens_df.to_csv('bbdpc.csv')
        # Read in the number density profile for the top beads
        tbdens_data = read_chunk('tbeads_cdz')
        tbdens_df = pd.DataFrame(tbdens_data)
        tbdens_df.to_csv('tbdpc.csv')
        # Read in the number density of all the beads combined
        abdens_data = read_chunk('abeads_cdz')
        abdens_df = pd.DataFrame(abdens_data)
        abdens_df.to_csv('abdpc.csv')


    else: #then these are the shear runs with skip = 1 in ecs.in
        shear_data = np.genfromtxt(logname,comments=None, dtype=float, skip_header=(starts[0]+2), max_rows=(int(Nshear/Nthermo)+1))
        shear_df = pd.DataFrame(shear_data)
        shear_df.to_csv('shear.csv')

    # Read in the number density profile of bottom beads
    bbdens_data = read_chunk('bbeads_sdz')
    bbdens_df = pd.DataFrame(bbdens_data)
    bbdens_df.to_csv('bbdps.csv')
    # Read in the number density profile for the top beads
    tbdens_data = read_chunk('tbeads_sdz')
    tbdens_df = pd.DataFrame(tbdens_data)
    tbdens_df.to_csv('tbdps.csv')
    # Read in the number density of all the beads combined
    abdens_data = read_chunk('abeads_sdz')
    abdens_df = pd.DataFrame(abdens_data)
    abdens_df.to_csv('abdps.csv')

    #Read in the velocity profile for shear
    velp_data = read_chunk('velp_sz')
    velp_df = pd.DataFrame(velp_data)
    velp_df.to_csv('velps.csv')

    #Read in the velocity profile for shear
    temp_data = read_chunk('temp_sz')
    temp_df = pd.DataFrame(temp_data)
    temp_df.to_csv('temps.csv')
