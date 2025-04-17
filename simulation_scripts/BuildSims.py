#!/usr/bin/python
#$: chmod 755 yourfile.py
#$: dos2unix yourfile.py
#$: ./yourfile.py

import os
import shutil

def make_dirs(vals,name):
    for v in vals:
        folderName = name + str(v)
        if os.path.isdir(folderName) == False:
            os.mkdir(folderName)

    return None

def myFindLine(filename,keywords):
    with open(filename, 'r') as f:
        count = 0
        for n, line in enumerate(f):
            checks = [True if (keywords[i] in line) else False for i in range(len(keywords))]
            if all(checks):
                fLine=n
                count+=1
        if(count != 1):
            print('Be more specific: {0} results'.format(count))
    return fLine
def myReadLines(filename, rlines):
    lines = []
    with open(filename, 'r') as f:
        for n, line in enumerate(f):
            if n in rlines:
                lines.append(line)
        f.close()
    return lines

def mod_main(l,line):
    f=open('main.in','r')
    lines = f.readlines()
    f.close()
    lines[l] = line
    f2 = open('main.in','w')
    f2.writelines(lines)
    f2.close()

    return None

def myFilter(dir):
    if (dir != 'PBB-ECS') and (dir != '.DS_Store'):
        v = int(dir.split("=")[1])
    else:
        v = dir
    # print('filter',dir, v)
    return v


if __name__ == '__main__':

    # To run the sims on Home PC go to any L directory and run
    # for d in ./P*/LPBB-ECS; do (cd "d" && lmp -in main.in);
    # To run on cx1 I imagine something like
    # for d in ./N*/M*/L*/P*/LPBB-ECS; do (cd "d" && qsub lmpJS.pbs); done;

    owd = os.getcwd()

    Wall_control_l = myFindLine(owd + r'/PBB-ECS/main.in', ['variable ',' Wall_control '])
    Wall_control = myReadLines(owd + r'/PBB-ECS/main.in',[Wall_control_l])[0].split('variable       Wall_control string ')[1][0]
    Lx = 30
    Ly = 25
    Axy = Lx * Ly

    seriesName= 'Sims'
    if os.path.isdir(seriesName+r'/PBB-ECS') == False:
        shutil.copytree('PBB-ECS',seriesName+r'/PBB-ECS')
    os.chdir(seriesName)



    Ns = [30,40,50,60]                                       # [20,30,40,50,60,70,80,90,100,120]
    Ms = [6,12,18,38,44,58,64,78,100,158,208]                                       # [20,46,72,100,126,152,178,204,230,258]
    Xs = [0]                                       # [0,3]
    PD = ['D=' if 'D' in Wall_control else 'P='][0]
    PDs = [10,20,30,40,50,60,70]                                     # [8,10,14,20,28,38,50,64,80]
    rho_melt = 0.8

    make_dirs(Ns,'N=')
    Ndirs = os.listdir('.')
    Ndirs.sort(key=myFilter)
    # print(os.getcwd())
    Ni = 0
    for Ndir in Ndirs:
        if (Ndir != 'PBB-ECS') and (Ndir != '.DS_Store'):
            if os.path.isdir(Ndir+r'/PBB-ECS') == False:
                shutil.copytree('PBB-ECS',Ndir+r'/PBB-ECS')

            os.chdir(Ndir)
            make_dirs(Ms,'M=')
            os.chdir('PBB-ECS')
            v = Ndir.split("=")[1]
            mod_main(2, 'variable      N      equal  '+v+'\n')
            os.chdir('..')

            Mdirs = os.listdir('.')
            Mdirs.sort(key=myFilter)
            # print('sorted',Mdirs)

            ################################
            Mi = 0
            for Mdir in Mdirs:
                if (Mdir != 'PBB-ECS') and (Mdir != '.DS_Store'):
                    if os.path.isdir(Mdir+r'/PBB-ECS') == False:
                        shutil.copytree('PBB-ECS',Mdir+r'/PBB-ECS')
                    os.chdir(Mdir)
                    make_dirs(Xs,'X=')

                    os.chdir('PBB-ECS')
                    v = Mdir.split("=")[1]
                    mod_main(1, 'variable      M      equal  ' +v+'\n')
                    os.chdir('..')

                    Xdirs = os.listdir('.')
                    Xdirs.sort(key=myFilter)
                    # print(Xdirs)
                    #####################################################
                    Xi = 0
                    for Xdir in Xdirs:
                        if (Xdir != 'PBB-ECS') and (Xdir != '.DS_Store'):
                            if os.path.isdir(Xdir+r'/PBB-ECS') == False:
                                shutil.copytree('PBB-ECS',Xdir+r'/PBB-ECS')

                            os.chdir(Xdir)
                            make_dirs(PDs,PD)

                            os.chdir('PBB-ECS')
                            v = Xdir.split("=")[1]
                            mod_main(37, 'variable       X      equal  ' +v+ '\n')
                            os.chdir('..')

                            PDdirs = os.listdir('.')
                            PDdirs.sort(key=myFilter)
                            # print(PDdirs)
                            ###############################################
                            PDi = 0
                            for PDdir in PDdirs:
                                if (PDdir != 'PBB-ECS') and (PDdir != '.DS_Store'):
                                    if os.path.isdir(PDdir+r'/PBB-ECS') == False:
                                        shutil.copytree('PBB-ECS',PDdir+r'/PBB-ECS')

                                    os.chdir(PDdir)
                                    os.chdir('PBB-ECS')
                                    v = PDdir.split("=")[1]
                                    if "D=" in PD:

                                        mod_main(39, 'variable       Dcomp      equal  ' +v+ '\n')
                                        N_solvent = int(rho_melt * Axy * PDs[PDi] - (2*Ns[Ni]*Ms[Mi]))
                                        print(os.getcwd(),'N=',Ns[Ni], 'M=',Ms[Mi], 'D=',PDs[PDi], 'Solvent=', str(N_solvent))
                                        mod_main(23, 'variable      N_solv    equal ' +str(N_solvent)+ '\n')
                                    else:
                                        mod_main(40, 'variable       Pcomp      equal  ' +v+ '\n')

                                    os.chdir('..')

                                    os.chdir('..')
                                    PDi += 1
                            ###############################################
                            os.chdir('..')
			    Xi += 1

                    #####################################################
                    os.chdir('..')
		    Mi += 1


            #################################
            os.chdir('..')
	    Ni += 1


    os.chdir(owd)
