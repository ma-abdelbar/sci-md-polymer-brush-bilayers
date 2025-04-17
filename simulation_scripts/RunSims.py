#!/usr/bin/python
#$: chmod 755 yourfile.py
#$: dos2unix RunSims.py
#$: ./yourfile.py


import os
import shutil
import glob
import argparse

def myFindLine(filename,keywords):
    with open(filename, 'r') as f:
	line_numbers = []
        for n, line in enumerate(f):
            checks = [True if (keywords[i] in line) else False for i in range(len(keywords))]
            if all(checks):
                line_numbers.append(n)
    return line_numbers
def mod_main(l,line):
    f=open('main.in','r')
    lines = f.readlines()
    f.close()
    lines[l] = line
    f2 = open('main.in','w')
    f2.writelines(lines)
    f2.close()
    return None
def make_dirs(vals,name):
    for v in vals:
        folderName = name + str(v)
        os.mkdir(folderName)

    return None
def compare_dirs(master,path):
    mast_list = os.listdir(master)
    dir_list = os.listdir(path)
    intersect = list(set(dir_list).intersection(set(mast_list)))
    # print(intersect)
    if len(intersect) == len(mast_list):
        same = True
    else:
        same = False
    return same

def myFilter(dir):
    if (dir != 'PBB-ECS') and (dir != '.DS_Store'):
        v = int(dir.split("=")[1])
    else:
        v = dir
    # print('filter',dir, v)
    return v


def run_sims(type,b1,b2):
    owd = os.getcwd()
    fname = 'PBB-ECS/main.in'
    var = 'Vwalli'
    vLine = myFindLine(fname, ['variable ', ' {0} '.format(var)])[0]
    var2 = 'rest'
    restLines = myFindLine(fname, ['variable ', ' {0} '.format(var2)])
    master_path = os.path.join(owd,'Sims','N=30','M=44','X=0','D=20','PBB-ECS','V=1')
    f_root=open(fname,'r')
    line_root = f_root.readlines()[vLine].split('#')[0]
    vs_root = line_root.split('index ')[1].split()
    header = line_root.split('x ')[0]+'x '
    nvs_root = len(vs_root)
    count = 0
    os.chdir('Sims')

    if os.path.exists(".DS_Store"):
        os.remove(".DS_Store")

    Ndirs = os.listdir('.')
    Ndirs.sort(key=myFilter)
    i = 1
    # master_flag = True
    healthy = False
    runs_list = []
    for Ndir in Ndirs:
        if Ndir != 'PBB-ECS':
            os.chdir(Ndir)
            if os.path.exists(".DS_Store"):
                os.remove(".DS_Store")
            Mdirs = os.listdir('.')
            Mdirs.sort(key=myFilter)
            for Mdir in Mdirs:
                if Mdir != 'PBB-ECS':
                    os.chdir(Mdir)
                    if os.path.exists(".DS_Store"):
                        os.remove(".DS_Store")
                    Ldirs = os.listdir('.')
                    Ldirs.sort(key=myFilter)
                    for Ldir in Ldirs:
                        if Ldir != 'PBB-ECS':
                            os.chdir(Ldir)
                            if os.path.exists(".DS_Store"):
                                os.remove(".DS_Store")
                            Pdirs = os.listdir('.')
                            Pdirs.sort(key=myFilter)
                            for Pdir in Pdirs:
                                if Pdir != 'PBB-ECS':
                                    os.chdir(Pdir)
                                    os.chdir('PBB-ECS')
                                    # print(os.getcwd())
                                    status = 'E'
                                    truncated = False
                                    Vdirs = filter(os.path.isdir,os.listdir('.'))
                                    # if master_flag:
                                    #     master_path = os.path.join(owd,'Sims',Ndirs[0],Mdirs[0],Ldirs[1],Pdirs[0],'PBB-ECS',Vdirs[0])
                                    #     master_flag = False
                                    checks = []
                                    for Vdir in Vdirs:
                                        checks.append(compare_dirs(master_path,Vdir))

                                    nvs_leaf = len(Vdirs)

                                    try:
                                        with open('shear.log','r') as f:
                                            for n, line in enumerate(f):
                                                pass
                                        final_line = line
                                    except:
                                        final_line = 'shear.log missing'

                                    restart = final_line[0:8].strip().isdigit()

                                    if os.path.exists('run_check') == True:
                                        status = 'R'
                                    if os.path.exists('EC.rst') or os.path.exists('log.lammps'):
                                        status = 'B'
                                    if os.path.exists('ECS.rst'):
                                        status = 'I'

                                    if os.path.exists('final_state.rst') == True:
                                        status = 'C'
                                    if i <= b2 and i >= b1 and type == 'f':
                                        status = 'F'
                                        mod_main(restLines[0],'variable       rest     equal 0\n')
                                        mod_main(restLines[1],'variable      rest      equal 0\n')
                                    if i <= b2 and i >= b1 and status == 'I' and type == 'p':
                                        vs_leaf_o = vs_root[:nvs_leaf]
                                        vs_leaf_n = vs_root[nvs_leaf:]
                                        vstr_leaf_n = ' '.join(vs_leaf_n)
                                        line_leaf = header + vstr_leaf_n + '\n'
                                        mod_main(vLine,line_leaf)
                                        if restart:
                                            mod_main(restLines[0],'variable       rest     equal 1\n')
                                        else:
                                            mod_main(restLines[0],'variable       rest     equal 0\n')


                                    if i <= b2 and i >= b1 and (status == 'E' or status == 'I') and type != 'p':
                                        status = 'Q'
                                        count += 1
                                        if type == "a":
                                            runs_list.append(os.getcwd())
                                        else:
                                            #print('Single Submit')
                                            os.system('chmod 755 BSMolBuilder.py')
                                            os.system('qsub serJS.pbs')
                                    try:
                                        f_leaf=open('main.in','r')
                                        line_leaf_m = f_leaf.readlines()[vLine].split('#')[0]
                                        vs_leaf_m = line_leaf_m.split('index ')[1].split()
                                        nvs_leaf_m = len(vs_leaf_m)
                                    except:
                                        pass
                                    print([i,os.getcwd(),glob.glob('*JS.pbs.o*'),status,nvs_leaf,nvs_leaf_m,'restart',restart,'healthy',all(checks)])
                                    # print(final_line)
                                    i = i + 1
                                    os.chdir('..')
                                    os.chdir('..')
                            os.chdir('..')
                    os.chdir('..')
            os.chdir('..')

    os.chdir(owd)
    print("{0} Sims to run".format(count))

    if type == "a":
        print("Writing List")
        textfile = open("runs_list", "w")
        for run_path in runs_list:
            textfile.write(run_path + "\n")
        textfile.close()
        print("Modifiying arrayJS.pbs")
        arrayLine = myFindLine('arrayJS.pbs',['#PBS','-J'])[0]
        fin = open("arrayJS.pbs", "r")
        list_of_lines = fin.readlines()
        list_of_lines[arrayLine] = "#PBS -J 1-{0}\n".format(count)
        fout = open("arrayJS.pbs", "w")
        fout.writelines(list_of_lines)
        fout.close()

    return None



if __name__ == '__main__':

    # To run the sims on Home PC go to any L directory and run
    # for d in ./P*/PBB-ECS; do (cd "d" && lmp -in main.in);
    # To run on cx1 I imagine something like
    # for d in ./N*/M*/L*/P*/PBB-ECS; do (cd "d" && qsub lmpJS.pbs);
    parser = argparse.ArgumentParser()
    parser.add_argument("type", help= "Starting Sim", type = str)
    parser.add_argument("start", help= "Starting Sim", type = int)
    parser.add_argument("end", help= "Ending Sim", type = int)
    args = parser.parse_args()
    run_sims(args.type,args.start,args.end)
