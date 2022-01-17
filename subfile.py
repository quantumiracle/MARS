import os, fnmatch
import shutil, errno

folder = 'data/model/'

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

def iters_filter(folder, max_itr):
    """ filter out index larger than max_itr in folder """
    files = os.listdir(folder)
    for f in files:
        try:
            indx = f.split('_')[0]
            if int(indx) > max_itr:
                for file in os.listdir(folder):
                    if fnmatch.fnmatch(file, '*indx*'):
                        os.remove(file)
        except:
            print(f'{f} is exceptional.')

def subfiles_generate(filename, method):
    olddir = folder+filename

    for i in range(1, 11):
        idx = 10000 * i
        
        newpath = olddir + f'/epi_{idx}/'
        oldpath = olddir + f'/mdp_arbitrary_mdp_{method}/'

        print(oldpath, newpath)

        copyanything(oldpath, newpath)
        # iters_filter(newpath, 20000)

        # replace the meta policy
        os.remove(newpath+'meta_strategies.npy')
        os.remove(newpath+'policy_checkpoints.npy')

        os.rename(newpath+f'{idx}_meta_strategies.npy', newpath+'meta_strategies.npy')
        os.rename(newpath+f'{idx}_policy_checkpoints.npy', newpath+'policy_checkpoints.npy')


subfiles_generate('20220117153310', method='fictitious_selfplay2')
# subfiles_generate('20220116204408', method='nxdo2')