#Description ezsave
#================================================
#save multiple objects in a file
#=================================================
#Kenta Ninomiya @ Sanford burnham prebys medical discovery institute: 2021/08/20

#import modules=======================
import pickle
import h5py
#=====================================
# @jit(nopython=True)
def ezsave(vList,file):
    with open(file, 'wb') as f:
        pickle.dump(vList, f)
        
def ezsave_hdf5(vList,file):
    with h5py.File(file, 'w') as f:
        for key in vList.keys():
            f.create_dataset(key, data=vList[key])
