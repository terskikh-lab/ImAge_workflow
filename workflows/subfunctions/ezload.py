#Description ezload
#================================================
#load multiple objects in a file
#=================================================
#Kenta Ninomiya @ Sanford burnham prebys medical discovery institute: 2021/08/20

#import modules=======================
import pickle
#=====================================
# @jit(nopython=True)
def ezload(file):
    with open(file,'rb') as f:
        return(pickle.load(f))