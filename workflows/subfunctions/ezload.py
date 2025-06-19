#Description ezload
#================================================
#load multiple objects in a file
#=================================================
#Kenta Ninomiya @ Sanford burnham prebys medical discovery institute: 2021/08/20

#import modules=======================
import pickle
import os
from EXP_elements.element_easyio import ezload as ezl
#=====================================
# @jit(nopython=True)
def ezload(file):
    #check isf the file format is pickle, pkl
    root, ext = os.path.splitext(file)
    if ext != '.pkl':
        with open(file,'rb') as f:
            return(pickle.load(f))
    else:
        return(ezl(file))
    