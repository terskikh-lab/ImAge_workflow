#import modules=======================
import os
import fnmatch
#=====================================

def dir_rmv_folder(path=os.getcwd(),
                   match=''):
    #This function create the list of the folder on the path 'path'
    folderList=os.listdir(path)
    # remove hidden files name start with '.'
    cleanList = [folder for folder in folderList if folder.find('.')<0]
    #if match condition was there
    if match!='':
        cleanList = fnmatch.filter(cleanList, match)
    return(cleanList)

