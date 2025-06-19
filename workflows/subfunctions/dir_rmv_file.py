#import modules=======================
import os
import fnmatch
#=====================================

def dir_rmv_file(path=os.getcwd(),
                 match='',
                 rmHidden=True
                 ):
    #This function create the list of the folder on the path 'path'
    folderList=os.listdir(path)
    # remove hidden files name start with '.'
    if rmHidden==True:
        folderList = [folder for folder in folderList if folder.find('.')>0]
    #if match condition was there
    if match!='':
        folderList=find_matches(match, folderList)
    return(folderList)


def find_matches(pattern, filenames):
    parts = pattern.split('*')
    matches = []

    for filename in filenames:
        if fnmatch.fnmatchcase(filename, pattern):
            matches.append(filename)
        elif len(parts) > 2 and parts[0] == parts[-1] == '':
            match = True
            filename_parts = filename.split(parts[1])
            if len(filename_parts) >= len(parts):
                for i in range(1, len(parts)-1):
                    if not fnmatch.fnmatchcase(filename_parts[i], parts[i]):
                        match = False
                        break
                if match:
                    matches.append(filename)

    return matches
