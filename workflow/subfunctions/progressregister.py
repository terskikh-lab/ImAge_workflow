import os
import time
import random

#returnFileType
#returnVals: 0: have not started, 1: file exists, 2: progress idx exists

def progressregister(saveFileName0,idxFileName0,produceidx=True, recheck=True, returnFileType=False, verbose=False):
    returnVals=[False, True, True]
    if returnFileType:
        returnVals=[0,1,2]
        
    # #Check the existence of results (if exists, calculation is skipped)=======================
    if os.path.exists(saveFileName0):
        if verbose:
            print(saveFileName0+' exists')
        return(returnVals[1])
        
    elif os.path.exists(idxFileName0):
        if verbose:
            print(saveFileName0+' calculation ongoing')
        return(returnVals[2])
    
    if recheck==True:
        # recheck the current comutation status after short stop
        #-------random nap for avoiding multiple access----------#
        #sleep for 0-0.5 seconds
        time.sleep(random.random()/2)
        #--------------------------------------------------------#
        if os.path.exists(saveFileName0):
            if verbose:
                print(saveFileName0+' exists')
            return(returnVals[1])
        elif os.path.exists(idxFileName0):
            if verbose:
                print(saveFileName0+' calculation ongoing')
            return(returnVals[2])
    # #==========================================================================================
    
    if produceidx==True:
        open(idxFileName0, 'a').close()
    if verbose:
        print(saveFileName0+' starts')
    return(returnVals[0])



