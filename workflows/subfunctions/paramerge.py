import pandas as pd

def paramerge(paramPath, projects):
    paramList=list()
    for project in projects:
        #load the parameter file as string and convert to data frame
        tmpParam=pd.read_csv(paramPath+'/'+project+'.csv', dtype=str)
        #get the project name and the well number
        wellNum=[project+'_'+('00'+wellIdx)[-6:] for wellIdx in tmpParam['WellIndex']] #convert well number from alphabet-number to number format
        tmpParam['WellIndex']=[('00'+wellIdx)[-6:] for wellIdx in tmpParam['WellIndex']] #convert well number from alphabet-number to number format
        tmpParam.index=wellNum
        paramList.append(tmpParam)
    paramDF=pd.concat(paramList,axis=0,sort=False)
    return paramDF