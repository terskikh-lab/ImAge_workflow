#generate the path to load the result files
def loadPathGenerator(loadPath, projects, loadFolder):
    loadPathList=dict()
    for project in projects:
        loadPathList[project]=loadPath+'/'+project+'/'+loadFolder
    return loadPathList


def multiPathGenerator(loadPath, projects, loadFolder):
    loadPathList=dict()
    for project in projects:
        loadPathList[project]=loadPath+'/'+project+'/'+loadFolder
    return loadPathList