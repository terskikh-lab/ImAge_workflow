import os

def idxremover(idxFileName0):
    try:
        os.remove(idxFileName0)
    except:
        print('idx was already deleted/ doesn\'t exist so just ignored it')
    