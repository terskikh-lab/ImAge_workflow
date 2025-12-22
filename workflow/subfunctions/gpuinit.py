
import tensorflow as tf
import random
import platform
def gpuinit(gpuN=None):
    if gpuN is False:
        tf.config.set_visible_devices([], 'GPU')
        return
    if platform.system()!='Darwin':
        gpus = tf.config.list_physical_devices(device_type='GPU')
        if len(gpus)==0:
            tf.config.set_visible_devices([], 'GPU')
        else:
            if gpuN!=None:
                gpuIdx=gpuN
            else:    
                gpuIdx=random.sample(list(range(0,len(gpus))), 1)[0]
            tf.config.experimental.set_memory_growth(gpus[gpuIdx], True)
            tf.config.set_visible_devices(gpus[gpuIdx], 'GPU')