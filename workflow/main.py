"""
ImAge workflow runner

Overview:
    Orchestrates illumination correction (optional), nuclei segmentation,
    feature extraction, and ImAge axis validation. Configure the variables in
    the "configuration" block below.

Inputs and paths:
    - orgDataLoadPath (str): Absolute or workspace-relative path to the dataset root.
    - orgDataSubFolder (str): Subfolder that contains images (default: 'Images').
    - resultsSavePath (str): Output root for pipeline results (default: 'Data/Results').
    - imageFileRegEx (re.Pattern): Regex for parsing image filenames (e.g.,
      matching 'r01c11f01p01-ch1sk1fk1fl1.tiff').
    - imageFileFormat (str): Image extension (e.g., '.tiff').

Project settings:
    - p (str): Project name used for organizing outputs.
    - chs (List[str]): Channels to analyze for feature extraction (e.g., ['DAPI', 'H3K27me3']).
    - imageIndex (Dict[str, str]): Map from 'chN' to a human-readable channel name.
    - segmentation_ch (str): Channel used for nuclei segmentation (default: 'DAPI').
    - illumiCorrection (bool): Whether to run illumination correction (default: False).
    - voxel_dim (List[float]): [z, x, y] voxel sizes (µm).

CLI usage:
    Run the script from the repository root with an optional GPU index for the
    segmentation step. If omitted, CPU or the default GPU is used.

        python workflows/main.py          # CPU/default GPU
        python workflows/main.py 0        # use GPU 0

Pipeline stages:
    1) Illumination correction (optional):
       Builds per-channel flat-field/dark-field and corrects images when
       `illumiCorrection` is True.
    2) Segmentation:
       Detects nuclei on `segmentation_ch` and writes masks/labels.
    3) Feature extraction:
       Computes per-nucleus features for channels in `chs`.
    4) ImAge validation:
       Constructs the ImAge axis and evaluates via bootstrapped validation.

Notes:
    - Seeds for bootstrap are deterministically generated for reproducibility.
    - Tune nWorkers values per step based on your machine to avoid oversubscription.
"""
#exeP_Reprogramming.py
import re

'''
configuration
'''
p='brain_3ages_k27me3' #project name, this will be used to save the results
chs=['DAPI',
     'H3K27me3',
     'H3K27ac',
    #  'H3K9ac'
     ] #list of channels to be analyzed. These channels will be used to extract imaging features
imageIndex={'1':'Channel1',
            '2':'Channel2',
            '3':'Channel3'}
orgDataLoadPath='../Data/Original'
orgDataSubFolder='Images'
resultsSavePath='Data/Results'
# r01c01f01p01-ch1sk1fk1fl1.tiff
imageFileRegEx = re.compile(r"r(?P<raw>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<zposition>\d+)-ch(?P<channel>ch\d+)sk1fk1fl1\.tiff")
imageFileFormat='.tiff'

segmentation_ch='DAPI'
illumiCorrection=False #whether to run illumination correction or not
voxel_dim=[1,0.6,0.6] #z, x, y

#%% ========================================================================================
'''
Illumination correction 
Optional. However recommended to run this step first when data consists of more than 25 wells
'''
if illumiCorrection:
    from o1_illumination_correction import o1_illumination_correction
    o1_illumination_correction(project=p,
                               orgDataLoadPath=orgDataLoadPath,
                               orgDataSubFolder=orgDataSubFolder,
                               resultsSavePath=resultsSavePath,
                               imageFileRegEx=imageFileRegEx,
                               imageFileFormat=imageFileFormat,
                               nWorkers=10
                               )

#%% ========================================================================================
'''
Segmentation
'''
import sys
try:
    gpuN=int(sys.argv[1])
except:
    gpuN=None
    
from subfunctions.gpuinit import gpuinit
gpuinit(gpuN=gpuN)

from o2_segmentation import o2_segmentation
o2_segmentation(project=p,
                orgDataLoadPath=orgDataLoadPath,
                orgDataSubFolder=orgDataSubFolder,
                resultsSavePath=resultsSavePath,
                imageFileRegEx=imageFileRegEx,
                imageFileFormat=imageFileFormat,
                imageIndex=imageIndex,
                segCh=segmentation_ch,
                illumiCorrection=illumiCorrection,
                nWorkers=3,
                voxelDim=voxel_dim,
                )


#%% ========================================================================================
'''
feature extraction
'''
from o3_extract_features import o3_extract_features
o3_extract_features(project=p,
                    resultsSavePath=resultsSavePath,
                    contents=chs,
                    segCh=segmentation_ch,
                    illumiCorrection=illumiCorrection,
                    nWorkers=50)
        

#%% ========================================================================================
'''
ImAge axis construction and prediction with validation
'''
    
from o4_ImAge_validation import o4_ImAge_validation
import random
#generate 10 random intenger values using
rndVals=[]
for i in range(100):
    rndVals.append(random.Random(i).randint(0,10000))
            
for meanSize in [10]:
    o4_ImAge_validation(projects=[p],
                        segCh=segmentation_ch,
                        illumiCorrection=illumiCorrection,
                        contents=chs,
                        seeds=rndVals,
                        meanSize=meanSize,
                        nBoot=1000,
                        )
