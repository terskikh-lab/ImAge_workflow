"""
Main workflow script for the ImAge analysis pipeline.

Description:
    This script orchestrates the entire ImAge analysis workflow, from raw image
    processing to feature extraction and validation. It is configured via a set
    of variables at the beginning of the file, allowing users to specify project
    names, data paths, and analysis parameters.

    The workflow consists of the following main steps:
    1.  **Illumination Correction (Optional):** Corrects for non-uniform
        illumination in the images. This step is controlled by the
        `illumiCorrection` flag.
    2.  **Segmentation:** Segments nuclei in the images based on a specified
        channel (e.g., DAPI).
    3.  **Feature Extraction:** Extracts various imaging features from the
        segmented nuclei across all specified channels.
    4.  **ImAge Validation:** Performs validation of the ImAge model by training
        and testing on the extracted features.

Configuration:
    - `p` (str): Project name. Used for organizing results.
    - `chs` (List[str]): List of channels to be analyzed.
    - `imageIndex` (Dict[str, str]): Mapping of channel numbers to antibody names.
    - `orgDataLoadPath` (str): Path to the original imaging data.
    - `orgDataSubFolder` (str): Subfolder within the data path containing images.
    - `resultsSavePath` (str): Path where analysis results will be saved.
    - `imageFileRegEx` (re.Pattern): Regular expression to parse image filenames.
    - `imageFileFormat` (str): File format of the images (e.g., '.tiff').
    - `illumiCorrection` (bool): Flag to enable or disable illumination correction.

Usage:
    Configure the variables at the top of the script and then run it from the
    command line. A GPU can be specified as a command-line argument for the
    segmentation step.
    
    Example:
        python main.py 0  # Runs segmentation on GPU 0
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
imageIndex={'ch1':'Channel1',
            'ch2':'Channel2',
            'ch3':'Channel3'}
# orgDataLoadPath='../Data/Original'
orgDataLoadPath='/mnt/m/imaging_data/old_ImAge_publication'
orgDataSubFolder='Images'
resultsSavePath='Data/Results'
# r01c11f01p01-ch1sk1fk1fl1.tiff
imageFileRegEx = re.compile(r"r(?P<raw>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<zposition>\d+)-(?P<channel>ch\d+)sk1fk1fl1\.tiff")
imageFileFormat='.tiff'

illumiCorrection=False #whether to run illumination correction or not

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
                               resultsSavePath='Data/Results',
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
                           segCh='DAPI',
                           illumiCorrection=illumiCorrection,
                           nWorkers=3,
                           voxelDim=[1,0.6,0.6],
                           )


#%% ========================================================================================
'''
feature extraction
'''
from o3_extract_features import o3_extract_features
o3_extract_features(project=p,
                    resultsSavePath=resultsSavePath,
                    contents=chs,
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
                        illumiCorrection=True,
                        contents=chs,
                        seeds=rndVals,
                        meanSize=meanSize,
                        nBoot=1000,
                        sampleGroups=['Passage'],
                        )
