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
imageIndex={'ch1':'Channel1PrimaryAntibody',
            'ch2':'Channel2PrimaryAntibody',
            'ch3':'Channel3PrimaryAntibody'}
# orgDataLoadPath='../Data/Original'
orgDataLoadPath='/mnt/m/imaging_data/old_ImAge_publication'
orgDataSubFolder='Images'
resultsSavePath='Data/Results'
# r01c11f01p01-ch1sk1fk1fl1.tiff
imageFileRegEx = re.compile(r"r(?P<raw>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<zposition>\d+)-(?P<channel>ch\d+)sk1fk1fl1\.tiff")
imageFileFormat='.tiff'

illumiCorrection=False #whether to run illumination correction or not

#%% ========================================================================================
# '''
# Illumination correction 
# Optional. However recommended to run this step first when data consists of more than 25 wells
# '''
# if illumiCorrection:
    # from o1_illumination_correction import o1_illumination_correction
    # o1_illumination_correction(project=p,
    #                            orgDataLoadPath=orgDataLoadPath,
    #                            orgDataSubFolder=orgDataSubFolder,
    #                            resultsSavePath='Data/Results',
    #                            imageFileRegEx=imageFileRegEx,
    #                            imageFileFormat=imageFileFormat,
    #                            nWorkers=10
    #                            )

#%% ========================================================================================
'''
Segmentation
'''
# import sys
# try:
#     gpuN=int(sys.argv[1])
# except:
#     gpuN=None
    
# from subfunctions.gpuinit import gpuinit
# gpuinit(gpuN=gpuN)

# from o2_segmentation import o2_segmentation
# o2_segmentation(project=p,
#                 orgDataLoadPath=orgDataLoadPath,
#                 orgDataSubFolder=orgDataSubFolder,
#                 resultsSavePath=resultsSavePath,
#                            imageFileRegEx=imageFileRegEx,
#                            imageFileFormat=imageFileFormat,
#                            imageIndex=imageIndex,
#                            segCh='DAPI',
#                            illumiCorrection=illumiCorrection,
#                            nWorkers=3,
#                            voxelDim=[1,0.6,0.6],
#                            )


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
        
    
    
# from o4_ImAge_validation import o4_ImAge_validation
# import random
# #generate 10 random intenger values using
# rndVals=[]
# for i in range(100):
#     rndVals.append(random.Random(i).randint(0,10000))
            
# for meanSize in [10]:
#     o4_ImAge_validation(projects=[p],
#                         illumiCorrection=True,
#                         contents=chs,
#                         seeds=rndVals,
#                         meanSize=meanSize,
#                         statParas=['TAS'],
#                         sampleGroups=['Passage'],
#                         )
        
# from fig_s2_o5_randboot_EpiAge_lsvm_VIOLIN_TTCOMP import fig_s2_o5_randboot_EpiAge_lsvm_VIOLIN_TTCOMP
# for p in ps:
#     # for meanSize in [200, 100, 50, 10]:
#     for meanSize in [200]:
#         fig_s2_o5_randboot_EpiAge_lsvm_VIOLIN_TTCOMP(projects=[p],
#                                             contents=chs,
#                                             meanSize=meanSize,)        

# # from s2_o5_randboot_EpiAge_lsvm_singlecell import s2_o5_randboot_EpiAge_lsvm_singlecell
# # import random
# # #generate 10 random intenger values using
# # rndVals=[]
# # for i in range(100):
# #     rndVals.append(random.Random(i).randint(0,10000))
    
# # for p in ps:
# #     for meanSize in [200]:
# #         for sd in rndVals:
# #             s2_o5_randboot_EpiAge_lsvm_singlecell(projects=[p],
# #                                                 illumiCorrection=True,
# #                                                 contents=chs,
# #                                                 seed=sd,
# #                                                 meanSize=meanSize,
# #                                                 sampleGroups=['Passage'],
# #                                                 )
            

# from fig_s2_o5_randboot_EpiAge_lsvm_singlecell_VIOLIN import fig_s2_o5_randboot_EpiAge_lsvm_singlecell_VIOLIN
# for p in ps:
#     for meanSize in [200]:
#         fig_s2_o5_randboot_EpiAge_lsvm_singlecell_VIOLIN(projects=[p],
#                                             contents=chs,
#                                             meanSize=meanSize,)

# # from fig_s2_o5_randboot_EpiAge_lsvm_singlecell_hUMAP import fig_s2_o5_randboot_EpiAge_lsvm_singlecell_hUMAP
# # for p in ps:
# #     for meanSize in [200]:
# #         fig_s2_o5_randboot_EpiAge_lsvm_singlecell_hUMAP(projects=[p],
# #                                             contents=chs,
# #                                             meanSize=meanSize,)
        

# # from fig_s2_o5_randboot_EpiAge_lsvm_singlecell_UMAP import fig_s2_o5_randboot_EpiAge_lsvm_singlecell_UMAP
# # for p in ps:
# #     for meanSize in [200]:
# #         fig_s2_o5_randboot_EpiAge_lsvm_singlecell_UMAP(projects=[p],
# #                                             contents=chs,
# #                                             meanSize=meanSize,)
        

# # from fig_s2_o5_randboot_EpiAge_lsvm_singlecell_UMAP_annotation import fig_s2_o5_randboot_EpiAge_lsvm_singlecell_UMAP_annotation
# # for p in ps:
# #     for meanSize in [200]:
# #         fig_s2_o5_randboot_EpiAge_lsvm_singlecell_UMAP_annotation(projects=[p],
# #                                                                     contents=chs,
# #                                                                     meanSize=meanSize,)
        