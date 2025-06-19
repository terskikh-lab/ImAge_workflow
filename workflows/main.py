#exeP_Reprogramming.py
import re

'''
configuration
'''
p='brain_3ages_k27me3' #project name, this will be used to save the results
chs=['DAPI',
     'H3K27me3',
     'H3K27ac',
     'H3K9ac'] #list of channels to be analyzed. These channels will be used to extract imaging features
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

#%% ========================================================================================
# '''
# Illumination correction 
# Optional. However recommended to run this step first when data consists of more than 25 wells
# '''
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
                           illumiCorrection=False,
                           nWorkers=1,
                           )

# import sys
# try:
#     gpuN=int(sys.argv[1])
# except:
#     gpuN=None
    
# from s2_o2_BSC_p3Dsegmentation import gpuinit
# gpuinit(gpuN=gpuN)
# from s2_o2_BSC_p3Dsegmentation import s2_o2_BSC_p3Dsegmentation
# for p in ps:
#     s2_o2_BSC_p3Dsegmentation(project=p,segCh='DAPI',illumiCorrection=True)
    

# from fig_s2_o2_BSC_segmentation_MOVIEWHOLECOMPARE import fig_s2_o2_BSC_segmentation_MOVIEWHOLECOMPARE
# for p in ps:
#     fig_s2_o2_BSC_segmentation_MOVIEWHOLECOMPARE(project=p,segCh='DAPI',illumiCorrection=True)

# from fig_s2_o2_BSC_segmentation_MOVIEWHOLECOMPARE_LABEL import fig_s2_o2_BSC_segmentation_MOVIEWHOLECOMPARE_LABEL
# for p in ps:
#     fig_s2_o2_BSC_segmentation_MOVIEWHOLECOMPARE_LABEL(project=p,segCh='DAPI',
#                                                        illumiCorrection=True,
#                                                        orgDataLoadPath='../Data/Original/OSKMtissueLiver/Images',)
# from fig_s2_o2_BSC_segmentation_MOVIEWHOLECOMPARE_NUMCELL import fig_s2_o2_BSC_segmentation_MOVIEWHOLECOMPARE_NUMCELL
# for p in ps:
#     fig_s2_o2_BSC_segmentation_MOVIEWHOLECOMPARE_NUMCELL(project=p,segCh='DAPI',
#                                                         illumiCorrection=True,
#                                                         conditions=['ExperimentalCondition','Passage'],
#                                                         orgDataLoadPath='../Data/Original/OSKMtissueLiver/Images',)


# from s2_o3_BSC_segmentation_SCOUTVIEW import s2_o3_BSC_segmentation_SCOUTVIEW
# for p in ps:
#     s2_o3_BSC_segmentation_SCOUTVIEW(project=p,segCh='DAPI',illumiCorrection=True)

# from s2_o3_ptile import s2_o3_ptile
# for p in ps:
#     s2_o3_ptile(project=p,minTile=1, maxTile=99,sizeTh=0,segCh='DAPI',illumiCorrection=True)

# from s2_o4_3Dinterp_exfeatures import s2_o4_3Dinterp_exfeatures
# from s2_o4_3Dinterp_exfeatures_pyrads import s2_o4_3Dinterp_exfeatures_pyrads
# import time, random
# time.sleep(random.random())
# for p in ps:
#     for statPara in [
#                         'prob',
#                         'periphmean',
#                         'periphvar',
#                         'autoproxmean',
#                         'autoproxvar',
#                         'habitatmean',
#                         'habitatvar',
#                     ]:
#         s2_o4_3Dinterp_exfeatures(project=p,contents=chs,
#                                 statPara=statPara,illumiCorrection=True,
#                                 binS=5)
#     for statPara in [
#                         'prob',
#                         'periphmean',
#                         'periphvar',
#                         'autoproxmean',
#                         'autoproxvar',
#                         'habitatmean',
#                         'habitatvar',
#                     ]:
#         s2_o4_3Dinterp_exfeatures(project=p,contents=chs,
#                                 statPara=statPara,illumiCorrection=True,
#                                 binS=3)
    # for statPara in [
    #                     'TAS',
    #                 ]:
    #     s2_o4_3Dinterp_exfeatures(project=p,contents=chs,
    #                             statPara=statPara,illumiCorrection=True,
    #                             binS=3)
    # for statPara in [
    #                     '2DTAS',
    #                 ]:
    #     s2_o4_3Dinterp_exfeatures(project=p,contents=chs,
    #                             statPara=statPara,illumiCorrection=True,
    #                             binS=3)
    # for statPara in [
    #                     # 'Hist',
    #                     # 'GLCM',
    #                     # 'GLSZM',
    #                     # 'GLRLM',
    #                     # 'NGTDM',
    #                     # 'GLDM',
    #                 ]:
    #     s2_o4_3Dinterp_exfeatures_pyrads(project=p,contents=chs,
    #                             statPara=statPara,illumiCorrection=True,
    #                             binS=256)
    # s2_o4_3Dinterp_exfeatures_pyrads(project=p,contents=['DAPI'],
    #                         statPara='Shape',illumiCorrection=True,
    #                         binS=256)
        
    
    
# # from s2_o5_randboot_EpiAge_lsvm import s2_o5_randboot_EpiAge_lsvm
# # import random
# # #generate 10 random intenger values using
# # rndVals=[]
# # for i in range(100):
# #     rndVals.append(random.Random(i).randint(0,10000))
    
# # for p in ps:
# #     # for meanSize in [200, 100, 50, 10]:
# #     for meanSize in [200]:
# #         for sd in rndVals:
# #             s2_o5_randboot_EpiAge_lsvm(projects=[p],
# #                                 illumiCorrection=True,
# #                                 contents=chs,
# #                                 seed=sd,
# #                                 meanSize=meanSize,
# #                                 sampleGroups=['Passage'],
# #                                 )
            
# # for p in ps:
# #     for meanSize in [200, 100, 50, 10]:
# #         for sd in rndVals:
# #             s2_o5_randboot_EpiAge_lsvm(projects=[p],
# #                                 illumiCorrection=True,
# #                                 contents=chs,
# #                                 seed=sd,
# #                                 meanSize=meanSize,
# #                                 statParas=['2DTAS'],
# #                                 sampleGroups=['Passage'],
# #                                 )
    
# # for p in ps:
# #     for meanSize in [200, 100, 50, 10]:
# #         for sd in rndVals:
# #             s2_o5_randboot_EpiAge_lsvm(projects=[p],
# #                                 illumiCorrection=True,
# #                                 contents=chs,
# #                                 seed=sd,
# #                                 statParas=['prob',
# #                                             'periphmean',
# #                                             'periphvar',
# #                                             'autoproxmean',
# #                                             'autoproxvar',
# #                                             'habitatmean',
# #                                             'habitatvar',
# #                                             ],
# #                                 binS=3,
# #                                 meanSize=meanSize,
# #                                 )
        
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
        
        
        
# # from fig_s2_o2_BSC_segmentation_MOVIE_MULTICOLOR import fig_s2_o2_BSC_segmentation_MOVIE_MULTICOLOR
# # for p in ps:
# #     fig_s2_o2_BSC_segmentation_MOVIE_MULTICOLOR(project=p,segCh='DAPI',illumiCorrection=True)






# '''
# single channel analysis
# '''
# from s2_o4_3Dinterp_exfeatures import s2_o4_3Dinterp_exfeatures
# import time, random
# time.sleep(random.random())
# for p in ps:
#     for statPara in [
#                         'TAS',
#                     ]:
#         for ch in chs:
#             s2_o4_3Dinterp_exfeatures(project=p,contents=[ch],
#                                     statPara=statPara,illumiCorrection=True,
#                                     binS=3)


# from s2_o5_randboot_EpiAge_lsvm import s2_o5_randboot_EpiAge_lsvm
import random
#generate 10 random intenger values using
rndVals=[]
for i in range(100):
    rndVals.append(random.Random(i).randint(0,10000))
    
# for p in ps:
#     for meanSize in [200]:
#         for sd in rndVals:
#             for ch in chs:
#                 s2_o5_randboot_EpiAge_lsvm(projects=[p],
#                                     illumiCorrection=True,
#                                     contents=[ch],
#                                     seed=sd,
#                                     meanSize=meanSize,
#                                     sampleGroups=['Passage'],
#                                     )
                
# from fig_s2_o5_randboot_EpiAge_lsvm_VIOLIN_TTCOMP import fig_s2_o5_randboot_EpiAge_lsvm_VIOLIN_TTCOMP
# for p in ps:
#     # for meanSize in [200, 100, 50, 10]:
#     for meanSize in [200]:
#         for ch in chs:
#             fig_s2_o5_randboot_EpiAge_lsvm_VIOLIN_TTCOMP(projects=[p],
#                                                 contents=[ch],
#                                                 meanSize=meanSize,) 
