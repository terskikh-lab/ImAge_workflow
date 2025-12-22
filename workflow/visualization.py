#exeP_Reprogramming.py
import re

'''
configuration for common parameters
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
'''
visualization
'''
from fig_o4_ImAge_validation_VIOLIN import fig_o4_ImAge_validation_VIOLIN
for meanSize in [10]:
    fig_o4_ImAge_validation_VIOLIN(projects=[p],
                                contents=chs,
                                meanSize=meanSize,
                                groups=[
                                    'ExperimentalCondition',
                                ],
                                labels=['young', 'old'],
                                segCh='DAPI',
                                nBoot=1000,
                                illumiCorrection=True,
                                sampleSpecific=True,
                                colorsTrain={  # young
                                            'mouse1':'#20fc08',
                                            'mouse2':'#13ad02',
                                            'mouse3':'#45f731',
                                            'mouse4':'#299e1c',
                                            'mouse5':'#60f250',
                                            #old
                                            'mouse11':'#99360c',
                                            'mouse12':'#752a0b',
                                            'mouse13':'#9e4520',
                                            'mouse14':'#c74914',
                                            'mouse15':'#9e4c29',
                                            },
                                colorsTrainAll={  # young
                                                'young':'#20fc08',
                                                'old':'#99360c',
                                                },
                                colorsTest={  # young
                                            'mouse1':'#20fc08',
                                            'mouse2':'#13ad02',
                                            'mouse3':'#45f731',
                                            'mouse4':'#299e1c',
                                            'mouse5':'#60f250',
                                            #old treated
                                            'mouse6':'#d9a600',
                                            'mouse7':'#a68003',
                                            'mouse8':'#ffca1c',
                                            'mouse9':'#ad8a17',
                                            'mouse10':'#f0c435',
                                            #old untreated
                                            'mouse11':'#99360c',
                                            'mouse12':'#752a0b',
                                            'mouse13':'#9e4520',
                                            'mouse14':'#c74914',
                                            'mouse15':'#9e4c29',
                                            },
                                colorsTestAll = {
                                            'young':'#20fc08',
                                            'middle':'#d9a600',
                                            'old':'#99360c',
                                            }
                                )



'''
ImAge readouts export
'''
from export_o4_ImAge_validation import export_o4_ImAge_validation
export_o4_ImAge_validation(projects=[p],
                            contents=chs,
                            meanSize=meanSize,
                            groups=['ExperimentalCondition'],
                            labels=['young', 'old'],
                            segCh='DAPI',
                            nBoot=1000,
                            illumiCorrection=True
                            )