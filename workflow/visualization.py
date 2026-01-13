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
imageIndex={'ch1':'Channel1',
            'ch2':'Channel2',
            'ch3':'Channel3'}
orgDataLoadPath='Data/Original'
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
                                resultsSavePath=resultsSavePath,
                                contents=chs,
                                meanSize=meanSize,
                                groups=[
                                    'condition',
                                ],
                                labels=['young', 'old'],
                                segCh='DAPI',
                                nBoot=1000,
                                illumiCorrection=illumiCorrection,
                                sampleSpecific=True,
                                colorsTrain={  # young
                                            'Mouse1':'#20fc08',
                                            'Mouse2':'#13ad02',
                                            'Mouse3':'#45f731',
                                            'Mouse4':'#299e1c',
                                            'Mouse5':'#60f250',
                                            'Mouse559':'#82f774',
                                            #old
                                            'Mouse15':'#99360c',
                                            'Mouse17':'#752a0b',
                                            'Mouse18':'#9e4520',
                                            'Mouse19':'#c74914',
                                            },
                                colorsTrainAll={  # young
                                                'young':'#20fc08',
                                                'old':'#99360c',
                                                },
                                colorsTest={  # young
                                            'Mouse1':'#20fc08',
                                            'Mouse2':'#13ad02',
                                            'Mouse3':'#45f731',
                                            'Mouse4':'#299e1c',
                                            'Mouse5':'#60f250',
                                            'Mouse559':'#82f774',
                                            #middle age
                                            'Mouse1638':'#d9a600',
                                            'Mouse1639':'#a68003',
                                            'Mouse1640':'#ffca1c',
                                            'Mouse1647':'#ad8a17',
                                            'Mouse1666':'#f0c435',
                                            #old
                                            'Mouse15':'#99360c',
                                            'Mouse17':'#752a0b',
                                            'Mouse18':'#9e4520',
                                            'Mouse19':'#c74914',
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
                            resultsSavePath=resultsSavePath,
                            contents=chs,
                            meanSize=meanSize,
                            groups=['condition'],
                            labels=['young', 'old'],
                            segCh='DAPI',
                            nBoot=1000,
                            illumiCorrection=illumiCorrection
                            )