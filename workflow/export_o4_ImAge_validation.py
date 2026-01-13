"""
This script exports the ImAge validation results to CSV files.

Description:
    This script processes the output of the ImAge validation workflow, aggregates
    the prediction results, and exports them into separate CSV files for the
    training and test sets. This allows for further analysis and visualization
    of the model's performance.

    The main function, `export_o4_ImAge_validation`, handles:
    - Loading the validation result files (.pickle).
    - Aggregating predictions and labels for each sample.
    - Creating a pandas DataFrame with the combined data.
    - Separating the data into training and test sets.
    - Saving the training and test data to CSV files.

Usage:
    This script is intended to be run as part of the ImAge analysis workflow.
    The `export_o4_ImAge_validation` function can be called from other scripts
    (e.g., a main workflow script) with appropriate parameters to export
    the validation results.

Author:
    Kenta Ninomiya
    Harry Perkins Institute of Medical Research/ the Univeristy of Western Australia
    Date: 2025/07/15
"""

#import modules=======================
import os
import pandas as pd
import numpy as np
from typing import List

#import self defined functions========
from subfunctions.paramerge import paramerge
from subfunctions.dir_rmv_file import dir_rmv_file
from subfunctions.ezload import ezload
#=====================================

def export_o4_ImAge_validation(projects: List[str] = ['longiBLOOD'],
                            resultsSavePath: str = '../Data/Results',
                            contents: List[str] = [
                                'DAPI',
                                'H3K4me1',
                                'H3K27ac',
                                      ],
                            meanSize: int = 200,
                            groups: List[str] = [
                                'ExperimentalCondition',    
                                    ],
                            labels: List[str] = ['young','old'],
                            segCh: str = 'DAPI',
                            nBoot: int = 1000,
                            illumiCorrection: bool = True
                            ) -> None:
    """
    Exports the ImAge validation results to CSV files.

    Args:
        projects (List[str], optional): List of project names. Defaults to ['longiBLOOD'].
        resultsSavePath (str, optional): Path to the results directory. Defaults to '../Data/Results'.
        binS (int, optional): Bin size. Defaults to 3.
        statParas (List[str], optional): List of statistical parameters. Defaults to ['TAS'].
        contents (List[str], optional): List of content names. Defaults to ['DAPI', 'H3K4me1', 'H3K27ac'].
        meanSize (int, optional): Mean size. Defaults to 200.
        groups (List[str], optional): List of group names. Defaults to ['ExperimentalCondition'].
        labels (List[str], optional): List of labels. Defaults to ['young', 'old'].
        segCh (str, optional): Segmentation channel. Defaults to 'DAPI'.
        nBoot (int, optional): Number of bootstrap iterations. Defaults to 1000.
        illumiCorrection (bool, optional): Whether illumination correction was used. Defaults to True.
    """
    #Initialization=======================
    loadPath=f'{resultsSavePath}/{"_".join(projects)}/o4_ImAge_validation'
    savePath=f'{resultsSavePath}/{"_".join(projects)}/export_o4_ImAge_validation'
    
    os.makedirs(savePath, exist_ok=True)
    saveNameAdd=''
    if segCh!='DAPI':
        saveNameAdd=f'seg_{segCh}'
    if illumiCorrection==False:
        saveNameAdd=f'noIC_{saveNameAdd}'
        
    statPara= 'TAS'
        
    #reorder them by alphabetical order
    contents.sort()
    groups.sort()
    labels.sort()
    #======================================
    loadFileNamesW=f'{saveNameAdd}_{statPara}_{"_".join(contents)}_meanS{meanSize}_nBoot{nBoot}_SEEDNUM_{"_".join(groups)}_{"_".join(labels)}.pickle'
    loadFileNames=dir_rmv_file(loadPath,loadFileNamesW.replace('SEEDNUM','*'))

    allPredList=[]
    allLabelList=[]
    allTrainBinList=[]
    allParamsList=[]
    for i in loadFileNames:
        res=ezload(f'{loadPath}/{i}')
        trainProj=res['trainProj']
        trainLabelList=res['trainLabelList']
        trainSampleList=res['trainSampleList']
        testProj=res['testProj']
        testLabelList=res['testLabelList']
        testSampleList=res['testSampleList']
        
        #obtain average prediction and labels for each sample
        tmpTrainUniqueSample=np.unique(trainSampleList)
        tmpTestUniqueSample=np.unique(testSampleList)
        allPredList.append(np.concatenate([[np.mean(trainProj[trainSampleList==ts]) for ts in tmpTrainUniqueSample],
                                             [np.mean(testProj[testSampleList==ts]) for ts in tmpTestUniqueSample]]))
        allLabelList.append(np.concatenate([np.array([trainLabelList[trainSampleList==ts][0] for ts in tmpTrainUniqueSample]),
                                           np.array([testLabelList[testSampleList==ts][0] for ts in tmpTestUniqueSample])]))
        allTrainBinList.append(np.concatenate([np.ones(len(tmpTrainUniqueSample),dtype=bool),np.zeros(len(tmpTestUniqueSample),dtype=bool)]))
        allParamsList.append(np.concatenate([tmpTrainUniqueSample,tmpTestUniqueSample]))
    
    allPredList=np.concatenate(allPredList)
    allLabelList=np.concatenate(allLabelList)
    allTrainBinList=np.concatenate(allTrainBinList)
    allParamsList=np.concatenate(allParamsList)
    
    plotData=pd.DataFrame({'pred':allPredList,'label':allLabelList,'group':'Test','sample':allParamsList})
    plotData['group'].iloc[allTrainBinList]='Training'

    # Separate training and test data
    train_data = plotData[plotData['group'] == 'Training']
    test_data = plotData[plotData['group'] == 'Test']
    # Save training data to CSV
    train_saveName = f"{savePath}/{saveNameAdd}_{statPara}_{'_'.join(contents)}_meanS{meanSize}_nBoot{nBoot}_{'_'.join(groups)}_{'_'.join(labels)}_pred_train.csv"
    train_data.to_csv(train_saveName, index=False)
    print(f"Training data saved to {train_saveName}")

    # Save test data to CSV
    test_saveName = f"{savePath}/{saveNameAdd}_{statPara}_{'_'.join(contents)}_meanS{meanSize}_nBoot{nBoot}_{'_'.join(groups)}_{'_'.join(labels)}_pred_test.csv"
    test_data.to_csv(test_saveName, index=False)
    print(f"Test data saved to {test_saveName}")

