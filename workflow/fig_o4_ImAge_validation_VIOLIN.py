"""
This script visualizes ImAge validation results using violin plots.

Description:
    This script generates violin plots to visualize the results of the ImAge validation
    workflow. It processes the output files from the validation step, calculates
    performance metrics like accuracy, and creates various plots to compare
    predictions across different samples and groups for both training and test sets.

    The main function, `fig_o4_ImAge_validation_VIOLIN`, handles:
    - Loading validation data.
    - Calculating accuracy and other metrics.
    - Generating and saving violin plots and box plots for prediction distributions.
    - Performing statistical tests (Mann-Whitney U test, Levene's test) to compare groups.

    The script is configurable through the parameters of the main function, allowing
    for analysis of different projects, feature sets, and experimental conditions.

Usage:
    This script is intended to be run as part of the ImAge analysis workflow.
    The `fig_o4_ImAge_validation_VIOLIN` function can be called from other scripts
    (e.g., a main workflow script) with appropriate parameters.

Author:
    Kenta Ninomiya
    Harry Perkins Institute of Medical Research/ the Univeristy of Western Australia
    Date: 2025/07/15
"""

#import modules=======================
import os
import random
import time
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
import scipy.stats as stats
from typing import List, Dict

#import self defined functions========
from subfunctions.paramerge import paramerge
from subfunctions.loadPathGenerator import loadPathGenerator
from subfunctions.dir_rmv_file import dir_rmv_file
from subfunctions.imreqant import imreqant
from subfunctions.ezload import ezload
#=====================================


def fig_o4_ImAge_validation_VIOLIN(projects: List[str] = ['longiBLOOD'],
                            resultsSavePath: str = '../Data/Results',
                            statParas: List[str] = ['TAS'],
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
                            illumiCorrection: bool = True,
                            sampleSpecific: bool = True,
                            colorsTrain: Dict[str, str] = {#young  
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
                            colorsTrainAll: Dict[str, str] = {#young  
                                            'young':'#20fc08',
                                            'old':'#99360c',
                                            },
                            colorsTest: Dict[str, str] = {#young  
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
                            colorsTestAll: Dict[str, str] = {
                                        'young':'#20fc08',
                                        'middle':'#d9a600',
                                        'old':'#99360c',
                                        }
                            ) -> None:
    """
    Visualizes the result of ImAge validation with violin plots.

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
        sampleSpecific (bool, optional): Whether to generate sample-specific plots. Defaults to True.
        colorsTrain (Dict[str, str], optional): Colors for training samples. Defaults to a predefined dict.
        colorsTrainAll (Dict[str, str], optional): Colors for all training groups. Defaults to a predefined dict.
        colorsTest (Dict[str, str], optional): Colors for test samples. Defaults to a predefined dict.
        colorsTestAll (Dict[str, str], optional): Colors for all test groups. Defaults to a predefined dict.
    """
    #Initialization=======================
    parameter=paramerge(paramPath=f'{resultsSavePath}/platemap', projects=[project.split('iC_')[-1] for project in projects])
    loadPath=f'{resultsSavePath}/{"_".join(projects)}/o4_ImAge_validation'
    savePath=f'{resultsSavePath}/{"_".join(projects)}/fig_o4_ImAge_validation_VIOLIN'
    
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

    accList=[]
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
        
        #ROC analysis
        fpr, tpr, thresholds = roc_curve(trainLabelList==labels[0], trainProj)
        roc_auc = auc(fpr, tpr)
        
        #get the best threshold
        if roc_auc==1:
            bestTh=np.mean([trainProj[trainLabelList==labels[0]].min(),trainProj[trainLabelList!=labels[0]].max()])
        else:
            bestTh=thresholds[np.argmax(tpr-fpr)]
        
        #calculate the accuracy, sensitivity, specificity with the best threshold
        binPredTrain=trainProj>bestTh
        TPTrain=np.logical_and(trainLabelList==labels[0],binPredTrain)
        TNTrain=np.logical_and(trainLabelList!=labels[0],~binPredTrain)
        accTrain=(TPTrain.sum()+TNTrain.sum())/len(TPTrain)
        
        #get the labels for the test data
        labelsIn=[]
        for label in labels:
            labelsIn.append(testLabelList==label)
        labelsIn=np.stack(labelsIn).any(axis=0)
        
        binPredTest=testProj>bestTh
        TPTest=np.logical_and(testLabelList==labels[0],binPredTest)[labelsIn]
        TNTest=np.logical_and(testLabelList!=labels[0],~binPredTest)[labelsIn]
        accTest=(TPTest.sum()+TNTest.sum())/len(TPTest)
        
        accList.append(pd.DataFrame({'accTrain':[accTrain],'accTest':[accTest]}))
        #obtain average prediction and labels for each sample
        tmpTrainUniqueSample=np.unique(trainSampleList)
        tmpTestUniqueSample=np.unique(testSampleList)
        allPredList.append(np.concatenate([[np.mean(trainProj[trainSampleList==ts]) for ts in tmpTrainUniqueSample],
                                             [np.mean(testProj[testSampleList==ts]) for ts in tmpTestUniqueSample]]))
        allLabelList.append(np.concatenate([np.array([trainLabelList[trainSampleList==ts][0] for ts in tmpTrainUniqueSample]),
                                           np.array([testLabelList[testSampleList==ts][0] for ts in tmpTestUniqueSample])]))
        allTrainBinList.append(np.concatenate([np.ones(len(tmpTrainUniqueSample),dtype=bool),np.zeros(len(tmpTestUniqueSample),dtype=bool)]))
        allParamsList.append(np.concatenate([tmpTrainUniqueSample,tmpTestUniqueSample]))
    
    #visiuallize the accuracy with bar plot with error bar and annotation of the average accuracy in the middle of the bar
    accList=pd.concat(accList)
    plotData=pd.DataFrame({'acc':accList.mean(),'std':accList.std()})
    fig = go.Figure()
    fig.add_trace(go.Bar(x=plotData.index,  
                         y=plotData['acc'], 
                         error_y=dict(type='data', 
                                      array=plotData['std'],
                                      color='black'),
                         marker_color='#7d7d7d',
                         marker_line_color='black',
                         marker_line_width=2,))
    #replace the x axis with the parameter names, and set maximum and minimum to 1 and 0, respectively
    fig.update_layout(xaxis=dict(tickmode='array', 
                                tickvals=np.arange(len(plotData.index)), 
                                # ticktext=plotNames,
                                showline=True,
                                mirror=True,
                                linecolor='black',
                                linewidth=3,),
                      yaxis=dict(ticks='inside',
                                 tickwidth=3,
                                 tickcolor='black',
                                 showline=True,
                                 mirror=True,
                                 linecolor='black',
                                linewidth=3),)

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_yaxes(range=[0,1])
    fig.update_layout(showlegend=False)
    #add the average accuracy as annotation
    fig.update_layout(title='', xaxis_title='Parameters', yaxis_title='Accuracy',
                        font=dict(family="Arial",size=20,color='black'),
                        plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    # fig.show()
    saveFileName=f"{savePath}/{saveNameAdd}_{statPara}_{'_'.join(contents)}_meanS{meanSize}_nBoot{nBoot}_{'_'.join(groups)}_{'_'.join(labels)}_acc.html"
    fig.write_html(saveFileName)
    
    #visiuallize the prediciton probability with violin plot    
    allPredList=np.concatenate(allPredList)
    allLabelList=np.concatenate(allLabelList)
    allTrainBinList=np.concatenate(allTrainBinList)
    allParamsList=np.concatenate(allParamsList)
    plotData=pd.DataFrame({'pred':allPredList,'label':allLabelList,'group':'Test','sampleID':allParamsList})
    plotData['group'].iloc[allTrainBinList]='Training'
    
    plotData['pred']=imreqant(plotData['pred'],
                              plotData['pred'].min(),
                              plotData['pred'].max(),
                              0, 1, outLow=False, outHigh=False, getInt=False)
    
    if sampleSpecific==True:
        #make violin plot for each sample for training and test
        shapes=['circle','square','diamond','cross','x','triangle-up','triangle-down','triangle-left','triangle-right','pentagon','hexagon','octagon','star']
        
#=======#training================================
        tmpPlotData=plotData[allTrainBinList]
        fig = go.Figure()
        leyLists=list(colorsTrain.keys())
        
        count=0
        meanList=[]
        labelList=[]
        foundSamples=[]
        for sample in leyLists:
            if (tmpPlotData['sampleID']==sample).any():
                fig.add_trace(go.Violin(y=tmpPlotData['pred'].loc[tmpPlotData['sampleID']==sample], 
                                        name=sample,
                                        line_color=colorsTrain[sample]))
                count+=1
                meanList.append(tmpPlotData['pred'].loc[tmpPlotData['sampleID']==sample].mean())
                labelList.append(tmpPlotData['label'].loc[tmpPlotData['sampleID']==sample].values[0])
                foundSamples.append(sample)
        sampleDf=pd.DataFrame({'sampleID':foundSamples,'mean':meanList,'label':labelList})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        layout = dict(xaxis = dict(title = '', showgrid=False, linewidth=3.5, linecolor='black', ticks='inside', mirror=True),
                yaxis = dict(title = '', showgrid=True, ticks='inside',showline=True, linewidth=3.5, linecolor='black', mirror=True,),
                font=dict(size=30, color='black', family='Arial'),
                )
        fig.update_layout(layout)
        # fig.show()
        #save the plot
        saveFileName=f"{savePath}/{saveNameAdd}_{statPara}_{'_'.join(contents)}_meanS{meanSize}_nBoot{nBoot}_{'_'.join(groups)}_{'_'.join(labels)}_pred_persample_training.html"
        fig.write_html(saveFileName)
        
        #train all
        fig = go.Figure()
        leyLists=list(colorsTrain.keys())
        
        count=0
        for label in leyLists:
            if (sampleDf['sampleID']==label).any():
                tmpMean=sampleDf['mean'].loc[sampleDf['sampleID']==label].to_list()
                fig.add_trace(go.Box(y=tmpMean, 
                                x=[sampleDf['label'].loc[sampleDf['sampleID']==label].values[0]],
                                boxpoints='all',
                                pointpos=np.random.uniform(-0.1,0.1),
                                name=label,
                                marker = dict(color = colorsTrain[label],
                                            size=10,
                                            line = dict(color = 'rgba(0,0,0,1)',width=2),
                                            ),
                                line = dict(color = 'rgba(0,0,0,0)'),
                                fillcolor = 'rgba(0,0,0,0)'
                                ))
                count+=1
        leyLists=list(colorsTrainAll.keys())
        
        for label in leyLists:
            tmpMean=sampleDf['mean'].loc[sampleDf['label']==label].to_list()
            fig.add_trace(go.Scatter(y=[np.mean(tmpMean)], 
                            x=[label],
                            name=label,
                            error_y=dict(
                                type='data', # value of error bar given in data coordinates
                                array=[sampleDf['mean'].loc[sampleDf['label']==label].std()],
                                visible=True,),
                            line=dict(color='black', width=2),
                            ))
            
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        layout = dict(xaxis = dict(title = '', showgrid=False, linewidth=3.5, linecolor='black', ticks='inside', mirror=True),
                yaxis = dict(title = '', showgrid=True, ticks='inside',showline=True, linewidth=3.5, linecolor='black', mirror=True,),
                font=dict(size=30, color='black', family='Arial'),
                )
        fig.update_layout(layout)
        # fig.show()
        saveFileName=f"{savePath}/{saveNameAdd}_{statPara}_{'_'.join(contents)}_meanS{meanSize}_nBoot{nBoot}_{'_'.join(groups)}_{'_'.join(labels)}_pred_persample_mean_training_scatter.html"
        fig.write_html(saveFileName)
        
#=======#test==============================
        tmpPlotData=plotData[~allTrainBinList]
        fig = go.Figure()
        leyLists=list(colorsTest.keys())
        
        count=0
        meanList=[]
        labelList=[]
        foundSamples=[]
        for sample in leyLists:
            if (tmpPlotData['sampleID']==sample).any():
                #violin with box plot and small sized points for outliers
                fig.add_trace(go.Violin(y=tmpPlotData['pred'].loc[tmpPlotData['sampleID']==sample], 
                                        name=sample,
                                        line_color='black',
                                        fillcolor=colorsTest[sample],
                                        opacity=0.7,                                    
                                        box_visible=True, 
                                        meanline_visible=True,
                                        marker_size=2,
                                        ))
                count+=1
                meanList=np.append(meanList,tmpPlotData['pred'].loc[tmpPlotData['sampleID']==sample].mean())
                labelList=np.append(labelList,tmpPlotData['label'].loc[tmpPlotData['sampleID']==sample].values[0])
                foundSamples.append(sample)
        sampleDf=pd.DataFrame({'sampleID':foundSamples,'mean':meanList,'label':labelList})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        layout = dict(xaxis = dict(title = '', 
                                   showline=True,
                                   showgrid=False, 
                                   linewidth=3.5, 
                                   linecolor='black', 
                                   ticks='outside', 
                                   tickwidth=3,
                                   mirror=False),
                      yaxis = dict(title = '', 
                                   showline=True,
                                   showgrid=True, 
                                   linewidth=3,
                                   linecolor='black', 
                                   ticks='outside',
                                   tickwidth=3,
                                   mirror=False,),
                font=dict(size=30, color='black', family='Arial'),
                )
        
        fig.update_layout(layout)
        # fig.show()
        #save the plot
        saveFileName=f"{savePath}/{saveNameAdd}_{statPara}_{'_'.join(contents)}_meanS{meanSize}_nBoot{nBoot}_{'_'.join(groups)}_{'_'.join(labels)}_pred_persample_test.html"
        fig.write_html(saveFileName)  
        
        #obtain the p value for each sample
        leyLists=sampleDf['sampleID'].unique()
        pValMat=np.zeros((len(leyLists),len(leyLists)))
        pValMat=pd.DataFrame(pValMat,index=leyLists,columns=leyLists)
        valVec=np.zeros((len(leyLists)))
        valVec=pd.DataFrame(valVec,index=leyLists,columns=['val'])
        for ref in leyLists:
            valVec['val'].loc[ref]=np.median(tmpPlotData['pred'].loc[tmpPlotData['sampleID']==ref])
            #check the normality of the distribution using Shapiro-Wilk test
            # refNorm=stats.shapiro(tmpPlotData['pred'].loc[tmpPlotData['sampleID']==ref])[1]>0.05
            for com in leyLists:
                pValMat.loc[ref,com]=stats.mannwhitneyu(tmpPlotData['pred'].loc[tmpPlotData['sampleID']==ref],tmpPlotData['pred'].loc[tmpPlotData['sampleID']==com])[1]
                # comNorm=stats.shapiro(tmpPlotData['pred'].loc[tmpPlotData['sampleID']==com])[1]>0.05
                # if refNorm==True and comNorm==True:
                #     # pValMat.loc[ref,com]=stats.ttest_ind(tmpPlotData['pred'].loc[tmpPlotData['sampleID']==ref],tmpPlotData['pred'].loc[tmpPlotData['sampleID']==com])[1]
                # else:
                #     pValMat.loc[ref,com]=stats.mannwhitneyu(tmpPlotData['pred'].loc[tmpPlotData['sampleID']==ref],tmpPlotData['pred'].loc[tmpPlotData['sampleID']==com])[1]

        #save the p value matrix
        pValMat.to_csv(f"{savePath}/{saveNameAdd}_{statPara}_{'_'.join(contents)}_meanS{meanSize}_nBoot{nBoot}_{'_'.join(groups)}_{'_'.join(labels)}_pred_persample_pValMat.csv")
        valVec.to_csv(f"{savePath}/{saveNameAdd}_{statPara}_{'_'.join(contents)}_meanS{meanSize}_nBoot{nBoot}_{'_'.join(groups)}_{'_'.join(labels)}_pred_persample_valVec.csv")
             
        #test all
        leyLists=list(colorsTestAll.keys())
        #test all with distribution
        fig = go.Figure()
        count=0
        for label in leyLists:
            if (tmpPlotData['label']==label).any():
                #violin with box plot and small sized points for outliers
                fig.add_trace(go.Violin(y=tmpPlotData['pred'].loc[tmpPlotData['label']==label], 
                                        name=label,
                                        line_color='rgba(0,0,0,0)',
                                        fillcolor=colorsTestAll[label],
                                        opacity=0.7,
                                        box_visible=False, 
                                        meanline_visible=False,
                                        side='negative',
                                        marker_size=2
                                        ))
                #add box plot using same violiplot parameters
                fig.add_trace(go.Violin(y=tmpPlotData['pred'].loc[tmpPlotData['label']==label], 
                                        name=label,
                                        line_color='rgba(0,0,0,0)',
                                        fillcolor='rgba(0,0,0,0)',
                                        opacity=1,
                                        box_visible=True, 
                                        box_fillcolor=colorsTestAll[label],
                                        box_line_color='black',
                                        meanline_visible=True,
                                        marker_size=2
                                        ))
                count+=1
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        layout = dict(xaxis = dict(title = '', 
                                   showline=True,
                                   showgrid=False, 
                                   linewidth=3.5, 
                                   linecolor='black', 
                                   ticks='outside', 
                                   tickwidth=3,
                                   mirror=False),
                      yaxis = dict(title = '', 
                                   showline=True,
                                   showgrid=True, 
                                   linewidth=3,
                                   linecolor='black', 
                                   ticks='outside',
                                   tickwidth=3,
                                   mirror=False,),
                font=dict(size=30, color='black', family='Arial'),
                )
        fig.update_layout(layout)
        
        leyLists=list(colorsTest.keys())
        count=0
        for label in leyLists:
            if (sampleDf['sampleID']==label).any():
                tmpMean=sampleDf['mean'].loc[sampleDf['sampleID']==label].to_list()
                fig.add_trace(go.Box(y=tmpMean, 
                                x=[sampleDf['label'].loc[sampleDf['sampleID']==label].values[0]],
                            boxpoints='all',
                                pointpos=0.5,
                                name=label,
                                marker = dict(color = colorsTest[label],
                                            line = dict(color = 'rgba(0,0,0,1)',width=2),
                                            size=10,),
                                line = dict(color = 'rgba(0,0,0,0)'),
                                fillcolor = 'rgba(0,0,0,0)' 
                                ))
                count+=1
        # fig.show()
        saveFileName=f"{savePath}/{saveNameAdd}_{statPara}_{'_'.join(contents)}_meanS{meanSize}_nBoot{nBoot}_{'_'.join(groups)}_{'_'.join(labels)}_pred_persample_mean_test_scatter_dist.html"
        fig.write_html(saveFileName)
        
        
        leyLists=list(colorsTestAll.keys())
        leyLists=[label for label in leyLists if (tmpPlotData['label']==label).any()]
        #obtain the p value for each sample
        pValMat=np.zeros((len(leyLists),len(leyLists)))
        pValMat=pd.DataFrame(pValMat,index=leyLists,columns=leyLists)
        valVec=np.zeros((len(leyLists)))
        valVec=pd.DataFrame(valVec,index=leyLists,columns=['val'])
        for ref in leyLists:
            valVec['val'].loc[ref]=np.median(tmpPlotData['pred'].loc[tmpPlotData['label']==ref])
            #check the normality of the distribution using Shapiro-Wilk test
            # refNorm=stats.shapiro(tmpPlotData['pred'].loc[tmpPlotData['label']==ref])[1]>0.05
            for com in leyLists:
                pValMat.loc[ref,com]=stats.mannwhitneyu(tmpPlotData['pred'].loc[tmpPlotData['label']==ref],tmpPlotData['pred'].loc[tmpPlotData['label']==com])[1]
                # comNorm=stats.shapiro(tmpPlotData['pred'].loc[tmpPlotData['label']==com])[1]>0.05
                # if refNorm==True and comNorm==True:
                #     pValMat.loc[ref,com]=stats.ttest_ind(tmpPlotData['pred'].loc[tmpPlotData['label']==ref],tmpPlotData['pred'].loc[tmpPlotData['label']==com])[1]
                # else:
                #     pValMat.loc[ref,com]=stats.mannwhitneyu(tmpPlotData['pred'].loc[tmpPlotData['label']==ref],tmpPlotData['pred'].loc[tmpPlotData['label']==com])[1]
        #save the p value matrix
        pValMat.to_csv(f"{savePath}/{saveNameAdd}_{statPara}_{'_'.join(contents)}_meanS{meanSize}_nBoot{nBoot}_{'_'.join(groups)}_{'_'.join(labels)}_means_pValMat_mean.csv")
        valVec.to_csv(f"{savePath}/{saveNameAdd}_{statPara}_{'_'.join(contents)}_meanS{meanSize}_nBoot{nBoot}_{'_'.join(groups)}_{'_'.join(labels)}_means_valVec_mean.csv")
        
        #obtain the p value for each sample about the variance
        pValMat=np.zeros((len(leyLists),len(leyLists)))
        pValMat=pd.DataFrame(pValMat,index=leyLists,columns=leyLists)
        valVec=np.zeros((len(leyLists)))
        valVec=pd.DataFrame(valVec,index=leyLists,columns=['val'])
        for ref in leyLists:
            valVec['val'].loc[ref]=tmpPlotData['pred'].loc[tmpPlotData['label']==ref].var()
            #check the normality of the distribution using Shapiro-Wilk test
            # refNorm=stats.shapiro(tmpPlotData['pred'].loc[tmpPlotData['label']==ref])[1]>0.05
            for com in leyLists:
                pValMat.loc[ref,com]=stats.levene(tmpPlotData['pred'].loc[tmpPlotData['label']==ref],tmpPlotData['pred'].loc[tmpPlotData['label']==com])[1]
                # comNorm=stats.shapiro(tmpPlotData['pred'].loc[tmpPlotData['label']==com])[1]>0.05
                # if refNorm==True and comNorm==True:
                #     pValMat.loc[ref,com]=f_test(tmpPlotData['pred'].loc[tmpPlotData['label']==ref],tmpPlotData['pred'].loc[tmpPlotData['label']==com])
                # else:
                #     pValMat.loc[ref,com]=stats.levene(tmpPlotData['pred'].loc[tmpPlotData['label']==ref],tmpPlotData['pred'].loc[tmpPlotData['label']==com])[1]
        #save the p value matrix
        pValMat.to_csv(f"{savePath}/{saveNameAdd}_{statPara}_{'_'.join(contents)}_meanS{meanSize}_nBoot{nBoot}_{'_'.join(groups)}_{'_'.join(labels)}_means_pValMat_var.csv")
        valVec.to_csv(f"{savePath}/{saveNameAdd}_{statPara}_{'_'.join(contents)}_meanS{meanSize}_nBoot{nBoot}_{'_'.join(groups)}_{'_'.join(labels)}_means_valVec_var.csv")        
        
#f-test function
def f_test(x: np.ndarray, y: np.ndarray) -> float:
    """
    Performs an F-test to compare the variances of two arrays.

    Args:
        x (np.ndarray): First array of data.
        y (np.ndarray): Second array of data.

    Returns:
        float: The p-value of the F-test.
    """
    x=np.array(x)
    y=np.array(y)
    xVar=x.var()
    yVar=y.var()
    if xVar>yVar:
        f=xVar/yVar
    else:
        f=yVar/xVar
    df1=len(x)-1
    df2=len(y)-1
    pVal=stats.f.cdf(f,df1,df2)
    return pVal