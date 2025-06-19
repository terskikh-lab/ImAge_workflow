'''
Description fig_s2_o5_randboot_EpiAge_lsvm_VIOLIN_TTCOMP.py
========================================
visualize the result of s2_o5_randommean_LR_labels.py with violin plot
========================================
Kenta Ninomiya @ Sanford Burnham Prebys Medical Discovery Institute: 2023/03/27
'''

#import modules=======================
import os
import random
import time
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
import scipy.stats as stats

#import self defined functions========
from Functions.paramerge import *
from Functions.loadPathGenerator import *
from Functions.dir_rmv_folder import *
from Functions.dir_rmv_file import *
from Functions.glaylconvert import *
from Functions.dbgimshow import *
from Functions.progressregister import *
from Functions.idxremover import *
from Functions.imreqant import *
from Functions.ezsave import *
from Functions.ezload import *
#=====================================


def fig_s2_o5_randboot_EpiAge_lsvm_VIOLIN_TTCOMP(projects=['longiBLOOD'],
                            binS=3,
                            statParas=['TAS'],
                            contents=[
                                'DAPI',
                                'H3K4me1',
                                'H3K27ac',
                                      ],
                            meanSize=200,
                            groups=[
                                'ExperimentalCondition',    
                                    ],
                            labels=['young_i4F_untreated','old_i4F_untreated'],
                            segCh='DAPI',
                            nBoot=1000,
                            illumiCorrection=True,
                            sampleSpecific=True,
                            colorsTrain={#young  
                                        '457':'#20fc08',
                                        '1155':'#13ad02',
                                        '1158':'#45f731',
                                        '1163':'#299e1c',
                                        '1164':'#60f250',
                                        #old
                                        '1000':'#99360c',
                                        '1001':'#752a0b',
                                        '1002':'#9e4520',
                                        '1003':'#c74914',
                                        '1051':'#9e4c29',
                                        },
                            colorsTrainAll={#young  
                                            'young_i4F_untreated':'#20fc08',
                                            'old_i4F_untreated':'#99360c',
                                            },
                            colorsTest={#young  
                                        '457':'#20fc08',
                                        '1155':'#13ad02',
                                        '1158':'#45f731',
                                        '1163':'#299e1c',
                                        '1164':'#60f250',
                                        #old treated
                                        '384':'#d9a600',
                                        '1010':'#a68003',
                                        '1011':'#ffca1c',
                                        '1025':'#ad8a17',
                                        '1027':'#f0c435',
                                        #old untreated
                                        '1000':'#99360c',
                                        '1001':'#752a0b',
                                        '1002':'#9e4520',
                                        '1003':'#c74914',
                                        '1051':'#9e4c29',
                                        },
                            colorsTestAll={
                                        'young_i4F_untreated':'#20fc08',
                                        'old_i4F_treated':'#d9a600',
                                        'old_i4F_untreated':'#99360c',
                                        }
                            ):

    #Initialization=======================
    parameter=paramerge(paramPath='../Data/Results/Parameters', projects=[project.split('iC_')[-1] for project in projects])
    loadPaths0=loadPathGenerator(loadPath='../Data/Results', 
                                projects=projects, 
                                loadFolder='s2_o4_3Dinterp_exfeatures')
    loadPath='../Data/Results/'+'_'.join(projects)+'/s2_o5_randboot_EpiAge_lsvm'
    savePath='../Data/Results/'+'_'.join(projects)+'/fig_s2_o5_randboot_EpiAge_lsvm_VIOLIN_TTCOMP'
    os.makedirs(savePath, exist_ok=True)
    trainRatio=0.8
        
    saveNameAdd=''
    if segCh!='DAPI':
        saveNameAdd='seg_'+segCh
    if illumiCorrection==False:
        saveNameAdd='noIC_'+saveNameAdd
        
    #reorder them by alphabetical order
    contents.sort()
    groups.sort()
    labels.sort()
    #======================================
    loadFileNamesW=saveNameAdd+'_'.join(sorted(statParas))+'_'+'_'.join(contents)+'_'+'binS'+str(binS)+'_meanS'+str(meanSize)+'_nBoot'+str(nBoot)+'_seed*_'+'_'.join(groups)+'_'+'_'.join(labels)+'.pickle'
    loadFileNames=dir_rmv_file(loadPath,loadFileNamesW)

    accList=[]
    allPredList=[]
    allLabelList=[]
    allTrainBinList=[]
    allParamsList=[]
    for i in loadFileNames:
        res=ezload(loadPath+'/'+i)
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
    saveFileName=savePath+'/'+saveNameAdd+'_'.join(sorted(statParas))+'_'+'_'.join(contents)+'_'+'binS'+str(binS)+'_meanS'+str(meanSize)+'_'+'_'.join(groups)+'_acc.html'
    fig.write_html(saveFileName)
    
    #visiuallize the prediciton probability with violin plot    
    allPredList=np.concatenate(allPredList)
    allLabelList=np.concatenate(allLabelList)
    allTrainBinList=np.concatenate(allTrainBinList)
    allParamsList=np.concatenate(allParamsList)
    plotData=pd.DataFrame({'pred':allPredList,'label':allLabelList,'group':'Test','sample':allParamsList})
    plotData['group'].iloc[allTrainBinList]='Training'
    #save CSV file 
    saveFileName=savePath+'/'+saveNameAdd+'_'.join(sorted(statParas))+'_'+'_'.join(contents)+'_'+'binS'+str(binS)+'_meanS'+str(meanSize)+'_'+'_'.join(groups)+'_pred.csv'
    plotData.to_csv(saveFileName, index=False)
    
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
        for sample in leyLists:
            fig.add_trace(go.Violin(y=tmpPlotData['pred'].loc[tmpPlotData['sample']==sample], 
                                    name=sample,
                                    line_color=colorsTrain[sample]))
            count+=1
            meanList.append(tmpPlotData['pred'].loc[tmpPlotData['sample']==sample].mean())
            labelList.append(tmpPlotData['label'].loc[tmpPlotData['sample']==sample].values[0])
        sampleDf=pd.DataFrame({'sample':leyLists,'mean':meanList,'label':labelList})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        layout = dict(xaxis = dict(title = '', showgrid=False, linewidth=3.5, linecolor='black', ticks='inside', mirror=True),
                yaxis = dict(title = '', showgrid=True, ticks='inside',showline=True, linewidth=3.5, linecolor='black', mirror=True,),
                font=dict(size=30, color='black', family='Arial'),
                )
        fig.update_layout(layout)
        # fig.show()
        #save the plot
        saveFileName=savePath+'/'+saveNameAdd+'_'.join(sorted(statParas))+'_'+'_'.join(contents)+'_'+'binS'+str(binS)+'_meanS'+str(meanSize)+'_'+'_'.join(groups)+'_pred_persample_training.html'
        fig.write_html(saveFileName)
        
        #train all
        fig = go.Figure()
        leyLists=list(colorsTrain.keys())
        
        count=0
        for label in leyLists:
            tmpMean=sampleDf['mean'].loc[sampleDf['sample']==label].to_list()
            fig.add_trace(go.Box(y=tmpMean, 
                            x=[sampleDf['label'].loc[sampleDf['sample']==label].values[0]],
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
        saveFileName=savePath+'/'+saveNameAdd+'_'.join(sorted(statParas))+'_'+'_'.join(contents)+'_'+'binS'+str(binS)+'_meanS'+str(meanSize)+'_'+'_'.join(groups)+'_pred_persample_mean_training_scatter.html'
        fig.write_html(saveFileName)
        
#=======#test==============================
        tmpPlotData=plotData[~allTrainBinList]
        fig = go.Figure()
        leyLists=list(colorsTest.keys())
        
        count=0
        meanList=[]
        labelList=[]
        for sample in leyLists:
            #violin with box plot and small sized points for outliers
            fig.add_trace(go.Violin(y=tmpPlotData['pred'].loc[tmpPlotData['sample']==sample], 
                                    name=sample,
                                    line_color='black',
                                    fillcolor=colorsTest[sample],
                                    opacity=0.7,                                    
                                    box_visible=True, 
                                    meanline_visible=True,
                                    marker_size=2,
                                    ))
            count+=1
            meanList=np.append(meanList,tmpPlotData['pred'].loc[tmpPlotData['sample']==sample].mean())
            labelList=np.append(labelList,tmpPlotData['label'].loc[tmpPlotData['sample']==sample].values[0])
        sampleDf=pd.DataFrame({'sample':leyLists,'mean':meanList,'label':labelList})
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
        saveFileName=savePath+'/'+saveNameAdd+'_'.join(sorted(statParas))+'_'+'_'.join(contents)+'_'+'binS'+str(binS)+'_meanS'+str(meanSize)+'_'+'_'.join(groups)+'_pred_persample_test.html'
        fig.write_html(saveFileName)  
        
        #obtain the p value for each sample
        pValMat=np.zeros((len(leyLists),len(leyLists)))
        pValMat=pd.DataFrame(pValMat,index=leyLists,columns=leyLists)
        valVec=np.zeros((len(leyLists)))
        valVec=pd.DataFrame(valVec,index=leyLists,columns=['val'])
        for ref in leyLists:
            valVec['val'].loc[ref]=np.median(tmpPlotData['pred'].loc[tmpPlotData['sample']==ref])
            #check the normality of the distribution using Shapiro-Wilk test
            # refNorm=stats.shapiro(tmpPlotData['pred'].loc[tmpPlotData['sample']==ref])[1]>0.05
            for com in leyLists:
                pValMat.loc[ref,com]=stats.mannwhitneyu(tmpPlotData['pred'].loc[tmpPlotData['sample']==ref],tmpPlotData['pred'].loc[tmpPlotData['sample']==com])[1]
                # comNorm=stats.shapiro(tmpPlotData['pred'].loc[tmpPlotData['sample']==com])[1]>0.05
                # if refNorm==True and comNorm==True:
                #     # pValMat.loc[ref,com]=stats.ttest_ind(tmpPlotData['pred'].loc[tmpPlotData['sample']==ref],tmpPlotData['pred'].loc[tmpPlotData['sample']==com])[1]
                # else:
                #     pValMat.loc[ref,com]=stats.mannwhitneyu(tmpPlotData['pred'].loc[tmpPlotData['sample']==ref],tmpPlotData['pred'].loc[tmpPlotData['sample']==com])[1]

        #save the p value matrix
        pValMat.to_csv(savePath+'/'+saveNameAdd+'_'.join(sorted(statParas))+'_'+'_'.join(contents)+'_'+'binS'+str(binS)+'_meanS'+str(meanSize)+'_pred_persample_pValMat.csv')
        valVec.to_csv(savePath+'/'+saveNameAdd+'_'.join(sorted(statParas))+'_'+'_'.join(contents)+'_'+'binS'+str(binS)+'_meanS'+str(meanSize)+'_pred_persample_valVec.csv')
             
        #test all
        leyLists=list(colorsTestAll.keys())
        #test all with distribution
        fig = go.Figure()
        count=0
        for label in leyLists:
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
            tmpMean=sampleDf['mean'].loc[sampleDf['sample']==label].to_list()
            fig.add_trace(go.Box(y=tmpMean, 
                            x=[sampleDf['label'].loc[sampleDf['sample']==label].values[0]],
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
        saveFileName=savePath+'/'+saveNameAdd+'_'.join(sorted(statParas))+'_'+'_'.join(contents)+'_'+'binS'+str(binS)+'_meanS'+str(meanSize)+'_'+'_'.join(groups)+'_pred_persample_mean_test_scatter_dist.html'
        fig.write_html(saveFileName)
        
        
        leyLists=list(colorsTestAll.keys())
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
        pValMat.to_csv(savePath+'/'+saveNameAdd+'_'.join(sorted(statParas))+'_'+'_'.join(contents)+'_'+'binS'+str(binS)+'_meanS'+str(meanSize)+'_means_pValMat_mean.csv')
        valVec.to_csv(savePath+'/'+saveNameAdd+'_'.join(sorted(statParas))+'_'+'_'.join(contents)+'_'+'binS'+str(binS)+'_meanS'+str(meanSize)+'_means_valVec_mean.csv')
        
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
        pValMat.to_csv(savePath+'/'+saveNameAdd+'_'.join(sorted(statParas))+'_'+'_'.join(contents)+'_'+'binS'+str(binS)+'_meanS'+str(meanSize)+'_means_pValMat_var.csv')
        valVec.to_csv(savePath+'/'+saveNameAdd+'_'.join(sorted(statParas))+'_'+'_'.join(contents)+'_'+'binS'+str(binS)+'_meanS'+str(meanSize)+'_means_valVec_var.csv')
        

                
            

             
        # #swarm plot for each sample from seaborn
        # leyLists=list(colorsTestAll.keys())
        # 
        # for label in leyLists:
        #     #create figure and set the font to be Arial
        #     fig, ax = plt.subplots(figsize=(15, 10))
        #     ax=sns.swarmplot(x="sample", y="pred", 
        #                     data=tmpPlotData.loc[tmpPlotData['label']==label],
        #                     palette=colorsTest,
        #                     size=5*(max(meanSize,5)/20),)
        #     ax.set_xlabel('Sample',fontsize=20)
        #     ax.set_ylabel('Prediction of aging progresson',fontsize=20)
        #     ax.tick_params(axis='both', which='major', labelsize=20)
        #     ax.set_ylim([0,1])
            
        #     #save plot
        #     saveFileName=savePath+'/'+saveNameAdd+'_'.join(sorted(statParas))+'_'+'_'.join(contents)+'_'+'binS'+str(binS)+'_meanS'+str(meanSize)+'_'+'_'.join(groups)+'_treated_swarm_'+label+'.png'
        #     plt.savefig(saveFileName, bbox_inches='tight')
        #     # plt.show()
        #     plt.close()
        
        
#f-test function
def f_test(x,y):
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