#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:46:19 2019

@author: burcusyn
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('./Classes/')
# import various AL strategies
from active_learner import ActiveLearnerRandom
from active_learner import ActiveLearnerUncertainty
from active_learner import ActiveLearnerCertainty
from active_learner import ActiveLearnerBlockCertainty
#from active_learner import ActiveLearnerALBE
from active_learner import ActiveLearnerQUIRE
#from active_learner import ActiveLearnerHintSVM
from active_learner import ActiveLearnerLAL
from active_learner import ActiveLearnerMinExpError

# import the dataset class
from dataset import DatasetRecognizingTextualEntailment
from dataset import Emotion
from dataset import AmazonSentiment
from dataset import Crisis
from dataset import Exergame

# import Experiment and Result classes that will be responsible for running AL and saving the results
from experiment import Experiment
from results import Results

#from sklearn import metrics
#import matplotlib.pyplot as plt
#from sklearn.metrics import precision_score, recall_score


if __name__ == '__main__':                  
    
    # number of experiment repeats
    nExperiments = 20

    # number of labeled points at the beginning of the AL experiment
    nStart = 20
    # number of iterations in AL experiment
    nIterations = 40
    # number of items to be queried at each iteration
    batchSize = 20
    # maximum number of votes to ask per item
    maxVoteCount = np.inf

    # the quality metrics computed on the test set to evaluate active learners
    quality_metrics = ['numOfTrainedItems', 'accuracy1', 'accuracy2', 'TN1', 'TN2', 'TP1', 'TP2', 'FN1', 'FN2', 'FP1', 'FP2', 'fbeta11', 'fbeta12', 'fbeta31', 'fbeta32','auc1', 'auc2']
    
    # load dataset
    dtst = DatasetRecognizingTextualEntailment('cleaned')
    
    
    # set the starting point
    dtst.setStartState(nStart)
    
    #Set classifier model
    model = RandomForestClassifier(class_weight='balanced', random_state=2020)
#    
#    # Build classifiers for LAL strategies
#    fn = 'LAL-randomtree-simulatedunbalanced-big.npz'
#    # we found these parameters by cross-validating the regressor and now we reuse these expreiments
#    parameters = {'est': 2000, 'depth': 40, 'feat': 6 }
#    filename = './lal datasets/'+fn
#    regression_data = np.load(filename)
#    regression_features = regression_data['arr_0']
#    regression_labels = regression_data['arr_1']
#    
#    print('Building lal regression model..')
#    lalModel1 = RandomForestRegressor(n_estimators = parameters['est'], max_depth = parameters['depth'], 
#                                     max_features=parameters['feat'], oob_score=True, n_jobs=8)
#    
#    lalModel1.fit(regression_features, np.ravel(regression_labels))    
#    
#    n = 'LAL-iterativetree-simulatedunbalanced-big.npz'
#    # we found these parameters by cross-validating the regressor and now we reuse these expreiments
#    parameters = {'est': 1000, 'depth': 40, 'feat': 6 }
#    filename = './lal datasets/'+fn
#    regression_data = np.load(filename)
#    regression_features = regression_data['arr_0']
#    regression_labels = regression_data['arr_1']
#    
#    print('Building lal regression model..')
#    lalModel2 = RandomForestRegressor(n_estimators = parameters['est'], max_depth = parameters['depth'], 
#                                     max_features=parameters['feat'], oob_score=True, n_jobs=8)
#    
#    lalModel2.fit(regression_features, np.ravel(regression_labels)) 
#    
#    # Active learning strategies
#    alR = ActiveLearnerRandom(dtst, 'random', model, batchSize)
#    alU = ActiveLearnerUncertainty(dtst, 'uncertainty', model, batchSize)
#    alC = ActiveLearnerCertainty(dtst, 'certainty', model, batchSize)
#    alBC100 = ActiveLearnerBlockCertainty(dtst, 'block-certainty-100', model, batchSize, 100)
#    alBC10 = ActiveLearnerBlockCertainty(dtst, 'block-certainty-10', model, batchSize, 10)
#    alBC1 = ActiveLearnerBlockCertainty(dtst, 'block-certainty-1', model, batchSize, 1)
#    alBC01 = ActiveLearnerBlockCertainty(dtst, 'block-certainty-01', model, batchSize, 0.1)
#    alBC001 = ActiveLearnerBlockCertainty(dtst, 'block-certainty-001', model, batchSize, 0.01)
#    alQ = ActiveLearnerQUIRE(dtst, 'quire', model, batchSize, 1.0, 1., 'rbf', 1., 3)
#    alLALindepend = ActiveLearnerLAL(dtst, 'lal-rand', model, batchSize, lalModel1)
#    alLALiterative = ActiveLearnerLAL(dtst, 'lal-iter', model, batchSize, lalModel2)
##    alHSVM = ActiveLearnerHintSVM(dtst, nEstimators, 'hint-SVM', model, batchSize, K, 0.1, 0.1, .5, None, 'linear', 'rbf', 3, 0.1, 0., 1e-3, 1, 100., 0)
    alMinExp = ActiveLearnerMinExpError(dtst, 'minExpError', model, batchSize)

#    als = [alR, alU, alC, alBC100, alBC10, alBC1, alBC01, alBC001, alQ, alLALindepend, alLALiterative, alMinExp]
    als = [alMinExp]
#      
    exp = Experiment(nIterations, quality_metrics, dtst, als, maxVoteCount, 'here we can put a comment about the current experiments')
#    # the Results class helps to add, save and plot results of the experiments
    res = Results(exp, nExperiments)
    
    print("Running AL experiments...")
    for i in range(nExperiments):
        print('\n experiment #'+str(i+1))
        # run an experiment
        performance = exp.run()
        res.addPerformance(performance)
        # reset the experiment (including sampling a new starting state for the dataset)
        exp.reset()
    
    print("Done!")
    res.saveResults('rte_combined_minExp')
    
    resplot = Results()
    resplot.readResult('rte_combined_minExp')
    resplot.plotResults(metrics = ['accuracy', 'f-measure', 'f1', 'f3'])
    
#    res.mergeResults('emotion_combined2', 'emotion_k001', 'emotion_combined')
#    
#    resplot = Results()
#    resplot.readResult('emotion_combined')
#    resplot.plotResults(metrics = ['f1'])
    
    
    
#    # to prepare amazon sentiment dataset
#    da = pd.read_csv('./data/a143145.csv')
#    indexNames = da[ da['_golden'] == True ].index
#    da.drop(indexNames , inplace=True)
#    
#    dc = pd.read_csv('./data/f143145.csv')
#    indexNames = dc[ dc['_golden'] == True ].index
#    dc.drop(indexNames , inplace=True)
#    
#    dt = dc[['_worker_id', '_unit_id', 'choose_one', 'choose_one_gold', 'tweet']]
#    
#    dt.choose_one_gold=dt._unit_id.map(da.set_index('_unit_id').choose_one)
#    
#    dt.columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent']
#    
#    dt.to_csv (r'./data/crisis_determine_type_of_hurricane.csv', index = None, header=True)
#    df = pd.read_csv('./data/crisis_determine_type_of_hurricane.csv')
#    
#    values = df['response'].str.startswith('Informative').values
#    indices = [i for i, x in enumerate(values) if x]
#    df.loc[indices,'response'] = 1
#    values = df['goldLabel'].str.startswith('Informative').values
#    indices = [i for i, x in enumerate(values) if x]
#    df.loc[indices,'goldLabel'] = 1
#    
#    df.loc[df['response'] != 1, 'response'] = 0
#    df.loc[df['goldLabel'] != 1, 'goldLabel'] = 0
#    
#    df.to_csv (r'./data/crisis_determine_type_of_hurricane_related_tweet.csv', index = None, header=True)
    
#    # to see data proportion
#    df2 = df[['taskID', 'goldLabel']].drop_duplicates()
#    df2.goldLabel.value_counts()
    
#    # to find the minimum number of votes per item
#    workerIDs = df.groupby(['taskID'], as_index=False).count()['workerID']
#    workerIDs.tolist().count(0)
#    workerIDs.tolist().count(1)
#    workerIDs.tolist().count(2)
#    
#    # to find the number of workers
#    df.groupby(['workerID'], as_index=False).count()