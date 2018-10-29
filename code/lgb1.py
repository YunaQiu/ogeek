#!/usr/bin/env python
# -*-coding:utf-8-*-

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
from scipy import sparse
import csv
from datetime import *
import json, random, os

from sklearn.preprocessing import *
import lightgbm as lgb
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectPercentile
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.externals import joblib

from utils import *
from fea1 import getFeaDf


class LgbModel:
    def __init__(self, feaName, cateFea=[], params={}):
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.02,
        	'num_leaves': 80,
            'max_depth': -1,
            # 'min_data_in_leaf': 350,
            # 'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
        	'bagging_freq': 3,
            'verbose': 0,
            'seed': 0,
        }
        self.params.update(**params)
        self.feaName = feaName
        self.cateFea = cateFea
        self.thr = 0.5
        self.modelPath = "../model/"

    def train(self, X, y, num_round=8000, valid=0.05, validX=None, validy=None, early_stopping=200, verbose=True, params={}):
        trainParam = self.params
        trainParam.update(params)
        trainData = lgb.Dataset(X, label=y, feature_name=self.feaName, categorical_feature=self.cateFea)
        if validX is not None:
            validData = trainData.create_valid(validX, label=validy)
            bst = lgb.train(trainParam, trainData, num_boost_round=num_round, valid_sets=[trainData,validData], valid_names=['train', 'valid'], early_stopping_rounds=early_stopping, verbose_eval=verbose)
            self.thr = findF1Threshold(bst.predict(validX), validy)
        elif (valid is not None) and (valid is not False):
            np.random.seed(seed=trainParam['seed'])
            shuffleIdx = np.random.permutation(list(range(len(X))))
            if valid < 1:
                valid = max(int(len(X) * valid), 1)
            trainIdx = shuffleIdx[valid:]
            validIdx = shuffleIdx[:valid]
            preTrain = lgb.Dataset(X[trainIdx], label=y[trainIdx], feature_name=self.feaName, categorical_feature=self.cateFea)
            preValid = preTrain.create_valid(X[validIdx], label=y[validIdx])
            bst = lgb.train(trainParam, preTrain, num_boost_round=num_round, valid_sets=[preTrain,preValid], valid_names=['train', 'valid'], early_stopping_rounds=early_stopping, verbose_eval=verbose)
            self.thr = findF1Threshold(bst.predict(X[validIdx]), y[validIdx])
            num_round = bst.best_iteration
            bst = lgb.train(trainParam, trainData, num_boost_round=num_round)
        else:
            bst = lgb.train(trainParam, trainData, valid_sets=trainData, num_boost_round=num_round, verbose_eval=verbose)
        self.bst = bst
        return (bst.best_iteration if bst.best_iteration>0 else num_round)

    def cv(self, X, y, nfold=5, num_round=8000, early_stopping=200, verbose=True, params={}):
        trainParam = self.params
        trainParam.update(params)
        trainData = lgb.Dataset(X, label=y, feature_name=self.feaName, categorical_feature=self.cateFea)
        result = lgb.cv(trainParam, trainData, feature_name=self.feaName, categorical_feature=self.cateFea, num_boost_round=num_round, nfold=nfold, early_stopping_rounds=early_stopping, verbose_eval=verbose)
        return result

    def predict(self, X):
        return self.bst.predict(X)

    def save(self, modelName):
        joblib.dump(self.bst, self.modelPath + "%s.pkl"%modelName)
        fp = open(self.modelPath + "%s_thr.txt"%modelName, "w")
        fp.write("%.7f"%self.thr)
        fp.close()

    def load(self, modelName):
        self.bst = joblib.load(self.modelPath + "%s.pkl"%modelName)
        fp = open(self.modelPath + "%s_thr.txt"%modelName)
        try:
            self.thr = float(fp.read())
        finally:
            fp.close()

    def feaScore(self, show=True):
        scoreDf = pd.DataFrame({'fea': self.feaName, 'importance': self.bst.feature_importance()})
        scoreDf.sort_values(['importance'], ascending=False, inplace=True)
        if show:
            print(scoreDf[scoreDf.importance>0])
        return scoreDf

    def gridSearch(self, X, y, validX, validy, nFold=5, verbose=0):
        paramsGrids = {
            'num_leaves': [20*i for i in range(2,10)],
            # 'max_depth': list(range(8,13)),
            # 'min_data_in_leaf': [50*i for i in range(2,10)],
            # 'bagging_fraction': [1-0.05*i for i in range(0,5)],
            # 'bagging_freq': list(range(0,10)),

        }
        def getEval(params):
            iter = self.train(X, y, validX=validX, validy=validy, params=params, verbose=verbose)
            return metrics.log_loss(validy, self.predict(validX)), iter
        for k,v in paramsGrids.items():
            resultDf = pd.DataFrame({k: v})
            resultDf['metric_mean'] = list(map(lambda x: getEval({k: x}), v))
        exit()

def main():
    ORIGIN_DATA_PATH = "../data/"
    FEA_OFFLINE_FILE = "../temp/fea_offline.csv"
    FEA_ONLINE_FILE = "../temp/fea_online.csv"
    RESULT_PATH = "../result/"

    # 获取线下特征工程数据集
    if os.path.isfile(FEA_OFFLINE_FILE):
        offlineDf = pd.read_csv(FEA_OFFLINE_FILE)
        offlineDf['prefix_isin_title'] = (offlineDf['prefix_isin_title']>0).astype(int)
    else:
        trainDf = importDf(ORIGIN_DATA_PATH + "oppo_round1_train_20180929.txt", colNames=['prefix','query_prediction','title','tag','label'])
        validDf = importDf(ORIGIN_DATA_PATH + "oppo_round1_vali_20180929.txt", colNames=['prefix','query_prediction','title','tag','label'])
        trainDf, validDf = getFeaDf(trainDf, validDf)
        trainDf['flag'] = 0
        validDf['flag'] = -1
        offlineDf = pd.concat([trainDf, validDf])
        offlineDf.index = list(range(len(offlineDf)))
        exportResult(offlineDf, FEA_OFFLINE_FILE)
        print('offline dataset ready')

    # 获取线上特征工程数据集
    if os.path.isfile(FEA_ONLINE_FILE):
        onlineDf = pd.read_csv(FEA_ONLINE_FILE)
        onlineDf['prefix_isin_title'] = (onlineDf['prefix_isin_title']>0).astype(int)
    else:
        tempDf1 = importDf(ORIGIN_DATA_PATH + "oppo_round1_train_20180929.txt", colNames=['prefix','query_prediction','title','tag','label'])
        tempDf2 = importDf(ORIGIN_DATA_PATH + "oppo_round1_vali_20180929.txt", colNames=['prefix','query_prediction','title','tag','label'])
        df = pd.concat([tempDf1, tempDf2], ignore_index=True)
        testDf = importDf(ORIGIN_DATA_PATH + "oppo_round1_test_A_20180929.txt", colNames=['prefix','query_prediction','title','tag'])
        df, testDf = getFeaDf(df, testDf)
        df['flag'] = 0
        testDf['flag'] = -1
        onlineDf = pd.concat([df, testDf])
        onlineDf.index = list(range(len(onlineDf)))
        exportResult(onlineDf, FEA_ONLINE_FILE)
    print("feature dataset prepare: finished!")
    # exit()

    # 特征筛选
    cateFea = ['tag']
    numFea = [
        'prefix_title_nunique','title_prefix_nunique','prefix_tag_nunique','title_tag_nunique',
        'query_predict_num','query_predict_maxRatio',
        'prefix_newVal','title_newVal',
        'prefix_len','title_len',
        'prefix_isin_title',
        'prefix_label_len','title_label_len','tag_label_len','prefix_title_label_len','prefix_tag_label_len','title_tag_label_len',
        'prefix_label_sum','title_label_sum','tag_label_sum','prefix_title_label_sum','prefix_tag_label_sum','title_tag_label_sum',
        'prefix_label_ratio','title_label_ratio','tag_label_ratio','prefix_title_label_ratio','prefix_tag_label_ratio','title_tag_label_ratio',
        ]
    offlineDf = labelEncoding(offlineDf, cateFea)
    onlineDf = labelEncoding(onlineDf, cateFea)
    fea = cateFea + numFea
    print("model dataset prepare: finished!")

    # 线下数据集
    trainDf = offlineDf[offlineDf.flag==0].reset_index().drop(['index'],axis=1)
    validDf = offlineDf[offlineDf.flag==-1].reset_index().drop(['index'],axis=1)
    trainX = trainDf[fea].values
    trainy = trainDf['label'].values
    print('train:',trainX.shape, trainy.shape)
    validX = validDf[fea]
    validy = validDf['label'].values
    print('valid:',validX.shape, validy.shape)
    # 线上训练集
    df = onlineDf[onlineDf.flag==0].reset_index().drop(['index'],axis=1)
    testDf = onlineDf[onlineDf.flag==-1].reset_index().drop(['index'],axis=1)
    dfX = df[fea].values
    dfy = df['label'].values
    print('df:',dfX.shape, dfy.shape)
    testX = testDf[fea].values
    print('test:',testX.shape)
    print(df[fea].count())
    print('training dataset prepare: finished!')

    # 训练模型
    modelName = "lgb1_isin"
    model = LgbModel(fea)
    # model.load(modelName)
    # model.gridSearch(trainX, trainy, validX, validy)
    iterList = []
    thrList = []
    aucList = []
    f1List = []
    for rd in range(3):
        iterNum = model.train(trainX, trainy, validX=validX, validy=validy, params={'seed':rd}, verbose=2)
        iterList.append(iterNum)
        validDf['pred'] = model.predict(validX)
        # 计算AUC
        fpr, tpr, thresholds = metrics.roc_curve(validy, validDf['pred'], pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print('valid auc:', auc)
        aucList.append(auc)
        # 计算F1值
        thr = model.thr
        thrList.append(thr)
        validDf['predLabel'] = getPredLabel(validDf['pred'], thr)
        f1 = metrics.f1_score(validy, validDf['predLabel'])
        f1List.append(f1)
        print('F1阈值：', thr, '验证集f1分数：', f1)
        print(validDf[['pred','predLabel']].describe())
        print(validDf.groupby('prefix_newVal')[['pred','predLabel']].mean())
        print(validDf.groupby('title_newVal')[['pred','predLabel']].mean())
    print('迭代次数：',iterList, '平均：', np.mean(iterList))
    print('F1阈值：',thrList, '平均：', np.mean(thrList))
    print('auc：', aucList, '平均：', np.mean(aucList))
    print('F1：', f1List, '平均：', np.mean(f1List))
    # 正式模型
    model.thr = np.mean(thrList)
    model.train(dfX, dfy, num_round=int(np.mean(iterList)), valid=False, verbose=False)
    model.feaScore()
    model.save(modelName)

    # 预测结果
    testDf['pred'] = model.predict(testX)
    testDf['predLabel'] = getPredLabel(testDf['pred'], model.thr)
    print(testDf[['pred','predLabel']].describe())
    print(testDf.groupby('prefix_newVal')[['pred','predLabel']].mean())
    print(testDf.groupby('title_newVal')[['pred','predLabel']].mean())
    exportResult(testDf[['predLabel']], RESULT_PATH + "%s.csv"%modelName, header=False)

if __name__ == '__main__':
    startTime = datetime.now()
    main()
    print('total time:', datetime.now() - startTime)
