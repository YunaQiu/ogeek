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
from fea1 import addGlobalFeas, addHisFeas


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

    def train(self, X, y, num_round=8000, validX=None, validy=None, early_stopping=200, verbose=True, params={}):
        trainData = lgb.Dataset(X, label=y, feature_name=self.feaName, categorical_feature=self.cateFea)
        trainParam = self.params
        trainParam.update(params)
        if isinstance(validX, (pd.DataFrame, sparse.csr_matrix)):
            validData = trainData.create_valid(validX, label=validy)
            bst = lgb.train(trainParam, trainData, num_boost_round=num_round, valid_sets=[trainData,validData], valid_names=['train', 'valid'], early_stopping_rounds=early_stopping, verbose_eval=verbose)
        else:
            bst = lgb.train(trainParam, trainData, valid_sets=trainData, num_boost_round=num_round, verbose_eval=verbose)
        self.bst = bst
        return bst.best_iteration

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

def makeNewDf(df, statDf, ratio=[0.4,0.2]):
    '''
    构造无历史记录数据集：打乱df后按比例分为3种情况处理
    '''
    df = df.sample(frac=1, random_state=0)
    df['flag'] = 2
    prefixNum = round(len(df) * ratio[0])
    titleNum = round(len(df) * ratio[1])

    # 新prefix旧title
    prefixDf = df.iloc[:prefixNum]
    tempDf = statDf.copy()
    tempDf['prefix'] = np.nan
    prefixDf = addHisFeas(prefixDf, tempDf)
    # 新title旧prefix
    titleDf = df.iloc[prefixNum:prefixNum+titleNum]
    tempDf = statDf.copy()
    tempDf['title'] = np.nan
    titleDf = addHisFeas(titleDf, tempDf)
    # title和prefix都是新的
    bothDf = df.iloc[prefixNum+titleNum:]
    tempDf = statDf.copy()
    tempDf['prefix'] = tempDf['title'] = np.nan
    bothDf = addHisFeas(bothDf, tempDf)
    df = pd.concat([prefixDf,titleDf,bothDf], ignore_index=True)
    return df

def addTrainTiming(df, nFold=5):
    '''
    分批构造训练集历史特征
    '''
    kf = StratifiedKFold(n_splits=nFold, random_state=0, shuffle=True)
    dfList = []
    for i, (statIdx, taskIdx) in enumerate(kf.split(df.values, df['label'].values)):
        tempDf = addHisFeas(df.iloc[taskIdx], df.iloc[statIdx])
        dfList.append(tempDf)
    df = pd.concat(dfList, ignore_index=True)
    return df

def main():
    ORIGIN_DATA_PATH = "../data/"
    FEA_OFFLINE_FILE = "../temp/fea_offline.csv"
    FEA_ONLINE_FILE = "../temp/fea_online.csv"
    RESULT_PATH = "../result/"

    # 获取线下特征工程数据集
    if not os.path.isfile(FEA_OFFLINE_FILE):
        df = importDf(ORIGIN_DATA_PATH + "oppo_round1_train_20180929.txt", colNames=['prefix','query_prediction','title','tag','label'])
        df['order'] = list(range(len(df)))
        df['flag'] = 0
        validDf = importDf(ORIGIN_DATA_PATH + "oppo_round1_vali_20180929.txt", colNames=['prefix','query_prediction','title','tag','label'])
        validDf['order'] = list(range(len(validDf)))
        validDf['flag'] = 1
        testADf = importDf(ORIGIN_DATA_PATH + "oppo_round1_test_A_20180929.txt", colNames=['prefix','query_prediction','title','tag'])
        testADf['order'] = list(range(len(testADf)))
        testADf['flag'] = -1
        statDf = pd.concat([df,validDf], ignore_index=True)
        offlineDf = pd.concat([df,validDf,testADf], ignore_index=True)

        offlineDf = addGlobalFeas(offlineDf, df)
        copyDf = makeNewDf(offlineDf[offlineDf.flag==0], df, [0.55,0.09])
        offlineDf = addHisFeas(offlineDf, df)
        offlineDf = pd.concat([offlineDf,copyDf], ignore_index=True)
        # trainDf = addTrainTiming(offlineDf[offlineDf.flag==0])
        # testDf = addHisFeas(offlineDf[offlineDf.flag!=0], df)
        # offlineDf = pd.concat([trainDf,testDf], ignore_index=True)
        exportResult(offlineDf, FEA_OFFLINE_FILE)
        print('offline dataset ready')
    else:
        offlineDf = pd.read_csv(FEA_OFFLINE_FILE)
    # 获取线上特征工程数据集
    if not os.path.isfile(FEA_ONLINE_FILE):
        df = importDf(ORIGIN_DATA_PATH + "oppo_round1_train_20180929.txt", colNames=['prefix','query_prediction','title','tag','label'])
        df['order'] = list(range(len(df)))
        df['flag'] = 0
        validDf = importDf(ORIGIN_DATA_PATH + "oppo_round1_vali_20180929.txt", colNames=['prefix','query_prediction','title','tag','label'])
        validDf['order'] = list(range(len(validDf)))
        validDf['flag'] = 1
        testADf = importDf(ORIGIN_DATA_PATH + "oppo_round1_test_A_20180929.txt", colNames=['prefix','query_prediction','title','tag'])
        testADf['order'] = list(range(len(testADf)))
        testADf['flag'] = -1
        statDf = pd.concat([df,validDf], ignore_index=True)
        onlineDf = pd.concat([df,validDf,testADf], ignore_index=True)
        statDf = pd.concat([df,validDf], ignore_index=True)

        onlineDf = addGlobalFeas(onlineDf, statDf)
        copyDf = makeNewDf(onlineDf[onlineDf.flag==0], statDf, [0.31,0.24])
        onlineDf = addHisFeas(onlineDf, statDf)
        onlineDf = pd.concat([onlineDf,copyDf], ignore_index=True)
        # trainDf = addTrainTiming(onlineDf[onlineDf.flag>=0])
        # testDf = addHisFeas(onlineDf[onlineDf.flag<0], statDf)
        # onlineDf = pd.concat([trainDf,testDf], ignore_index=True)
        exportResult(onlineDf, FEA_ONLINE_FILE)
    else:
        onlineDf = pd.read_csv(FEA_ONLINE_FILE)
    print("feature dataset prepare: finished!")
    # exit()

    # 特征筛选
    cateFea = ['tag']
    numFea = [
        'prefix_title_nunique','title_prefix_nunique','prefix_tag_nunique','title_tag_nunique',
        'query_predict_num','query_predict_maxRatio',
        'prefix_newVal','title_newVal',
        'prefix_label_len','title_label_len','tag_label_len','prefix_title_label_len','prefix_tag_label_len','title_tag_label_len',
        'prefix_label_sum','title_label_sum','tag_label_sum','prefix_title_label_sum','prefix_tag_label_sum','title_tag_label_sum',
        'prefix_label_ratio','title_label_ratio','tag_label_ratio','prefix_title_label_ratio','prefix_tag_label_ratio','title_tag_label_ratio',
        ]
    offlineDf = labelEncoding(offlineDf, cateFea)
    onlineDf = labelEncoding(onlineDf, cateFea)
    fea = cateFea + numFea
    print("model dataset prepare: finished!")

    # 线下数据集
    copyIdx = offlineDf[offlineDf.flag==2].sample(frac=0.3, random_state=0).index.tolist()
    trainIdx = offlineDf[offlineDf.flag==0].index.tolist()
    validIdx = offlineDf[offlineDf.flag==1].index.tolist()
    trainX = offlineDf.loc[trainIdx+copyIdx][fea]
    trainX.index = list(range(len(trainX)))
    trainy = offlineDf.loc[trainIdx+copyIdx]['label'].values
    print('train:',trainX.shape, trainy.shape)
    validX = offlineDf.loc[validIdx][fea]
    validX.index = list(range(len(validX)))
    validy = offlineDf.loc[validIdx]['label'].values
    print('valid:',validX.shape, validy.shape)
    # 线上训练集
    copyIdx = onlineDf[onlineDf.flag==2].sample(frac=0.15, random_state=0).index.tolist()
    testIdx = onlineDf[onlineDf.flag==-1].index.tolist()
    dfX = onlineDf.loc[trainIdx + validIdx + copyIdx][fea]
    dfX.index = list(range(len(dfX)))
    dfy = onlineDf.loc[trainIdx + validIdx + copyIdx]['label'].values
    print('df:',dfX.shape, dfy.shape)
    testX = onlineDf.loc[testIdx].sort_values(by=['order'])[fea]
    testX.index = list(range(len(testX)))
    print('test:',testX.shape)
    print(dfX.count())
    print('training dataset prepare: finished!')

    # 训练模型
    modelName = "lgb1_split_onoff"
    model = LgbModel(fea)
    # model.load(modelName)
    # model.gridSearch(trainX, trainy, validX, validy)
    iterList = []
    thrList = []
    for rd in range(3):
        iterNum = model.train(trainX, trainy, validX=validX, validy=validy, params={'seed':rd})
        iterList.append(iterNum)
        # 计算F1值
        validX['predy'] = model.predict(validX)
        thr = findF1Threshold(validX['predy'], validy)
        thrList.append(thr)
        print('F1阈值：', thr, '验证集f1分数：', metrics.f1_score(validy, getPredLabel(validX['predy'], thr)))
        validX['predyLabel'] = getPredLabel(validX['predy'], thr)
        print(validX[['predy','predyLabel']].describe())
        print(validX.groupby('prefix_newVal')[['predy','predyLabel']].mean())
        print(validX.groupby('title_newVal')[['predy','predyLabel']].mean())
    print('迭代次数：',iterList, '平均：', np.mean(iterList))
    print('F1阈值：',thrList, '平均：', np.mean(thrList))
    # 正式模型
    model.thr = np.mean(thrList)
    model.train(dfX, dfy, num_round=int(np.mean(iterList)), verbose=False)
    model.feaScore()
    model.save(modelName)

    # 预测结果
    predictDf = onlineDf[onlineDf.flag==-1]
    predictDf['predicted_score'] = model.predict(testX)
    predictDf['predicted_label'] = getPredLabel(predictDf['predicted_score'], model.thr)
    print(predictDf[['predicted_score','predicted_label']].describe())
    print(predictDf.groupby('prefix_newVal')[['predicted_score','predicted_label']].mean())
    print(predictDf.groupby('title_newVal')[['predicted_score','predicted_label']].mean())
    exportResult(predictDf[['predicted_label']], RESULT_PATH + "%s.csv"%modelName, header=False)

if __name__ == '__main__':
    startTime = datetime.now()
    main()
    print('total time:', datetime.now() - startTime)
