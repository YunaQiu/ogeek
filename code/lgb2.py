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
from fea2 import FeaFactory


class LgbModel:
    def __init__(self, feaName, cateFea=[], params={}):
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
#             'metric': 'binary_logloss',
            'metric': 'custom',
            'learning_rate': 0.08,
        	'num_leaves': 100,
            'max_depth': -1,
            'min_data_in_leaf': 50,
            # 'feature_fraction': 0.9,
            'bagging_fraction': 0.95,
        	'bagging_freq': 1,
            'verbose': 0,
            'seed': 0,
            'num_threads': 15,
        }
        self.params.update(**params)
        self.feaName = feaName
        self.cateFea = cateFea
        # self.thr = 0.4
        self.modelPath = "./model/"

    def custom_eval(self, preds, train_data):
        '''
        自定义F1评价函数
        '''
        labels = train_data.get_label()
        predLabels = getPredLabel(preds, self.thr)
        f1 = metrics.f1_score(labels, predLabels)
        return 'f1', f1, True

    def train(self, X, y, num_round=8000, valid=0.05, validX=None, validy=None, early_stopping=500, verbose=True, params={}, thr=None):
        trainParam = self.params
        trainParam.update(params)
        self.thr = self.thr if thr is None else thr
        trainData = lgb.Dataset(X, label=y, feature_name=self.feaName, categorical_feature=self.cateFea)
        if validX is not None:
            validData = trainData.create_valid(validX, label=validy)
            bst = lgb.train(trainParam, trainData, num_boost_round=num_round, valid_sets=[trainData,validData], valid_names=['train', 'valid'], feval=self.custom_eval, early_stopping_rounds=early_stopping, verbose_eval=verbose)#
            # self.thr = findF1Threshold(bst.predict(validX), validy)
        elif (valid is not None) and (valid is not False):
            np.random.seed(seed=trainParam['seed'])
            shuffleIdx = np.random.permutation(list(range(len(X))))
            if valid < 1:
                valid = max(int(len(X) * valid), 1)
            trainIdx = shuffleIdx[valid:]
            validIdx = shuffleIdx[:valid]
            preTrain = lgb.Dataset(X[trainIdx], label=y[trainIdx], feature_name=self.feaName, categorical_feature=self.cateFea)
            preValid = preTrain.create_valid(X[validIdx], label=y[validIdx])
            bst = lgb.train(trainParam, preTrain, num_boost_round=num_round, valid_sets=[preTrain,preValid], valid_names=['train', 'valid'], feval=self.custom_eval, early_stopping_rounds=early_stopping, verbose_eval=verbose)
            # self.thr = findF1Threshold(bst.predict(X[validIdx]), y[validIdx])
            num_round = bst.best_iteration
            bst = lgb.train(trainParam, trainData, feval=self.custom_eval, num_boost_round=num_round)
        else:
            bst = lgb.train(trainParam, trainData, valid_sets=trainData, feval=self.custom_eval, num_boost_round=num_round, verbose_eval=verbose)
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
        with open(self.modelPath + "%s_thr.txt"%modelName, "w") as fp:
            fp.write("%.7f"%self.thr)
#         with open(self.modelPath + "%s_splitThr.txt"%modelName, "w") as fp:
#             fp.write("%.7f\n%.7f"%tuple(self.splitThr))

    def load(self, modelName):
        self.bst = joblib.load(self.modelPath + "%s.pkl"%modelName)
        with open(self.modelPath + "%s_thr.txt"%modelName) as fp:
            self.thr = float(fp.read())
#         with open(self.modelPath + "%s_splitThr.txt"%modelName) as fp:
#             self.splitThr = []
#             self.splitThr.append(float(fp.readline()))
#             self.splitThr.append(float(fp.readline()))

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
            'min_data_in_leaf': [20*i for i in range(1,10)],
            'bagging_fraction': [1-0.05*i for i in range(0,5)],
            'bagging_freq': list(range(0,10)),

        }
        def getEval(params):
            iter = self.train(X, y, validX=validX, validy=validy, params=params, verbose=verbose)
            fpr, tpr, thresholds = metrics.roc_curve(validy, self.predict(validX), pos_label=1)
            auc = metrics.auc(fpr, tpr)
            print(params, iter, auc)
            return auc, iter
        for k,v in paramsGrids.items():
            resultDf = pd.DataFrame({k: v})
            resultDf['metric_mean'] = list(map(lambda x: getEval({k: x}), v))
        exit()

def main():
    ORIGIN_DATA_PATH = "../data/oppo_data_ronud2_20181107/"
    RESULT_PATH = "./result/"

    # 获取线下特征工程数据集
    startTime = datetime.now()
    dfFile = {
        'train': ORIGIN_DATA_PATH + "data_train.txt",
        'valid': ORIGIN_DATA_PATH + "data_vali.txt",
        'testA': ORIGIN_DATA_PATH + "data_test.txt",
    }
    factory = FeaFactory(dfFile, name='fea2', cachePath="./temp/")
#     docList = factory.getDocLists()
#     print('length of doclist:', len(docList))
#     exit()

    offlineDf = factory.getOfflineDf()
    print('offline dataset ready!')
    onlineDf = factory.getOnlineDf()
    # onlineDf = factory.getOnlineDfB()
    print("feature dataset prepare: finished!")
    print("cost time:", datetime.now() - startTime)
    # exit()

    # 特征筛选
    cateFea = ['tag']
    numFea = [
        'prefix_title_nunique','title_prefix_nunique','prefix_tag_nunique','title_tag_nunique',
        'query_predict_num','query_predict_maxRatio',
        'title_newVal','prefix_newVal',
        'prefix_len','title_len','prefix_title_len_ratio',#'prefix_title_len_diff',
        'prefix_label_len','title_label_len','tag_label_len','prefix_title_label_len','prefix_tag_label_len','title_tag_label_len',
        'prefix_label_sum','title_label_sum','tag_label_sum','prefix_title_label_sum','prefix_tag_label_sum','title_tag_label_sum',
        # 'prefix_label_ratio2','title_label_ratio2','tag_label_ratio2','prefix_title_label_ratio2','prefix_tag_label_ratio2','title_tag_label_ratio2',
        'prefix_label_ratio','title_label_ratio','tag_label_ratio','prefix_title_label_ratio','prefix_tag_label_ratio','title_tag_label_ratio',
        'prefix_title_levenshtein','prefix_title_longistStr','prefix_title_jaccard',#'prefix_title_cosine','prefix_title_l2',
        # 'query_title_aver_cosine','query_title_maxRatio_cosine','query_title_min_cosine','query_title_minCosine_predictRatio',
        # 'query_title_aver_l2','query_title_maxRatio_l2','query_title_min_l2','query_title_minL2_predictRatio',
        'query_title_aver_jacc','query_title_maxRatio_jacc','query_title_min_jaccard','query_title_minJacc_predictRatio',
        'query_title_aver_leven','query_title_maxRatio_leven','query_title_min_leven','query_title_minLeven_predictRatio',
        ]
    offlineDf = labelEncoding(offlineDf, cateFea)
    onlineDf = labelEncoding(onlineDf, cateFea)
    fea = cateFea + numFea
    print("model dataset prepare: finished!")

    # 线下数据集
    trainDf = offlineDf[offlineDf.flag==0].sort_values(by=['id'])#.reset_index().drop(['index'],axis=1)
    validDf = offlineDf[offlineDf.flag==1].sort_values(by=['id'])#.reset_index().drop(['index'],axis=1)
    trainX = trainDf[fea].values
    trainy = trainDf['label'].values
    print('train:',trainX.shape, trainy.shape)
    validX = validDf[fea]
    validy = validDf['label'].values
    print('valid:',validX.shape, validy.shape)
    # 线上训练集
    df = onlineDf[onlineDf.flag>=0].sort_values(by=['id'])#.reset_index().drop(['index'],axis=1)
    testDf = onlineDf[onlineDf.flag==-1].sort_values(by=['id'])#.reset_index().drop(['index'],axis=1)
    dfX = df[fea].values
    dfy = df['label'].values
    print('df:',dfX.shape, dfy.shape)
    testX = testDf[fea].values
    print('test:',testX.shape)
    print(onlineDf.groupby('flag')[fea].count().T)
    print('training dataset prepare: finished!')

    # 训练模型
    modelName = "lgb2_new"
    model = LgbModel(fea)
    # model.load(modelName)
    # model.gridSearch(trainX, trainy, validX, validy)
    iterList = []
    thrList = []
    loglossList = []
    f1List = []
    splitF1List = []
    thrExistList = []
    thrNewList = []
    for rd in range(5):
#         trainX = trainDf.sample(frac=0.95, random_state=rd)[fea].values
#         trainy = trainDf.sample(frac=0.95, random_state=rd)['label'].values
        iterNum = model.train(trainX, trainy, validX=validX, validy=validy, params={'seed':rd}, verbose=3)
        iterList.append(iterNum)
        validDf['pred'] = model.predict(validX)
        # 计算AUC
        logloss = metrics.log_loss(validy, validDf['pred'])
        loglossList.append(logloss)
        # 计算F1值
        thr = findF1Threshold(validDf['pred'], validy)
        thrList.append(thr)
        validDf['predLabel'] = getPredLabel(validDf['pred'], model.thr)
        validDf['predLabel2'] = getPredLabel(validDf['pred'], thr)
        f1 = metrics.f1_score(validy, validDf['predLabel'])
        f1List.append(f1)

        print('F1阈值：', thr, '验证集f1分数：', f1List[-1], '修正阈值后F1：', metrics.f1_score(validy, validDf['predLabel2']))
        print(validDf[['pred','predLabel']].describe())
        print(validDf.groupby('prefix_newVal')[['pred','predLabel']].mean())
        print(validDf.groupby('title_newVal')[['pred','predLabel']].mean())
    # exportResult(validDf[['pred']], RESULT_PATH + "%s_valid.csv"%modelName)
    print('迭代次数：',iterList, '平均：', removeExtremeMean(iterList))
    print('F1阈值：',thrList, '平均：', np.mean(thrList))
    print('logloss：', loglossList, '平均：', np.mean(loglossList))
    print('F1：', f1List, '平均：', np.mean(f1List))
    # 正式模型
    model.thr = np.mean(thrList)
#     model.splitThr = [np.mean(thrExistList),np.mean(thrNewList)]
    model.train(dfX, dfy, num_round=int(removeExtremeMean(iterList)), valid=False, verbose=False)
    model.save(modelName)
    model.feaScore()

    # 预测结果
    testDf['pred'] = model.predict(testX)
    testDf['predLabel'] = getPredLabel(testDf['pred'], model.thr)
#     testDf['predLabel2'] = 0
#     testDf.loc[testDf.prefix_newVal==1,'predLabel2'] = getPredLabel(testDf[testDf.prefix_newVal==1]['pred'], model.splitThr[1])
#     testDf.loc[testDf.prefix_newVal==0,'predLabel2'] = getPredLabel(testDf[testDf.prefix_newVal==0]['pred'], model.splitThr[0])
    print(testDf[['pred','predLabel']].describe())
    print(testDf.groupby('prefix_newVal')[['pred','predLabel']].mean())
    print(testDf.groupby('title_newVal')[['pred','predLabel']].mean())
    exportResult(testDf[['pred']], RESULT_PATH + "%s_pred.csv"%modelName)
    exportResult(testDf[['predLabel']], RESULT_PATH + "%s.csv"%modelName, header=False)
#     exportResult(testDf[['predLabel2']], RESULT_PATH + "%s_thr.csv"%modelName, header=False)

if __name__ == '__main__':
    startTime = datetime.now()
    main()
    print('total time:', datetime.now() - startTime)
