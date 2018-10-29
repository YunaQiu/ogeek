#!/usr/bin/env python
# -*-coding:utf-8-*-

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
from scipy import sparse
import csv
import matplotlib.dates
import matplotlib.pyplot as plt
from datetime import *
import urllib, urllib.parse, urllib.request
import json, random, os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
import jieba
from gensim.corpora import Dictionary
# from gensim.models import *

from utils import *

jieba.load_userdict("../data/user_dict.dat")
STOP_WORDS = [w.replace('\n','') for w in open("../data/user_stopwords.dat", 'r', encoding='utf-8').readlines()]


def formatQuery(df):
    '''
    格式化预测词字段
    '''
    startTime = datetime.now()
    tempDf = df[['prefix','query_prediction']].drop_duplicates(subset='prefix')
    def format(x):
        x = eval(x)
        x = {k:float(v) for k,v in x.items()}
        return x
    tempDf['query_prediction'] = tempDf['query_prediction'].map(lambda x: format(x))
    tempDf['query_predict_num'] = tempDf['query_prediction'].map(lambda x: len(x))
    tempDf.loc[tempDf.query_predict_num==0, 'query_prediction'] = np.nan
    tempDf['query_word'] = tempDf['query_prediction'].dropna().map(lambda x: list(x.keys()))
    df = df.drop(['query_prediction'], axis=1).merge(tempDf, how='left', on=['prefix'])
    print('format query time:', datetime.now() - startTime)
    return df

def addQueryFea(df):
    '''
    提取搜索词字段特征
    '''
    startTime = datetime.now()
    tempDf = df[['prefix','query_prediction']].drop_duplicates(subset='prefix')
    tempDf['query_predict_maxRatio'] = tempDf['query_prediction'].dropna().map(lambda x: max(list(x.values())))
    tempDf['query_predict_max'] = tempDf['query_prediction'].dropna().map(lambda x: list(x.keys())[list(x.values()).index(max(list(x.values())))])
    df = df.merge(tempDf.drop(['query_prediction'], axis=1), how='left', on=['prefix'])
    print('query fea time:', datetime.now() - startTime)
    return df

def addTextLenFea(df):
    '''
    计算文本长度
    '''
    df['prefix_len'] = df['prefix'].map(lambda x: len(x) if x==x else 0)
    df['title_len'] = df['title'].map(lambda x: len(x) if x==x else 0)
    return df

def addPrefixIsinTitle(df):
    '''
    title中是否包含了prefix
    '''
    startTime = datetime.now()
    df['prefix_isin_title'] = df[['prefix_seg','title']].dropna().apply(lambda x: np.mean([1 if x.title.lower().find(w)>=0 else 0 for w in x.prefix_seg]), axis=1)
    print('prefix in title cost time:', datetime.now() - startTime)
    return df

def addColSegList(df):
    '''
    将数据集中的文本字段分词
    '''
    startTime = datetime.now()
    df['prefix_seg'] = df['prefix'].dropna().map(lambda x: getStrSeg(x, STOP_WORDS))
    df['temp'] = df['prefix_seg'].dropna().map(lambda x: len(x))
    df.loc[df.temp==0, 'prefix_seg'] = np.nan
    print('prefix cutword time:', datetime.now() - startTime)

    startTime = datetime.now()
    df['title_seg'] = df['title'].dropna().map(lambda x: getStrSeg(x, STOP_WORDS))
    df['temp'] = df['title_seg'].dropna().map(lambda x: len(x))
    df.loc[df.temp==0, 'title_seg'] = np.nan
    print('title cutword time:', datetime.now() - startTime)

    startTime = datetime.now()
    df['query_seg'] = df['query_word'].dropna().map(lambda x: strList2SegList(x, STOP_WORDS))
    print('query cutword time:', datetime.now() - startTime)
    return df

def addColBowList(df, dictionary):
    '''
    将文档分词转成词袋向量
    '''
    startTime = datetime.now()
    df['prefix_bow'] = df['prefix_seg'].dropna().map(lambda x: dictionary.doc2bow(x))
    df['title_bow'] = df['title_seg'].dropna().map(lambda x: dictionary.doc2bow(x))
    print('prefix & title bow time:', datetime.now() - startTime)

    startTime = datetime.now()
    df['query_bow'] = df['query_seg'].dropna().map(lambda x: [dictionary.doc2bow(doc) for doc in x])
    print('query bow time:', datetime.now() - startTime)
    return df

def addCrossColNunique(df, statDf, col1, col2, alias=None):
    '''
    统计每个col1中col2的种类数
    '''
    if alias is None:
        alias = '%s_%s' % (col1, col2)
    if '%s_nunique'%alias in statDf.columns:
        print('%s col exist'%('%s_nunique'%alias))
        tempDf = statDf[[col1, '%s_nunique'%alias]].drop_duplicates(subset=col1)
        df = df.merge(tempDf, how='left', on=col1)
    else:
        tempDf = statDf.groupby(col1)[col2].nunique().to_frame()
        tempDf.columns = ['%s_nunique'%alias]
        df = df.merge(tempDf.reset_index(), how='left', on=col1)
    return df

def addLabelFea(df, statDf, colArr, alias=None):
    '''
    统计列的标签独立数和标签点击率
    '''
    if len(np.array(colArr).shape) == 0:
        colArr = [colArr]
    if alias is None:
        alias = '_'.join(np.array(colArr).astype(str))
    if statDf[colArr].count().min() == 0:
        df['%s_label_len'%alias] = df['%s_label_sum'%alias] = np.nan
        df['%s_label_ratio'%alias] = np.nan
        return df
    if '%s_label_ratio'%alias in statDf.columns:
        print('%s col exist'%('%s_label_ratio'%alias))
        tempDf = statDf[colArr + ['%s_label_len'%alias, '%s_label_sum'%alias, '%s_label_ratio'%alias]].drop_duplicates(subset=colArr)
        df = df.merge(tempDf, how='left', on=colArr)
    else:
        tempDf = statDf.groupby(colArr)['label'].agg([len, 'sum'])
        tempDf['ratio'] = biasSmooth(tempDf['sum'], tempDf['len'])
        tempDf.loc[:,['len','sum']] /= tempDf[['len','sum']].sum()
        tempDf.columns = ['%s_label_%s'%(alias,x) for x in tempDf.columns]
        df = df.merge(tempDf.reset_index(), 'left', on=colArr)
        # df.fillna({'%s_label_len'%alias: 0, '%s_label_sum'%alias: 0}, inplace=True)
    return df

def addNewValFea(df, statDf, cols):
    '''
    判断字段值是否在统计表中出现过
    '''
    if len(np.array(cols).shape) == 0:
        cols = [cols]
    for col in cols:
        if statDf[col].count() == 0:
            df['%s_newVal'%col] = 1
        else:
            df['%s_newVal'%col] = (~df[col].isin(statDf[col].unique())).astype(int)
    return df

def addHisFeas(df, statDf):
    '''
    添加历史统计类特征
    '''
    startTime = datetime.now()
    df = addNewValFea(df, statDf, ['prefix','title','tag'])
    # 统计点击率特征
    colList = ['prefix','title','tag',['prefix','title'],['prefix','tag'],['title','tag']]
    for col in colList:
        df = addLabelFea(df, statDf, col)
    return df

def addGlobalFeas(df, statDf=None):
    '''
    添加全局特征
    '''
    startTime = datetime.now()
    if statDf is None:
        statDf = df
    # 统计交叉维度独立数
    crossColList = [
        ['prefix','title'],
        ['title','prefix'],
        ['prefix','tag'],
        ['title','tag'],
    ]
    for c1,c2 in crossColList:
        df = addCrossColNunique(df, statDf, c1, c2)
    df = formatQuery(df)
    df = addQueryFea(df)
    df = addTextLenFea(df)
    df = addColSegList(df)
    df = addPrefixIsinTitle(df)
    print('add global fea time:', datetime.now() - startTime)
    return df

def addTextFeas(df):
    '''
    添加文本特征
    '''
    startTime = datetime.now()
    df = addQueryFea(df)
    df = addTextLenFea(df)
    df = addColSegList(df)
    df = addPrefixIsinTitle(df)
    print('add text fea time:', datetime.now() - startTime)
    return df

def getFeaDf(trainDf, testDf, nFold=5):
    '''
    对传入的训练集和测试集构造特征
    '''
    trainDf['index_order'] = list(range(trainDf.shape[0]))
    testDf['index_order'] = list(range(testDf.shape[0]))

    trainDf = formatQuery(trainDf)
    testDf = formatQuery(testDf)
    trainDf = addTextFeas(trainDf)
    testDf = addTextFeas(testDf)
    trainDf = addGlobalFeas(trainDf, trainDf)
    testDf = addGlobalFeas(testDf, trainDf)
    testDf = addHisFeas(testDf, trainDf)
    kf = StratifiedKFold(n_splits=nFold, random_state=0, shuffle=True)
    dfList = []
    for i, (statIdx, taskIdx) in enumerate(kf.split(trainDf.values, trainDf['label'].values)):
        tempDf = addHisFeas(trainDf.iloc[taskIdx], trainDf.iloc[statIdx])
        dfList.append(tempDf)
    trainDf = pd.concat(dfList, ignore_index=True)

    trainDf = trainDf.sort_values(by=['index_order']).drop(['index_order'], axis=1)
    testDf = testDf.sort_values(by=['index_order']).drop(['index_order'], axis=1)
    trainDf.index = list(range(trainDf.shape[0]))
    testDf.index = list(range(testDf.shape[0]))
    return trainDf, testDf

class FeaFactory:
    def __init__(self, dfFile, name='fea', cachePath="../temp/", nFold=5, seed=0):
        self.dfFile = dfFile
        self.name = name
        self.dictionary = None
        self.cachePath = cachePath if cachePath[-1]=="/" else (cachePath+'/')
        self.nfold = nFold
        self.seed = seed

    def getOriginDf(self, dfName):
        df = importDf(self.dfFile[dfName], colNames=['prefix','query_prediction','title','tag','label']).head(10000)
        df['prefix'] = df['prefix'].astype(str)
        df['title'] = df['title'].astype(str)
        df['id'] = list(range(len(df)))
        return df

    def getFormatDf(self, dfName):
        '''
        格式化原始数据集
        '''
        if os.path.isfile(self.cachePath + '%s_format_%s.csv'%(self.name, dfName)):
            df = pd.read_csv(self.cachePath + '%s_format_%s.csv'%(self.name, dfName))
            df['query_prediction'] = df['query_prediction'].dropna().map(lambda x: eval(x))
            df['query_word'] = df['query_word'].dropna().map(lambda x: eval(x))
        else:
            df = self.getOriginDf(dfName)
            df = formatQuery(df)
            exportResult(df, self.cachePath + '%s_format_%s.csv'%(self.name, dfName))
        return df

    def getTextSegDf(self, dfName):
        '''
        对数据集文本进行分词处理
        '''
        filePath = self.cachePath + '%s_text_%s.csv'%(self.name, dfName)
        if os.path.isfile(filePath):
            df = pd.read_csv(filePath)
            evalList = ['query_prediction','query_word','prefix_seg','title_seg','query_seg']
            df.loc[:,evalList] = df[evalList].applymap(lambda x: eval(x) if x==x else x)
        else:
            df = self.getFormatDf(dfName)
            df = addColSegList(df)
            exportResult(df, filePath)
        return df

    # def loadDictionary(self):
    #     '''
    #     加载字典，若字典不存在则建立字典
    #     '''
    #     filePath = self.cachePath + '%s_dictionary.txt'%(self.name, dfName)
    #     if os.path.isfile(dictFile):
    #         self.dictionary = Dictionary.load_from_text(filePath)
    #     else:
    #         docList = []
    #         for dfName in self.dfFile.keys():
    #             tempDf = self.getTextSegDf(dfName)

    def getOfflineDf(self):
        if os.path.isfile(self.cachePath + '%s_offline.csv'%self.name):
            return pd.read_csv(self.cachePath + '%s_offline.csv'%self.name)

        # 获取规范化数据集
        trainDf = self.getFormatDf('train')
        testDf = self.getFormatDf('valid')
        trainDf['flag'] = 0
        testDf['flag'] = 1
        offlineDf = pd.concat([trainDf,testDf])

        # 全局统计特征
        if os.path.isfile(self.cachePath + '%s_offline_global.csv'%self.name):
            globalDf = pd.read_csv(self.cachePath + '%s_offline_global.csv'%self.name)
        else:
            globalDf = addGlobalFeas(offlineDf, offlineDf)
            exportResult(globalDf, self.cachePath + '%s_offline_global.csv'%self.name)
        offlineDf = offlineDf.merge(globalDf[['flag','id']+np.setdiff1d(globalDf.columns, offlineDf.columns).tolist()], 'left', on=['flag','id'])

        # 历史统计特征
        if os.path.isfile(self.cachePath + '%s_offline_his.csv'%self.name):
            hisDf = pd.read_csv(self.cachePath + '%s_offline_his.csv'%self.name)
        else:
            hisDf = addHisFeas(testDf, trainDf)
            kf = StratifiedKFold(n_splits=self.nfold, random_state=self.seed, shuffle=True)
            dfList = []
            for i, (statIdx, taskIdx) in enumerate(kf.split(trainDf.values, trainDf['label'].values)):
                tempDf = addHisFeas(trainDf.iloc[taskIdx], trainDf.iloc[statIdx])
                dfList.append(tempDf)
            hisDf = pd.concat(dfList+[hisDf])
            exportResult(hisDf, self.cachePath + '%s_offline_his.csv'%self.name)
        offlineDf = offlineDf.merge(hisDf[['flag','id']+np.setdiff1d(hisDf.columns, offlineDf.columns).tolist()], 'left', on=['flag','id'])
        print(offlineDf.info())

        # 文本分词特征
        trainDf = self.getTextSegDf('train')
        print(trainDf.info())
        print(trainDf.head())
        # offlineDf.index = list(range(len(offlineDf)))
        # exportResult(offlineDf, FEA_OFFLINE_FILE)
        # print('offline dataset ready')
        return offlineDf

if __name__ == '__main__':
    startTime = datetime.now()
    ORIGIN_DATA_PATH = "../data/"
    dfFile = {
        'train': ORIGIN_DATA_PATH + "oppo_round1_train_20180929.txt",
        'valid': ORIGIN_DATA_PATH + "oppo_round1_vali_20180929.txt",
        'testA':ORIGIN_DATA_PATH + "oppo_round1_test_A_20180929.txt",
    }
    factory = FeaFactory(dfFile, name="fea_test", cachePath="../temp/")
    df = factory.getOfflineDf()
    # print(df.head())
    print('feaFactory time:', datetime.now() - startTime)
