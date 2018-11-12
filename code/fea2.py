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
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold, GroupKFold
import jieba
from gensim.corpora import Dictionary
from gensim.utils import SaveLoad
import Levenshtein
from gensim.models import TfidfModel

from utils import *
from nlp import *

pd.set_option('display.max_columns',20)

def formatQuery(df):
    '''
    格式化预测词字段
    '''
    tempDf = df[['prefix','query_prediction']].drop_duplicates(subset='prefix')
    def format(x):
        x = eval(x)
        x = {k:float(v) for k,v in x.items()}
        return x
    tempDf['query_prediction'] = tempDf['query_prediction'].dropna().map(lambda x: format(x))
    tempDf['query_predict_num'] = tempDf['query_prediction'].dropna().map(lambda x: len(x))
    tempDf.loc[tempDf.query_predict_num==0, 'query_prediction'] = np.nan
    tempDf['query_predict_num'].fillna(0, inplace=True)
    tempDf['query_word'] = tempDf['query_prediction'].dropna().map(lambda x: sorted(x.keys()))
    tempDf['query_ratio'] = tempDf['query_prediction'].dropna().map(lambda x: [x[i] for i in sorted(x.keys())])
    df = df.drop(['query_prediction'], axis=1).merge(tempDf, how='left', on=['prefix'])
    return df

def addQueryFea(df):
    '''
    提取搜索词字段特征
    '''
    tempDf = df[['prefix','query_prediction','query_word','query_ratio']].drop_duplicates(subset='prefix')
    tempDf['query_predict_maxRatio_pos'] = tempDf['query_ratio'].dropna().map(lambda x: x.index(max(x)))
    tempDf['query_predict_maxRatio'] = tempDf['query_ratio'].dropna().map(lambda x: max(x))
    tempDf['query_predict_max'] = tempDf[['query_word','query_predict_maxRatio_pos']].dropna().apply(lambda x: x.query_word[int(x.query_predict_maxRatio_pos)], axis=1)
    df = df.merge(tempDf[['prefix']+np.setdiff1d(tempDf.columns,df.columns).tolist()], how='left', on=['prefix'])
    return df

def addTextLenFea(df):
    '''
    计算文本长度
    '''
    df['prefix_len'] = df['prefix'].dropna().map(lambda x: len(x.strip()))
    df['title_len'] = df['title'].dropna().map(lambda x: len(x.strip()))
    df['prefix_title_len_diff'] = df['prefix_len'] - df['title_len']
    df['prefix_title_len_ratio'] = df['prefix_len'] / df['title_len']
    df.fillna({'prefix_len':0,'title_len':0}, inplace=True)
    return df

def addPrefixIsinTitle(df):
    '''
    title中是否包含了prefix
    '''
    df['prefix_isin_title'] = df[['prefix','title']].dropna().apply(lambda x: x.title.lower().find(x.prefix.lower())>=0, axis=1).astype(int)
    df['prefix_in_title_ratio'] = df[['prefix_seg','title']].dropna().apply(lambda x: np.mean([1 if x.title.lower().find(w)>=0 else 0 for w in x.prefix_seg]), axis=1)
    return df

def addColSegList(df, stopWordList):
    '''
    将数据集中的文本字段分词
    '''
    startTime = datetime.now()
    tempDf = df[['prefix']].drop_duplicates()
    tempDf['prefix_seg'] = tempDf['prefix'].dropna().map(lambda x: getStrSeg(x, stopWordList))
    tempDf['temp'] = tempDf['prefix_seg'].dropna().map(lambda x: len(x))
    tempDf.loc[tempDf.temp==0, 'prefix_seg'] = np.nan
    df = df.merge(tempDf[['prefix','prefix_seg']], 'left', on='prefix')

    tempDf = df[['title']].drop_duplicates()
    tempDf['title_seg'] = tempDf['title'].dropna().map(lambda x: getStrSeg(x, stopWordList))
    tempDf['temp'] = tempDf['title_seg'].dropna().map(lambda x: len(x))
    tempDf.loc[tempDf.temp==0, 'title_seg'] = np.nan
    df = df.merge(tempDf[['title','title_seg']], 'left', on='title')

    tempDf = df[['prefix','query_word']].drop_duplicates(['prefix'])
    tempDf['query_seg'] = tempDf['query_word'].dropna().map(lambda x: strList2SegList(x, stopWordList))
    df = df.merge(tempDf[['prefix','query_seg']], 'left', on='prefix')
    print('cutword time:', datetime.now() - startTime)
    return df

def addColBowVector(df, dictionary):
    '''
    将文档分词转成词袋向量
    '''
    startTime = datetime.now()
    tempDf = df[['prefix','prefix_seg']].drop_duplicates(['prefix'])
    tempDf['prefix_bow'] = tempDf['prefix_seg'].dropna().map(lambda x: dictionary.doc2bow(x))
    tempDf['temp'] = tempDf['prefix_bow'].dropna().map(lambda x: len(x))
    tempDf.loc[tempDf.temp==0, 'prefix_bow'] = np.nan
    df = df.merge(tempDf[['prefix','prefix_bow']], 'left', on='prefix')

    tempDf = df[['title','title_seg']].drop_duplicates(['title'])
    tempDf['title_bow'] = tempDf['title_seg'].dropna().map(lambda x: dictionary.doc2bow(x))
    tempDf['temp'] = tempDf['title_bow'].dropna().map(lambda x: len(x))
    tempDf.loc[tempDf.temp==0, 'title_bow'] = np.nan
    df = df.merge(tempDf[['title','title_bow']], 'left', on='title')

    tempDf = df[['prefix','query_seg']].drop_duplicates(['prefix'])
    tempDf['query_bow'] = tempDf['query_seg'].dropna().map(lambda x: [dictionary.doc2bow(doc) for doc in x])
    tempDf['temp'] = tempDf['query_bow'].dropna().map(lambda x: max([len(doc) for doc in x]))
    tempDf.loc[tempDf.temp==0, 'query_bow'] = np.nan
    df = df.merge(tempDf[['prefix','query_bow']], 'left', on='prefix')
    print('seg 2 bow time:', datetime.now() - startTime)
    return df

def addTfidfVector(df, tfidfModel):
    '''
    将词袋向量转tfidf向量
    '''
    startTime = datetime.now()
    tempDf = df[['prefix','prefix_bow']].drop_duplicates(['prefix'])
    tempDf['prefix_tfidf'] = tempDf['prefix_bow'].dropna().map(lambda x: tfidfModel[x])
    df = df.merge(tempDf[['prefix','prefix_tfidf']], 'left', on='prefix')

    tempDf = df[['title','title_bow']].drop_duplicates(['title'])
    tempDf['title_tfidf'] = tempDf['title_bow'].dropna().map(lambda x: tfidfModel[x])
    df = df.merge(tempDf[['title','title_tfidf']], 'left', on='title')

    tempDf = df[['prefix','query_bow']].drop_duplicates(['prefix'])
    tempDf['query_tfidf'] = tempDf['query_bow'].dropna().map(lambda x: [tfidfModel[doc] for doc in x])
    df = df.merge(tempDf[['prefix','query_tfidf']], 'left', on='prefix')
    print('tfidf fea time:', datetime.now() - startTime)
    return df

def addTfidfMatrix(df):
    '''
    将搜索词、标题、预测词的tfidf向量拼接成稀疏矩阵
    '''
    tempList = []
    for prefix,title,query in df[['prefix_tfidf','title_tfidf','query_tfidf']].values:
        vecList = []
        vecList.append([] if prefix is np.nan else prefix)
        vecList.append([] if title is np.nan else title)
        vecList.extend([] if query is np.nan else query)
        matrix = sparseVec2Matrix(vecList)
        tempList.append(matrix)
    df['tfidfMatrix'] = tempList
    return df

def addPrefixTitleDist(df):
    '''
    计算搜索词与标题之间的距离
    '''
    startTime = datetime.now()
    tempDf = df.drop_duplicates(['prefix','title'])
    tempDf['prefix_title_levenshtein'] = tempDf[['prefix','title']].dropna().apply(lambda x: Levenshtein.distance(x.prefix.lower(), x.title.lower()), axis=1)
    tempDf['prefix_title_longistStr'] = tempDf[['prefix','title']].dropna().apply(lambda x: len(findLongistSubstr(x.prefix.lower(), x.title.lower())) / len(x.prefix), axis=1)
    # tempDf['prefix_title_cosine'] = tempDf[['prefix_tfidf','title_tfidf']].dropna().apply(lambda x: vectorsDistance([x.prefix_tfidf, x.title_tfidf], metric='cosine')[0,1], axis=1)
    # tempDf['prefix_title_l2'] = tempDf[['prefix_tfidf','title_tfidf']].dropna().apply(lambda x: vectorsDistance([x.prefix_tfidf, x.title_tfidf], metric='l2')[0,1], axis=1)
    tempDf['prefix_title_jaccard'] = tempDf[['prefix_seg','title_seg']].dropna().apply(lambda x: countJaccard(x.prefix_seg, x.title_seg, distance=True), axis=1)
    df = df.merge(tempDf[['prefix','title']+np.setdiff1d(tempDf.columns,df.columns).tolist()], 'left', on=['prefix','title'])
    print('prefix title dist time:', datetime.now() - startTime)
    return df

def addQueryTitleDist(df):
    '''
    计算预测词与标题之间的距离
    '''
    startTime = datetime.now()
    tempDf = df.drop_duplicates(['prefix','title'])
    if 'query_predict_maxRatio_pos' not in tempDf.columns:
        tempDf['query_predict_maxRatio_pos'] = tempDf['query_ratio'].dropna().map(lambda x: x.index(max(x)))

    # tempDf['query_title_cosine'] = tempDf.dropna(subset=['query_tfidf','title_tfidf'])['tfidfMatrix'].map(lambda x: pairwise_distances(x[1:], metric='cosine')[0,1:].tolist())
    # tempDf['query_title_l2'] = tempDf.dropna(subset=['query_tfidf','title_tfidf'])['tfidfMatrix'].map(lambda x: pairwise_distances(x[1:], metric='l2')[0,1:].tolist())
    tempDf['query_title_jaccard'] = tempDf[['query_seg','title_seg']].dropna().apply(lambda x: [countJaccard(doc, x.title_seg, distance=True) for doc in x.query_seg], axis=1)
    tempDf['query_title_levenRatio'] = tempDf[['query_word','title']].dropna().apply(lambda x: [1-Levenshtein.ratio(doc.lower(), x.title.lower()) for doc in x.query_word], axis=1)

    # tempDf['query_title_min_cosine'] = tempDf['query_title_cosine'].dropna().map(lambda x: min(x))
    # tempDf['query_title_minCosine_pos'] = tempDf['query_title_cosine'].dropna().map(lambda x: x.index(min(x)))
    # tempDf['query_title_minCosine_predictRatio'] = tempDf[['query_title_minCosine_pos','query_ratio']].dropna().apply(lambda x: x.query_ratio[int(x.query_title_minCosine_pos)], axis=1)
    # tempDf['query_title_maxRatio_cosine'] = tempDf[['query_title_cosine','query_predict_maxRatio_pos']].dropna().apply(lambda x: x.query_title_cosine[int(x.query_predict_maxRatio_pos)], axis=1)
    # tempDf['query_title_aver_cosine'] = tempDf[['query_title_cosine','query_ratio']].dropna().apply(lambda x: np.sum(np.array(x.query_title_cosine)*np.array(x.query_ratio)) / np.sum(x.query_ratio), axis=1)

    # tempDf['query_title_min_l2'] = tempDf['query_title_l2'].dropna().map(lambda x: min(x))
    # tempDf['query_title_minL2_pos'] = tempDf['query_title_l2'].dropna().map(lambda x: x.index(min(x)))
    # tempDf['query_title_minL2_predictRatio'] = tempDf[['query_title_minL2_pos','query_ratio']].dropna().apply(lambda x: x.query_ratio[int(x.query_title_minL2_pos)], axis=1)
    # tempDf['query_title_maxRatio_l2'] = tempDf[['query_title_l2','query_predict_maxRatio_pos']].dropna().apply(lambda x: x.query_title_l2[int(x.query_predict_maxRatio_pos)], axis=1)
    # tempDf['query_title_aver_l2'] = tempDf[['query_title_l2','query_ratio']].dropna().apply(lambda x: np.sum(np.array(x.query_title_l2)*np.array(x.query_ratio)) / np.sum(x.query_ratio), axis=1)

    tempDf['query_title_min_leven'] = tempDf['query_title_levenRatio'].dropna().map(lambda x: min(x))
    tempDf['query_title_min_leven_pos'] = tempDf['query_title_levenRatio'].dropna().map(lambda x: x.index(min(x)))
    tempDf['query_title_minLeven_predictRatio'] = tempDf[['query_title_min_leven_pos','query_ratio']].dropna().apply(lambda x: x.query_ratio[int(x.query_title_min_leven_pos)], axis=1)
    tempDf['query_title_maxRatio_leven'] = tempDf[['query_title_levenRatio','query_predict_maxRatio_pos']].dropna().apply(lambda x: x.query_title_levenRatio[int(x.query_predict_maxRatio_pos)], axis=1)
    tempDf['query_title_aver_leven'] = tempDf[['query_title_levenRatio','query_ratio']].dropna().apply(lambda x: np.sum(np.array(x.query_title_levenRatio)*np.array(x.query_ratio)) / np.sum(x.query_ratio), axis=1)

    tempDf['query_title_min_jaccard'] = tempDf['query_title_jaccard'].dropna().map(lambda x: min(x))
    tempDf['query_title_min_jaccard_pos'] = tempDf['query_title_jaccard'].dropna().map(lambda x: x.index(min(x)))
    tempDf['query_title_minJacc_predictRatio'] = tempDf[['query_title_min_jaccard_pos','query_ratio']].dropna().apply(lambda x: x.query_ratio[int(x.query_title_min_jaccard_pos)], axis=1)
    tempDf['query_title_maxRatio_jacc'] = tempDf[['query_title_jaccard','query_predict_maxRatio_pos']].dropna().apply(lambda x: x.query_title_jaccard[int(x.query_predict_maxRatio_pos)], axis=1)
    tempDf['query_title_aver_jacc'] = tempDf[['query_title_jaccard','query_ratio']].dropna().apply(lambda x: np.sum(np.array(x.query_title_jaccard)*np.array(x.query_ratio)) / np.sum(x.query_ratio), axis=1)

    df = df.merge(tempDf[['prefix','title'] + np.setdiff1d(tempDf.columns,df.columns).tolist()], 'left', on=['prefix','title'])
    print('query title dist time:', datetime.now() - startTime)
    return df

def addCrossColNunique(df, statDf, col1, col2, alias=None):
    '''
    统计每个col1中col2的种类数
    '''
    if alias is None:
        alias = '%s_%s' % (col1, col2)
    if '%s_nunique'%alias in statDf.columns:
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
        df['%s_label_len'%alias] = df['%s_label_sum'%alias] = 0
        df['%s_label_ratio'%alias] = df['%s_label_ratio2'%alias] = np.nan
        return df
    if '%s_label_ratio'%alias in statDf.columns:
        tempDf = statDf[colArr + ['%s_label_len'%alias, '%s_label_sum'%alias, '%s_label_ratio'%alias, '%s_label_ratio2'%alias]].drop_duplicates(subset=colArr)
        df = df.merge(tempDf, how='left', on=colArr)
        df.fillna({'%s_label_len'%alias: 0, '%s_label_sum'%alias: 0}, inplace=True)
    else:
        tempDf = statDf.groupby(colArr)['label'].agg([len, 'sum'])
        tempDf['ratio'] = biasSmooth(tempDf['sum'], tempDf['len'])
        tempDf['ratio2'] = tempDf['sum'] / tempDf['len']
        tempDf.loc[:,['len','sum']] /= tempDf[['len','sum']].sum()
        tempDf.columns = ['%s_label_%s'%(alias,x) for x in tempDf.columns]
        df = df.merge(tempDf.reset_index(), 'left', on=colArr)
        df.fillna({'%s_label_len'%alias: 0, '%s_label_sum'%alias: 0}, inplace=True)
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
            tempDf = df[[col]].drop_duplicates()
            existList = np.intersect1d(tempDf[col].dropna(),statDf[col].dropna())
            tempDf['%s_newVal'%col] = (~tempDf[col].isin(existList)).astype(int)
            df = df.merge(tempDf, 'left', on=[col])
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

def addCvHisFea(df, nFold=5, random_state=0):
    '''
    多折交叉添加历史统计特征
    '''
    kf = StratifiedKFold(n_splits=nFold, random_state=random_state, shuffle=True)
    dfList = []
    for i, (statIdx, taskIdx) in enumerate(kf.split(df.values, df['label'].values)):
        tempDf = addHisFeas(df.iloc[taskIdx], df.iloc[statIdx])
        dfList.append(tempDf)
    df = pd.concat(dfList, ignore_index=True)
    return df

def getHisFeaDf(df, nFold=5, newRatio=0.4, random_state=0):
    '''
    构建含一定新数据比例的统计数据集
    '''
    startTime = datetime.now()
    kf = StratifiedKFold(n_splits=nFold, shuffle=True, random_state=random_state)
    dfList = []
    for i, (statIdx, taskIdx) in enumerate(kf.split(df.values, df['label'].values, groups=df['prefix'].values)):
        tempDf = addHisFeas(df.iloc[taskIdx], df.iloc[statIdx])
        tempDf.drop(tempDf[(tempDf.prefix_newVal==1)|(tempDf.title_newVal==1)].index, inplace=True)
        dfList.append(tempDf)
    oldDf = pd.concat(dfList, ignore_index=True)
    print('get exist his time:', datetime.now() - startTime)

    startTime = datetime.now()
    kf = GroupKFold(n_splits=nFold)
    dfList = []
    for i, (statIdx, taskIdx) in enumerate(kf.split(df.values, df['label'].values, groups=df['prefix'].values)):
        tempDf = addHisFeas(df.iloc[taskIdx], df.iloc[statIdx])
        dfList.append(tempDf)
    newDf = pd.concat(dfList, ignore_index=True)
    print('get new his time:', datetime.now() - startTime)

    newSampleNum = int(len(oldDf)/(1-newRatio)*newRatio)
    df = pd.concat([oldDf, newDf.sample(n=newSampleNum, random_state=random_state)], ignore_index=True)
    print(df.shape[0], oldDf.shape[0], newSampleNum)
    return df

def addGlobalFeas(df, statDf=None):
    '''
    添加全局特征
    '''
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
    return df

def addTextFeas(df, **params):
    '''
    添加文本特征
    '''
    df = addQueryFea(df)
    df = addTextLenFea(df)
    # df = addPrefixIsinTitle(df)
    # df = addTfidfMatrix(df)
    df = addPrefixTitleDist(df)
    df = addQueryTitleDist(df)
    return df

class FeaFactory:
    def __init__(self, dfFile, name='fea', cachePath="../temp/", nFold=5, seed=0):
        self.dfFile = dfFile
        self.name = name
        self.dictionary = {}
        self.tfidfModel = {}
        self.cachePath = cachePath if cachePath[-1]=="/" else (cachePath+'/')
        self.nfold = nFold
        self.seed = seed
        self.stopWords = [w.replace('\n','') for w in open("./user_stopwords.dat", 'r', encoding='utf-8').readlines()]
        jieba.load_userdict("./user_dict.dat")

        self.formatEval = ['query_prediction','query_word','query_ratio']
        self.textEval = self.formatEval + ['prefix_seg','title_seg','query_seg']
        self.tfidfEval = self.textEval + ['prefix_bow','title_bow','query_bow','prefix_tfidf','title_tfidf','query_tfidf']
        self.textFeaEval = self.formatEval + ['query_title_cosine','query_title_l2','query_title_jaccard','query_title_levenRatio']

    def getFormatDf(self, dfName):
        '''
        格式化原始数据集
        '''
        startTime = datetime.now()
        df = importDf(self.dfFile[dfName], colNames=['prefix','query_prediction','title','tag','label']).head(20000)
        df['prefix'] = df['prefix'].astype(str)
        df['title'] = df['title'].astype(str)
        df['prefix_title'] = df['prefix'] + '_' + df['title']
        df['id'] = list(range(len(df)))
        df = formatQuery(df)
        print('format %s cost time:'%dfName, datetime.now() - startTime)
        return df

    def getTextSegDf(self, dfName):
        '''
        对数据集文本进行分词处理
        '''
        # startTime = datetime.now()
        df = self.getFormatDf(dfName)
        df = addColSegList(df, self.stopWords)
        # print('get %s textSeg time:'%dfName, datetime.now() - startTime)
        return df

    def getTfidfVecDf(self, dfName, type='offline'):
        '''
        数据集分词文本转tfidf向量
        '''
        # startTime = datetime.now()
        df = self.getTextSegDf(dfName)
        if type not in self.dictionary:
            self.loadDictionary(type)
        if type not in self.tfidfModel:
            self.loadTfidfModel(type)
        df = addColBowVector(df, self.dictionary[type])
        df = addTfidfVector(df, self.tfidfModel[type])
        # print('get %s %s TfidfDf time:'%(dfName, type), datetime.now() - startTime)
        return df

    def getDocList(self, dfName):
        '''
        获取数据集文本字段的文档列表
        '''
        filePath = self.cachePath + '%s_doclist_%s.txt'%(self.name, dfName)
        if os.path.isfile(filePath):
            with open(filePath, encoding='utf-8') as fp:
                docList = [line.replace('\n','').split(" ") for line in fp.readlines()]
        else:
            startTime = datetime.now()
            docList = []
            df = self.getTextSegDf(dfName)
            docList.extend(df['title_seg'].dropna().values)
            df['query_seg'].dropna().map(lambda x:docList.extend(x))
            docList = [x for x in docList if len(x)>0]
            # saveDocList(docList, filePath)
            print('make %s doclist time:'%dfName, datetime.now() - startTime)
        return docList

    def getDocLists(self):
        '''
        获取数据集文本字段的文档列表
        '''
        filePath = self.cachePath + '%s_doclist.txt'%(self.name)
        if os.path.isfile(filePath):
            with open(filePath, encoding='utf-8') as fp:
                docList = [line.replace('\n','').split(" ") for line in fp.readlines()]
        else:
            startTime = datetime.now()
            dfList = []
            for dfName in self.dfFile.keys():
                dfList.append(self.getTextSegDf(dfName))
            df = pd.concat(dfList, ignore_index=True)
            docList = []
            docList.extend(df.drop_duplicates('title')['title_seg'].dropna().values)
            df.drop_duplicates('prefix')['query_seg'].dropna().map(lambda x:docList.extend(x))
            docList = [x for x in docList if len(x)>0]
            saveDocList(docList, filePath)
            print('make doclist time:', datetime.now() - startTime)
        return docList

    def loadDictionary(self, type='offline'):
        '''
        加载字典，若字典不存在则建立字典
        '''
        startTime = datetime.now()
        filePath = self.cachePath + '%s_dictionary_%s.txt'%(self.name, type)
        if os.path.isfile(filePath):
            dictionary = Dictionary.load_from_text(filePath)
        else:
            if type=='offline':
                docList = self.getDocList('train')
                dictionary = makeDictionary(docList)
            elif type=='all':
                docList = []
                if os.path.isfile(self.cachePath + '%s_dictionary_online.txt'%self.name):
                    print('dictionary continue')
                    docList.extend(self.getDocList('testA'))
                    docList.extend(self.getDocList('testB'))
                    dictionary = makeDictionary(docList, dictFile=self.cachePath + '%s_dictionary_online.txt'%self.name, add=True)
                else:
                    for dfName in self.dfFile.keys():
                        docList.extend(self.getDocList(dfName))
                    dictionary = makeDictionary(docList)
            elif type=='online' and os.path.isfile(self.cachePath + '%s_dictionary_offline.txt'%self.name):
                docList = self.getDocList('valid')
                dictionary = makeDictionary(docList, dictFile=self.cachePath + '%s_dictionary_offline.txt'%self.name, add=True)
            else:
                docList = self.getDocList('train')
                docList.extend(self.getDocList('valid'))
                dictionary = makeDictionary(docList)
            dictionary.save_as_text(filePath)
            print('make dictionary time:', datetime.now() - startTime)
        self.dictionary[type] = dictionary
        return dictionary

    def updateDictionary(self, dfName):
        '''
        更新词典
        '''
        startTime = datetime.now()
        filePath = self.cachePath + '%s_dictionary.txt'%self.name
        if self.dictionary is None:
            self.loadDictionary()
        docList = self.getDocList(dfName)
        self.dictionary = makeDictionary(docList, filePath, add=True)
        self.dictionary.save_as_text(filePath)
        print('update dictionary time:', datetime.now() - startTime)

    def loadTfidfModel(self, type='offline'):
        '''
        加载Tfidf模型，若模型不存在则建立模型
        '''
        filePath = self.cachePath + '%s_tfidf_%s.model'%(self.name,type)
        if os.path.isfile(filePath):
            tfidfModel = SaveLoad.load(filePath)
        else:
            startTime = datetime.now()
            if type not in self.dictionary:
                self.loadDictionary(type)
            tfidfModel = TfidfModel(dictionary=self.dictionary[type])
            # tfidfModel = makeTfidfModel(self.dictionary)
            tfidfModel.save(filePath)
            print('train tfidfModel time:', datetime.now() - startTime)
        self.tfidfModel[type] = tfidfModel
        return tfidfModel

    def getTextFeaDf(self, df, type='all'):
        '''
        获取数据集的文本特征，缓存文本特征文件
        '''
        startTime = datetime.now()
        print('----------get textFea begin----------')
        df = addColSegList(df, self.stopWords)
        if 'prefix_title' not in df.columns:
            df['prefix_title'] = df['prefix'].astype(str)+'_'+df['title'].astype(str)
        cachePath = self.cachePath + '%s_textFea_%s.csv'%(self.name,type)
        if os.path.isfile(cachePath):
            if self.textFeaCacheDf is None:
                totalDf = importCacheDf(cachePath)
            else:
                totalDf = self.textFeaCacheDf
            tempList = []
            tempSeries = df.prefix_title.isin(totalDf.prefix_title.values)
            tempDf = df[tempSeries].drop(['prefix_seg','title_seg','query_seg'],axis=1)
            tempDf = tempDf.merge(totalDf[['prefix','title']+np.setdiff1d(totalDf.columns,tempDf.columns).tolist()], 'left', on=['prefix','title'])
            tempList.append(tempDf)
            if tempDf.shape[0] < df.shape[0]:
                tempDf = df[~tempSeries]
                print('----------make %d new textFea begin----------'%tempDf.shape[0])
                startTime2 = datetime.now()
                tempDf = addTextFeas(tempDf).drop(['prefix_seg','title_seg','query_seg'],axis=1)
                print('----------make %d textFea end----------'%tempDf.shape[0])
                print('make %d new textFea his time:'%tempDf.shape[0], datetime.now() - startTime2)
                tempList.append(tempDf)
                addDf = tempDf.drop(['tag','label','id','query_prediction','query_predict_num','query_word','query_ratio'],axis=1).drop_duplicates(['prefix','title'])
                totalDf = pd.concat([totalDf,addDf], ignore_index=True)
                exportResult(totalDf, cachePath)
                self.textFeaCacheDf = totalDf
            df = pd.concat(tempList, ignore_index=True)
        else:
            print('----------make %d new textFea begin----------'%df.shape[0])
            startTime2 = datetime.now()
            df = addTextFeas(df).drop(['prefix_seg','title_seg','query_seg'],axis=1)
            print('----------make %d textFea end----------'%df.shape[0])
            print('make %d new textFea his time:'%df.shape[0], datetime.now() - startTime2)
            totalDf = df.drop(['tag','label','id','query_prediction','query_predict_num','query_word','query_ratio'],axis=1).drop_duplicates(['prefix','title'])
            exportResult(totalDf, cachePath)
            self.textFeaCacheDf = totalDf
        # exportResult(df, filePath)
        print('----------get textFea end----------')
        print('cost time:', datetime.now() - startTime)
        return df

    def getOfflineDf(self, type='all', newRatio=0.4):
        '''
        获取线下模型特征数据集
        '''
        filePath = self.cachePath + '%s_offline_stat.csv'%self.name
        if os.path.isfile(filePath):
            offlineDf = importCacheDf(filePath)
        else:
            # 获取规范化数据集
            trainDf = self.getFormatDf('train')
            testDf = self.getFormatDf('valid')
            trainDf['flag'] = 0
            testDf['flag'] = 1

            # 获取历史统计特征
            startTime = datetime.now()
            hisDf = addHisFeas(testDf, trainDf)
            tempDf = addCvHisFea(trainDf, newRatio=newRatio, random_state=self.seed)
            # tempDf = getHisFeaDf(trainDf, nFold=self.nfold, newRatio=newRatio, random_state=self.seed)
            offlineDf = pd.concat([tempDf,hisDf], ignore_index=True)
            print('prepare offline his time:', datetime.now() - startTime)

            # 获取全局统计特征
            startTime = datetime.now()
            offlineDf = addGlobalFeas(offlineDf, pd.concat([trainDf,testDf]))
            print('prepare offline global time:', datetime.now() - startTime)
            exportResult(offlineDf, filePath)

        # 文本分词特征
        startTime = datetime.now()
        self.textFeaCacheDf = None
        offlineDf = self.getTextFeaDf(offlineDf)
        print('prepare offline textFea time:', datetime.now() - startTime)

        print(offlineDf.count())
        # exportResult(offlineDf, self.cachePath + '%s_offline.csv'%self.name)
        return offlineDf

    def getOnlineDf(self, type='all', newRatio=0.4):
        '''
        获取线上模型特征数据集
        '''
        filePath = self.cachePath + '%s_online_stat.csv'%self.name
        if os.path.isfile(filePath):
            onlineDf = importCacheDf(filePath)
        else:
            # 获取规范化数据集
            trainDf = self.getFormatDf('train')
            validDf = self.getFormatDf('valid')
            testDf = self.getFormatDf('testA')
            trainDf['flag'] = 0
            validDf['flag'] = 1
            statDf = pd.concat([trainDf, validDf], ignore_index=True)
            testDf['flag'] = -1

            # 获取历史统计特征
            startTime = datetime.now()
            hisDf = addHisFeas(testDf, statDf)
            tempDf = addCvHisFea(statDf, newRatio=newRatio, random_state=self.seed)
            # tempDf = getHisFeaDf(statDf, nFold=self.nfold, newRatio=newRatio, random_state=self.seed)
            onlineDf = pd.concat([tempDf,hisDf], ignore_index=True)
            print('prepare online his time:', datetime.now() - startTime)

            # 获取全局统计特征
            startTime = datetime.now()
            onlineDf = addGlobalFeas(onlineDf, pd.concat([trainDf,validDf,testDf]))
            print('prepare online global time:', datetime.now() - startTime)
            exportResult(onlineDf, filePath)

        # 文本分词特征
        startTime = datetime.now()
        self.textFeaCacheDf = None
        onlineDf = self.getTextFeaDf(onlineDf)
        print('prepare online textFea time:', datetime.now() - startTime)

        print(onlineDf.count())
        # exportResult(onlineDf, self.cachePath + '%s_online.csv'%self.name)
        return onlineDf

if __name__ == '__main__':
    list1 = [2,23,4,4,2,3]
    list2 = list1.copy()
    list2.remove(min(list1))
    list2.remove(max(list1))
    print(list1, list2, np.mean(list2))
    exit()

    startTime = datetime.now()
    ORIGIN_DATA_PATH = "../data/"
    dfFile = {
        'train': ORIGIN_DATA_PATH + "oppo_round1_train_20180929.txt",
        'valid': ORIGIN_DATA_PATH + "oppo_round1_vali_20180929.txt",
        'testA': ORIGIN_DATA_PATH + "oppo_round1_test_A_20180929.txt",
        # 'testB':ORIGIN_DATA_PATH + "oppo_round1_test_A_20180929.txt",
    }
    factory = FeaFactory(dfFile, name="fea2", cachePath="../temp/")
    df = factory.getOfflineDf()
    df2 = factory.getOnlineDf()
    print('feaFactory A time:', datetime.now() - startTime)

    # dfFile = {
    #     'train': ORIGIN_DATA_PATH + "oppo_train.txt",
    #     'valid': ORIGIN_DATA_PATH + "oppo_round1_vali_20180929.txt",
    #     'testA': ORIGIN_DATA_PATH + "oppo_round1_test_A_20180929.txt",
    #     'testB': ORIGIN_DATA_PATH + "oppo_round1_test_A_20180929.txt",
    # }
    # factory = FeaFactory(dfFile, name="fea2", cachePath="../temp/")
    # # factory.updateDictionary('testB')
    # df = factory.getOfflineDf()
    # df3 = factory.getOnlineDfB()
    # print('feaFactory B time:', datetime.now() - startTime)
