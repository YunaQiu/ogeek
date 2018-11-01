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
from gensim.utils import SaveLoad
import Levenshtein
# from gensim.models import *

from utils import *
from nlp import *

pd.set_option('display.max_columns',10)

# STOP_WORDS = [w.replace('\n','') for w in open("../data/user_stopwords.dat", 'r', encoding='utf-8').readlines()]

def formatQuery(df):
    '''
    格式化预测词字段
    '''
    tempDf = df[['prefix','query_prediction']].drop_duplicates(subset='prefix')
    def format(x):
        x = eval(x)
        x = {k:float(v) for k,v in x.items()}
        return x
    tempDf['query_prediction'] = tempDf['query_prediction'].map(lambda x: format(x))
    tempDf['query_predict_num'] = tempDf['query_prediction'].map(lambda x: len(x))
    tempDf.loc[tempDf.query_predict_num==0, 'query_prediction'] = np.nan
    tempDf['query_word'] = tempDf['query_prediction'].dropna().map(lambda x: list(x.keys()))
    tempDf['query_ratio'] = tempDf['query_prediction'].dropna().map(lambda x: list(x.values()))
    df = df.drop(['query_prediction'], axis=1).merge(tempDf, how='left', on=['prefix'])
    return df

def addQueryFea(df):
    '''
    提取搜索词字段特征
    '''
    tempDf = df[['prefix','query_prediction']].drop_duplicates(subset='prefix')
    tempDf['query_ratio'] = tempDf['query_prediction'].dropna().map(lambda x: list(x.values()))
    tempDf['query_word'] = tempDf['query_prediction'].dropna().map(lambda x: list(x.keys()))
    tempDf['query_predict_maxRatio_pos'] = tempDf['query_ratio'].dropna().map(lambda x: x.index(max(x)))
    tempDf['query_predict_maxRatio'] = tempDf['query_ratio'].dropna().map(lambda x: max(x))
    tempDf['query_predict_max'] = tempDf[['query_word','query_predict_maxRatio_pos']].dropna().apply(lambda x: x.query_word[int(x.query_predict_maxRatio_pos)], axis=1)
    df = df.merge(tempDf[['prefix']+np.setdiff1d(tempDf.columns,df.columns).tolist()], how='left', on=['prefix'])
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
    df['prefix_isin_title'] = df[['prefix','title']].dropna().apply(lambda x: x.title.lower().find(x.prefix.lower())>=0, axis=1).astype(int)
    df['prefix_in_title_ratio'] = df[['prefix_seg','title']].dropna().apply(lambda x: np.mean([1 if x.title.lower().find(w)>=0 else 0 for w in x.prefix_seg]), axis=1)
    return df

def addColSegList(df, stopWordList):
    '''
    将数据集中的文本字段分词
    '''
    startTime = datetime.now()
    df['prefix_seg'] = df['prefix'].dropna().map(lambda x: getStrSeg(x, stopWordList))
    df['temp'] = df['prefix_seg'].dropna().map(lambda x: len(x))
    df.loc[df.temp==0, 'prefix_seg'] = np.nan
    print('prefix cutword time:', datetime.now() - startTime)

    startTime = datetime.now()
    df['title_seg'] = df['title'].dropna().map(lambda x: getStrSeg(x, stopWordList))
    df['temp'] = df['title_seg'].dropna().map(lambda x: len(x))
    df.loc[df.temp==0, 'title_seg'] = np.nan
    print('title cutword time:', datetime.now() - startTime)

    startTime = datetime.now()
    df['query_seg'] = df['query_word'].dropna().map(lambda x: strList2SegList(x, stopWordList))
    print('query cutword time:', datetime.now() - startTime)
    return df.drop(['temp'],axis=1)

def addColBowVector(df, dictionary):
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

def addTfidfVector(df, tfidfModel):
    '''
    将词袋向量转tfidf向量
    '''
    startTime = datetime.now()
    df['prefix_tfidf'] = df['prefix_bow'].dropna().map(lambda x: tfidfModel[x])
    df['title_tfidf'] = df['title_bow'].dropna().map(lambda x: tfidfModel[x])
    df['query_tfidf'] = df['query_bow'].dropna().map(lambda x: [tfidfModel[doc] for doc in x])
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
    tempDf['prefix_title_cosine'] = tempDf[['prefix_tfidf','title_tfidf']].dropna().apply(lambda x: vectorsDistance([x.prefix_tfidf, x.title_tfidf], metric='cosine')[0,1], axis=1)
    tempDf['prefix_title_l2'] = tempDf[['prefix_tfidf','title_tfidf']].dropna().apply(lambda x: vectorsDistance([x.prefix_tfidf, x.title_tfidf], metric='l2')[0,1], axis=1)
    tempDf['prefix_title_jaccard'] = tempDf[['prefix_seg','title_seg']].dropna().apply(lambda x: countJaccard(x.prefix_seg, x.title_seg, distance=True), axis=1)
    df = df.merge(tempDf[['prefix','title','prefix_title_levenshtein','prefix_title_longistStr','prefix_title_cosine','prefix_title_l2','prefix_title_jaccard']], 'left', on=['prefix','title'])
    print('prefix title dist time:', datetime.now() - startTime)
    return df

def addQueryTitleDist(df):
    '''
    计算预测词与标题之间的距离
    '''
    startTime = datetime.now()
    tempDf = df.drop_duplicates(['prefix','title'])
    if 'query_ratio' not in tempDf.columns:
        tempDf['query_ratio'] = tempDf['query_prediction'].dropna().map(lambda x: list(x.values()))
        tempDf['query_predict_maxRatio_pos'] = tempDf['query_ratio'].dropna().map(lambda x: x.index(max(x)))

    tempDf['query_title_cosine'] = tempDf.dropna(subset=['query_tfidf','title_tfidf'])['tfidfMatrix'].map(lambda x: pairwise_distances(x[1:], metric='cosine')[0,1:].tolist())
    tempDf['query_title_l2'] = tempDf.dropna(subset=['query_tfidf','title_tfidf'])['tfidfMatrix'].map(lambda x: pairwise_distances(x[1:], metric='l2')[0,1:].tolist())
    tempDf['prefix_title_jaccard'] = tempDf[['prefix_seg','title_seg']].dropna().apply(lambda x: countJaccard(x.prefix_seg, x.title_seg, distance=True), axis=1)

    tempDf['query_title_min_cosine'] = tempDf['query_title_cosine'].dropna().map(lambda x: min(x))
    tempDf['query_title_minCosine_pos'] = tempDf['query_title_cosine'].dropna().map(lambda x: x.index(min(x)))
    tempDf['query_title_minCosine_predictRatio'] = tempDf[['query_title_minCosine_pos','query_ratio']].dropna().apply(lambda x: x.query_ratio[int(x.query_title_minCosine_pos)], axis=1)
    tempDf['query_title_maxRatio_cosine'] = tempDf[['query_title_cosine','query_predict_maxRatio_pos']].dropna().apply(lambda x: x.query_title_cosine[int(x.query_predict_maxRatio_pos)], axis=1)
    tempDf['query_title_aver_cosine'] = tempDf[['query_title_cosine','query_ratio']].dropna().apply(lambda x: np.sum(np.array(x.query_title_cosine)*np.array(x.query_ratio)) / np.sum(x.query_ratio), axis=1)

    tempDf['query_title_min_l2'] = tempDf['query_title_l2'].dropna().map(lambda x: min(x))
    tempDf['query_title_minL2_pos'] = tempDf['query_title_l2'].dropna().map(lambda x: x.index(min(x)))
    tempDf['query_title_minL2_predictRatio'] = tempDf[['query_title_minL2_pos','query_ratio']].dropna().apply(lambda x: x.query_ratio[int(x.query_title_minL2_pos)], axis=1)
    tempDf['query_title_maxRatio_l2'] = tempDf[['query_title_l2','query_predict_maxRatio_pos']].dropna().apply(lambda x: x.query_title_l2[int(x.query_predict_maxRatio_pos)], axis=1)
    tempDf['query_title_aver_l2'] = tempDf[['query_title_l2','query_ratio']].dropna().apply(lambda x: np.sum(np.array(x.query_title_l2)*np.array(x.query_ratio)) / np.sum(x.query_ratio), axis=1)

    tempDf['query_title_jaccard'] = tempDf[['query_seg','title_seg']].dropna().apply(lambda x: [countJaccard(doc, x.title_seg, distance=True) for doc in x.query_seg], axis=1)
    tempDf['query_title_min_jaccard'] = tempDf['query_title_jaccard'].dropna().map(lambda x: min(x))
    tempDf['query_title_min_jaccard_pos'] = tempDf['query_title_jaccard'].dropna().map(lambda x: x.index(min(x)))
    tempDf['query_title_minJacc_predictRatio'] = tempDf[['query_title_min_jaccard_pos','query_ratio']].dropna().apply(lambda x: x.query_ratio[int(x.query_title_min_jaccard_pos)], axis=1)
    tempDf['query_title_maxRatio_jacc'] = tempDf[['query_title_jaccard','query_predict_maxRatio_pos']].dropna().apply(lambda x: x.query_title_jaccard[int(x.query_predict_maxRatio_pos)], axis=1)

    df = df.merge(tempDf[['prefix','title',
    'query_title_cosine','query_title_l2','query_title_jaccard',
    'query_title_min_cosine','query_title_minCosine_pos','query_title_minCosine_predictRatio','query_title_maxRatio_cosine','query_title_aver_cosine',
    'query_title_min_l2','query_title_minL2_pos','query_title_minL2_predictRatio','query_title_maxRatio_l2','query_title_aver_l2',
    'query_title_min_jaccard','query_title_min_jaccard_pos','query_title_minJacc_predictRatio','query_title_maxRatio_jacc']], 'left', on=['prefix','title'])
    print('query title dist time:', datetime.now() - startTime)
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
        df['%s_label_len'%alias] = df['%s_label_sum'%alias] = 0
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
    df = addPrefixIsinTitle(df)
    df = addTfidfMatrix(df)
    df = addPrefixTitleDist(df)
    df = addQueryTitleDist(df)
    return df

def extraTextFeas(df, tfidfDf):
    if 'query_title_minCosine_predictRatio' not in df.columns:
        tempDf = df.drop_duplicates(subset=['prefix','title'])
        tempDf = tempDf.merge(tfidfDf[['id','prefix_seg','query_seg','title_seg']],'left',on='id')
        tempDf['query_title_minCosine_pos'] = tempDf['query_title_cosine'].dropna().map(lambda x: x.index(min(x)))
        tempDf['query_title_minCosine_predictRatio'] = tempDf[['query_title_minCosine_pos','query_ratio']].dropna().apply(lambda x: x.query_ratio[int(x.query_title_minCosine_pos)], axis=1)

        tempDf['query_title_min_l2'] = tempDf['query_title_l2'].dropna().map(lambda x: min(x))
        tempDf['query_title_minL2_pos'] = tempDf['query_title_l2'].dropna().map(lambda x: x.index(min(x)))
        tempDf['query_title_minL2_predictRatio'] = tempDf[['query_title_minL2_pos','query_ratio']].dropna().apply(lambda x: x.query_ratio[int(x.query_title_minL2_pos)], axis=1)
        tempDf['query_title_maxRatio_l2'] = tempDf[['query_title_l2','query_predict_maxRatio_pos']].dropna().apply(lambda x: x.query_title_l2[int(x.query_predict_maxRatio_pos)], axis=1)

        tempDf['prefix_title_jaccard'] = tempDf[['prefix_seg','title_seg']].dropna().apply(lambda x: countJaccard(x.prefix_seg, x.title_seg, distance=True), axis=1)
        tempDf['query_title_jaccard'] = tempDf[['query_seg','title_seg']].dropna().apply(lambda x: [countJaccard(doc, x.title_seg, distance=True) for doc in x.query_seg], axis=1)
        tempDf['query_title_min_jaccard_pos'] = tempDf['query_title_jaccard'].dropna().map(lambda x: x.index(min(x)))
        tempDf['query_title_min_jaccard'] = tempDf['query_title_jaccard'].dropna().map(lambda x: min(x))
        tempDf['query_title_minJacc_predictRatio'] = tempDf[['query_title_min_jaccard_pos','query_ratio']].dropna().apply(lambda x: x.query_ratio[int(x.query_title_min_jaccard_pos)], axis=1)
        tempDf['query_title_maxRatio_jacc'] = tempDf[['query_title_jaccard','query_predict_maxRatio_pos']].dropna().apply(lambda x: x.query_title_jaccard[int(x.query_predict_maxRatio_pos)], axis=1)

        df = df.merge(tempDf[['prefix','title',
        'query_title_minCosine_pos','query_title_minCosine_predictRatio',
        'query_title_min_l2','query_title_minL2_pos','query_title_minL2_predictRatio','query_predict_maxRatio_pos','query_title_maxRatio_l2',
        'prefix_title_jaccard','query_title_jaccard','query_title_min_jaccard_pos','query_title_min_jaccard','query_title_minJacc_predictRatio','query_title_maxRatio_jacc']], 'left', on=['prefix','title'])
    # print(df.count())
    # exit()
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
        self.tfidfModel = None
        self.cachePath = cachePath if cachePath[-1]=="/" else (cachePath+'/')
        self.nfold = nFold
        self.seed = seed
        self.stopWords = [w.replace('\n','') for w in open("../data/user_stopwords.dat", 'r', encoding='utf-8').readlines()]
        jieba.load_userdict("../data/user_dict.dat")

    def getOriginDf(self, dfName):
        '''
        获取原始数据集，添加id列
        '''
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
            print('----------split %s text begin---------'%dfName)
            df = self.getFormatDf(dfName)
            df = addColSegList(df, self.stopWords)
            exportResult(df, filePath)
            print('----------split %s text end---------'%dfName)
        return df

    def getTfidfVecDf(self, dfName):
        '''
        数据集分词文本转tfidf向量
        '''
        filePath = self.cachePath + '%s_tfidf_%s.csv'%(self.name, dfName)
        if os.path.isfile(filePath):
            df = pd.read_csv(filePath)
            evalList = ['query_prediction','query_word','prefix_seg','title_seg','query_seg','prefix_bow','title_bow','query_bow','prefix_tfidf','title_tfidf','query_tfidf']
            df.loc[:,evalList] = df[evalList].applymap(lambda x: eval(x) if x==x else x)
        else:
            print('----------calc %s tfidf begin---------'%dfName)
            df = self.getTextSegDf(dfName)
            if self.dictionary is None:
                self.loadDictionary()
            if self.tfidfModel is None:
                self.loadTfidfModel()
            df = addColBowVector(df, self.dictionary)
            df = addTfidfVector(df, self.tfidfModel)
            exportResult(df, filePath)
            print('----------calc %s tfidf end---------'%dfName)
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
            saveDocList(docList, filePath)
            print('make %s doclist time:'%dfName, datetime.now() - startTime)
        return docList

    def loadDictionary(self):
        '''
        加载字典，若字典不存在则建立字典
        '''
        filePath = self.cachePath + '%s_dictionary.txt'%self.name
        if os.path.isfile(filePath):
            self.dictionary = Dictionary.load_from_text(filePath)
        else:
            startTime = datetime.now()
            docList = []
            for dfName in self.dfFile.keys():
                docList.extend(self.getDocList(dfName))
            self.dictionary = makeDictionary(docList, filePath)
            self.dictionary.save_as_text(filePath)
            print('make dictionary time:', datetime.now() - startTime)

    def loadTfidfModel(self):
        '''
        加载Tfidf模型，若模型不存在则建立模型
        '''
        filePath = self.cachePath + '%s_tfidf.model'%self.name
        if os.path.isfile(filePath):
            self.tfidfModel = SaveLoad.load(filePath)
        else:
            startTime = datetime.now()
            docList = []
            for dfName in self.dfFile.keys():
                docList.extend(self.getDocList(dfName))
            if self.dictionary is None:
                self.loadDictionary()
            self.tfidfModel = makeTfidfModel(docList, self.dictionary)
            self.tfidfModel.save(filePath)
            print('train tfidfModel time:', datetime.now() - startTime)

    def getOfflineDf(self):
        '''
        获取线下模型特征数据集
        '''
        if os.path.isfile(self.cachePath + '%s_offline.csv'%self.name):
            offlineDf = pd.read_csv(self.cachePath + '%s_offline.csv'%self.name)
            return offlineDf
        # 获取规范化数据集
        trainDf = self.getFormatDf('train')
        testDf = self.getFormatDf('valid')
        trainDf['flag'] = 0
        testDf['flag'] = 1
        offlineDf = pd.concat([trainDf,testDf], ignore_index=True)

        # 全局统计特征
        if os.path.isfile(self.cachePath + '%s_offline_global.csv'%self.name):
            globalDf = pd.read_csv(self.cachePath + '%s_offline_global.csv'%self.name)
        else:
            startTime = datetime.now()
            globalDf = addGlobalFeas(offlineDf, trainDf)
            exportResult(globalDf, self.cachePath + '%s_offline_global.csv'%self.name)
            print('make offline global time:', datetime.now() - startTime)
        offlineDf = offlineDf.merge(globalDf[['flag','id']+np.setdiff1d(globalDf.columns, offlineDf.columns).tolist()], 'left', on=['flag','id'])

        # 历史统计特征
        if os.path.isfile(self.cachePath + '%s_offline_his.csv'%self.name):
            hisDf = pd.read_csv(self.cachePath + '%s_offline_his.csv'%self.name)
        else:
            startTime = datetime.now()
            hisDf = addHisFeas(testDf, trainDf)
            kf = StratifiedKFold(n_splits=self.nfold, random_state=self.seed, shuffle=True)
            dfList = []
            for i, (statIdx, taskIdx) in enumerate(kf.split(trainDf.values, trainDf['label'].values)):
                tempDf = addHisFeas(trainDf.iloc[taskIdx], trainDf.iloc[statIdx])
                dfList.append(tempDf)
            hisDf = pd.concat(dfList+[hisDf], ignore_index=True)
            exportResult(hisDf, self.cachePath + '%s_offline_his.csv'%self.name)
            print('make offline his time:', datetime.now() - startTime)
        offlineDf = offlineDf.merge(hisDf[['flag','id']+np.setdiff1d(hisDf.columns, offlineDf.columns).tolist()], 'left', on=['flag','id'])

        # 文本分词特征
        dataList = [['train',0],['valid',1]]
        textDf = []
        for dfName,flag in dataList:
            filePath = self.cachePath + '%s_textFea_%s.csv'%(self.name,dfName)
            if os.path.isfile(filePath):
                tempDf = pd.read_csv(filePath)
                evalList = ['query_prediction','query_word','query_ratio','query_title_cosine','query_title_l2','query_title_jaccard']
                tempDf.loc[:,evalList] = tempDf[evalList].applymap(lambda x: eval(x) if x==x else x)
                # tempDf = extraTextFeas(tempDf)
                # exportResult(tempDf, filePath)
            else:
                print('----------make %s textFea begin----------'%dfName)
                startTime = datetime.now()
                tempDf = self.getTfidfVecDf(dfName)
                tempDf = addTextFeas(tempDf).drop(['prefix_seg','title_seg','query_seg','prefix_bow','title_bow','query_bow','prefix_tfidf','title_tfidf','query_tfidf','tfidfMatrix'],axis=1)
                exportResult(tempDf, filePath)
                print('----------make %s textFea end----------'%dfName)
                print('make %s textFea his time:'%dfName, datetime.now() - startTime)
            tempDf['flag'] = flag
            textDf.append(tempDf)
        textDf = pd.concat(textDf, ignore_index=True)
        offlineDf = offlineDf.merge(textDf[['flag','id']+np.setdiff1d(textDf.columns, offlineDf.columns).tolist()], 'left', on=['flag','id'])

        print(offlineDf.info())
        exportResult(offlineDf, self.cachePath + '%s_offline.csv'%self.name)
        print('offline dataset ready')
        return offlineDf

    def getOnlineDf(self):
        '''
        获取线上模型特征数据集
        '''
        if os.path.isfile(self.cachePath + '%s_online.csv'%self.name):
            onlineDf = pd.read_csv(self.cachePath + '%s_online.csv'%self.name)
            return onlineDf

        # 获取规范化数据集
        trainDf = self.getFormatDf('train')
        validDf = self.getFormatDf('valid')
        testDf = self.getFormatDf('testA')
        trainDf['flag'] = 0
        validDf['flag'] = 1
        testDf['flag'] = -1
        statDf = pd.concat([trainDf, validDf], ignore_index=True)
        onlineDf = pd.concat([trainDf,validDf,testDf], ignore_index=True)

        # 全局统计特征
        if os.path.isfile(self.cachePath + '%s_online_global.csv'%self.name):
            globalDf = pd.read_csv(self.cachePath + '%s_online_global.csv'%self.name)
        else:
            startTime = datetime.now()
            globalDf = addGlobalFeas(onlineDf, statDf)
            exportResult(globalDf, self.cachePath + '%s_online_global.csv'%self.name)
            print('make online global time:', datetime.now() - startTime)
        onlineDf = onlineDf.merge(globalDf[['flag','id']+np.setdiff1d(globalDf.columns, onlineDf.columns).tolist()], 'left', on=['flag','id'])

        # 历史统计特征
        if os.path.isfile(self.cachePath + '%s_online_his.csv'%self.name):
            hisDf = pd.read_csv(self.cachePath + '%s_online_his.csv'%self.name)
        else:
            startTime = datetime.now()
            hisDf = addHisFeas(testDf, statDf)
            kf = StratifiedKFold(n_splits=self.nfold, random_state=self.seed, shuffle=True)
            dfList = []
            for i, (statIdx, taskIdx) in enumerate(kf.split(statDf.values, statDf['label'].values)):
                tempDf = addHisFeas(statDf.iloc[taskIdx], statDf.iloc[statIdx])
                dfList.append(tempDf)
            hisDf = pd.concat(dfList+[hisDf], ignore_index=True)
            exportResult(hisDf, self.cachePath + '%s_online_his.csv'%self.name)
            print('make online his time:', datetime.now() - startTime)
        onlineDf = onlineDf.merge(hisDf[['flag','id']+np.setdiff1d(hisDf.columns, onlineDf.columns).tolist()], 'left', on=['flag','id'])

        # 文本分词特征
        dataList = [['train',0],['valid',1],['testA',-1]]
        textDf = []
        for dfName,flag in dataList:
            filePath = self.cachePath + '%s_textFea_%s.csv'%(self.name,dfName)
            if os.path.isfile(filePath):
                tempDf = pd.read_csv(filePath)
                evalList = ['query_prediction','query_word','query_ratio','query_title_cosine','query_title_l2','query_title_jaccard']
                tempDf.loc[:,evalList] = tempDf[evalList].applymap(lambda x: eval(x) if x==x else x)
                # tempDf = extraTextFeas(tempDf)
                # exportResult(tempDf, filePath)
            else:
                print('----------make %s textFea begin----------'%dfName)
                startTime = datetime.now()
                tempDf = self.getTfidfVecDf(dfName)
                tempDf = addTextFeas(tempDf).drop(['prefix_seg','title_seg','query_seg','prefix_bow','title_bow','query_bow','prefix_tfidf','title_tfidf','query_tfidf','tfidfMatrix'],axis=1)
                exportResult(tempDf, filePath)
                print('----------make %s textFea end----------'%dfName)
                print('make %s textFea his time:'%dfName, datetime.now() - startTime)
            tempDf['flag'] = flag
            textDf.append(tempDf)
        textDf = pd.concat(textDf, ignore_index=True)
        onlineDf = onlineDf.merge(textDf[['flag','id']+np.setdiff1d(textDf.columns, onlineDf.columns).tolist()], 'left', on=['flag','id'])

        print(onlineDf.info())
        exportResult(onlineDf, self.cachePath + '%s_online.csv'%self.name)
        print('online dataset ready')
        return onlineDf

if __name__ == '__main__':
    startTime = datetime.now()
    ORIGIN_DATA_PATH = "../data/"
    dfFile = {
        'train': ORIGIN_DATA_PATH + "oppo_round1_train_20180929.txt",
        'valid': ORIGIN_DATA_PATH + "oppo_round1_vali_20180929.txt",
        'testA':ORIGIN_DATA_PATH + "oppo_round1_test_A_20180929.txt",
    }
    factory = FeaFactory(dfFile, name="fea_new", cachePath="../temp/")
    df = factory.getOfflineDf()
    df2 = factory.getOnlineDf()
    print('feaFactory time:', datetime.now() - startTime)
