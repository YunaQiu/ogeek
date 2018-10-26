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
import json, random, re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectPercentile

from utils import *

def formatQuery(df):
    '''
    格式化预测词字段
    '''
    startTime = datetime.now()
    def format(x):
        x = eval(x)
        x = {k:float(v) for k,v in x.items()}
        return x
    df['query_prediction'] = df['query_prediction'].map(lambda x: format(x))
    print('format query time:', datetime.now() - startTime)
    return df

def addQueryFea(df):
    '''
    提取搜索词字段特征
    '''
    startTime = datetime.now()
    tempDf = df[['prefix','query_prediction']].drop_duplicates(subset='prefix')
    tempDf['query_predict_num'] = tempDf['query_prediction'].map(lambda x: len(x))
    tempDf['query_predict_maxRatio'] = tempDf[tempDf.query_predict_num>0]['query_prediction'].map(lambda x: max(list(x.values())))
    tempDf['query_predict_max'] = tempDf[tempDf.query_predict_num>0]['query_prediction'].map(lambda x: list(x.keys())[list(x.values()).index(max(list(x.values())))])
    df = df.merge(tempDf.drop(['query_prediction'], axis=1), how='left', on=['prefix'])
    print('query fea time:', datetime.now() - startTime)
    return df

def addCrossColNunique(df, statDf, col1, col2, alias=None):
    '''
    统计每个col1中col2的种类数
    '''
    if alias is None:
        alias = '%s_%s' % (col1, col2)
    if '%s_nunique'%alias in statDf.columns:
        print('col exist')
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
        df['%s_label_len'%alias] = df['%s_label_sum'%alias] = df['%s_label_ratio'%alias] = np.nan
        return df
    if '%s_label_ratio'%alias in statDf.columns:
        print('col exist')
        tempDf = statDf[colArr + ['%s_label_len'%alias, '%s_label_sum'%alias, '%s_label_ratio'%alias]].drop_duplicates(subset=colArr)
        df = df.merge(tempDf, how='left', on=colArr)
    else:
        tempDf = statDf.groupby(colArr)['label'].agg([len, 'sum'])
        tempDf['ratio'] = biasSmooth(tempDf['sum'], tempDf['len'])
        # tempDf.loc[:,['len','sum']] /= tempDf[['len','sum']].sum()
        tempDf.columns = ['%s_label_%s'%(alias,x) for x in tempDf.columns]
        df = df.merge(tempDf.reset_index(), 'left', on=colArr)
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
    df = addNewValFea(df, statDf, ['prefix','title'])
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
    print('add global fea time:', datetime.now() - startTime)
    return df

if __name__ == '__main__':
    startTime = datetime.now()
    df = importDf('../data/oppo_round1_train_20180929.txt', colNames=['prefix','query_prediction','title','tag','label'])
    df = df.reset_index()
    df['flag'] = 0
    validDf = importDf('../data/oppo_round1_vali_20180929.txt', colNames=['prefix','query_prediction','title','tag','label'])
    validDf = validDf.reset_index()
    validDf['flag'] = 1
    testADf = importDf('../data/oppo_round1_test_A_20180929.txt', colNames=['prefix','query_prediction','title','tag'])
    testADf = testADf.reset_index()
    testADf['flag'] = -1
    originDf = pd.concat([df,validDf,testADf], ignore_index=True)

    originDf = addGlobalFeas(originDf, df)
    originDf = addHisFeas(originDf, originDf)
    print(originDf.info())
    print('feaFactory time:', datetime.now() - startTime)
