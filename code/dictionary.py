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
import json, random, os, math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
import jieba
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

# 导入数据
def importDf(url, sep='\t', na_values=None, header=None, index_col=None, colNames=None, **params):
    df = pd.read_table(url, names=colNames, header=header, na_values='', keep_default_na=False, encoding='utf-8', quoting=3, **params)
    return df

class Dataset:
    def __init__(self, dfList):
        self.dfList = dfList
        self.sentenceList = None
        self.segList = None

    def getSentenceList(self):
        '''
        提取句子列表
        '''
        if self.sentenceList is not None:
            return self.sentenceList
        sentenceList = []
        for df in self.dfList:
            sentenceList.extend(df['prefix'].drop_duplicates().dropna().values)
            sentenceList.extend(df['title'].drop_duplicates().dropna().values)
            df['query_prediction'].drop_duplicates().dropna().map(
                lambda x: sentenceList.extend(list(eval(x).keys()))
            )
        self.sentenceList = sentenceList
        return sentenceList

    def getSegList(self, stopWords=[]):
        '''
        提取分词后的句子列表
        '''
        if self.segList is not None:
            return self.segList
        segList = []
        sentenceList = self.getSentenceList()
        for sent in sentenceList:
            seg = [word for word in jieba.cut(sent.lower()) if (word not in stopWords)]
            if len(seg)>0:
                segList.append(seg)
        print(segList[-5:])
        self.segList = segList
        return segList


def main():
    ORIGIN_DATA_PATH = "../data/"#oppo_data_ronud2_20181107/
    STOPWORD_PATH = "../data/stop_words.txt"
    USER_WORDS_PATH = "./user_dict.dat"
    Dictionary_PATH = "./dictionary.txt"

    filePath = {
        'train': ORIGIN_DATA_PATH + "oppo_round1_train_20180929.txt",
        # 'train': ORIGIN_DATA_PATH + "data_train.txt",
        # 'valid': ORIGIN_DATA_PATH + "data_vali.txt",
        # 'testA': ORIGIN_DATA_PATH + "data_test.txt",
    }
    dfList = [
        importDf(x, colNames=['prefix','query_prediction','title','tag','label'], nrows=10000)
        for x in filePath.values()
    ]
    data = Dataset(dfList)

    startTime = datetime.now()
    stopWords = []
    # stopWords = [w.replace('\n','') for w in open(STOPWORD_PATH, 'r', encoding='GBK').readlines()]
    # jieba.load_userdict(USER_WORDS_PATH)
    segList = data.getSegList(stopWords)
    print('seg time:', datetime.now() - startTime)

    startTime = datetime.now()
    dct = Dictionary(segList)
    # dct.save_as_text(Dictionary_PATH)
    print('dictionary time:', datetime.now() - startTime)

    print('语料库文档数', dct.num_docs)
    print('文档频数：\"刷机\"：', dct.dfs[dct.token2id['刷机']])
    print('逆文档频率：\"刷机\"：', math.log2(dct.num_docs / dct.dfs[dct.token2id['刷机']]))
    print('判断是否在词典中: \"我是说的\"', '我是说的' in dct.token2id)

if __name__ == '__main__':
    main()
