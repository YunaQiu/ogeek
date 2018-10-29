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
from gensim.models import TfidfModel

from utils import *

class NLP:
    def __init__(self, trainDocs=[], testDocs=[]):
        self.trainDocs = trainDocs
        self.testDocs = testDocs

    def saveDocList(self, docList, path):
        with open(path,'w',encoding='utf-8') as fp:
            for docs in docList:
                fp.write(" ".join(docs) + "\n")

def saveDocList(docList, filepath):
    '''
    保存文档分词
    '''
    with open(filepath,'w',encoding='utf-8') as fp:
        for docs in docList:
            fp.write(" ".join(docs) + "\n")

def makeDictionary(docList, dictFile, add=False):
    '''
    生成词典
    '''
    startTime = datetime.now()
    if os.path.isfile(dictFile) and type=='a':
        dictionary = Dictionary.load_from_text(dictFile)
        dictionary.add_documents(docList)
    else:
        dictionary = Dictionary(docList)
    dictionary.save_as_text(dictFile)
    print('make dictionary time:', datetime.now() - startTime)
    return dictionary

def getDfDoc(df, stopWordList=[]):
    '''
    对给定数据集提取文本片段并分词
    '''
    docList = []
    df['title'].dropna().map(lambda x: docList.append(x))
    df['query_prediction'].map(lambda x: docList.extend(list(x.keys())))
    print('doc count:',len(docList))
    startTime = datetime.now()
    docList = strList2SegList(docList, stopWordList=stopWordList)
    print('word to segList time:', datetime.now() - startTime)
    return docList

def dfDocRerun(docName="docList_search"):
    jieba.load_userdict("../data/user_dict.dat")
    stopWordList = [w.replace('\n','') for w in open("../data/user_stopwords.dat", 'r', encoding='utf-8').readlines()]
    ORIGIN_DATA_PATH = "../data/"
    EXPORT_PATH = "../temp/doc/"

    # trainDf = importDf(ORIGIN_DATA_PATH + "oppo_round1_train_20180929.txt", colNames=['prefix','query_prediction','title','tag','label'])
    validDf = importDf(ORIGIN_DATA_PATH + "oppo_round1_vali_20180929.txt", colNames=['prefix','query_prediction','title','tag','label'])
    # testADf = importDf(ORIGIN_DATA_PATH + "oppo_round1_test_A_20180929.txt", colNames=['prefix','query_prediction','title','tag'])

    docFileList = [
        # [trainDf, 'train'],
        [validDf, 'valid'],
        # [testADf, 'testA']
    ]
    docName = "docList_search"
    # for dataDf,name in docFileList:
    validDf['query_prediction'] = validDf['query_prediction'].map(lambda x: eval(x))
    docList = getDfDoc(validDf, stopWordList)
    saveDocList(docList, EXPORT_PATH + "%s_%s.txt"%(docName,'valid'))
    # print('save segList time:', datetime.now() - startTime)
    return docList

if __name__ == '__main__':
    ORIGIN_DATA_PATH = "../data/"
    DOC_PATH = "../temp/doc/"
    DOC_NAME = "docList_search"

    if not os.path.isfile(DOC_PATH+DOC_NAME+"_valid.txt"):
        docList = dfDocRerun(DOC_NAME)
    else:
        with open(DOC_PATH+DOC_NAME+"_valid.txt", encoding='utf-8') as fp:
            docList = [line.replace('\n','').split(" ") for line in fp.readlines()]

    dictionary = makeDictionary(docList, DOC_PATH+"dictionary.txt")
    print('prepare docList: ready!')

    startTime = datetime.now()
    bowList = [dictionary.doc2bow(doc) for doc in docList]
    # print(bowList[:5])
    print('doc2bow time:', datetime.now() - startTime)

    startTime = datetime.now()
    tfidfModel = TfidfModel(bowList)
    print('tfidf time:', datetime.now() - startTime)
    # tfidfModel.save("../model/nlp/offlineTfidf.model")
    print(tfidfModel[[(0,1),(1,1),(2,1)]])
    result = [tfidfModel[bow] for bow in bowList[:5]]
    print(result)
