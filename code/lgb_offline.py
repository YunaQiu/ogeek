# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import f1_score, log_loss
import scipy as sp
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import jieba
from Levenshtein import distance as lev_distance
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors, Word2Vec
from datetime import *
from time import time
from multiprocessing import Pool
import gc, os, math
import warnings

warnings.filterwarnings("ignore")

def importDf(url, sep='\t', na_values=None, header=None, index_col=None, colNames=None, **params):
    df = pd.read_table(url, names=colNames, header=header, na_values='', keep_default_na=False, encoding='utf-8', quoting=3, **params)
    return df

def importCacheDf(url):
    df = df = pd.read_csv(url, na_values='', keep_default_na=False, float_precision="%.10f")
    return df

def one_zero2(data,thre):
    if data<thre:
        return 0
    else:
        return 1

#统计prefix在title中的位置
def get_prefix_position(df):
    title = df['title']
    prefix = df['prefix']
    return str(title).find(str(prefix))

def get_tag_dict(raw):
    label_encoder = LabelEncoder()
    label_encoder.fit(raw['tag'])
    return label_encoder

def read_w2v_model(model_dir,persist=True):
    if persist:
        w2v_model = Word2Vec.load(model_dir).wv
    else:
        w2v_model = KeyedVectors.load_word2vec_format(model_dir)
    return w2v_model

def read_stop_word(stop_word_dir):
    filename = stop_word_dir
    with open(filename,encoding='GBK') as file:
        stop_words = file.read()
    stop_words = stop_words.split('\n')
    return stop_words

def str_lower(sample):
    sample['prefix'] = sample['prefix'].astype(str)
    sample['title'] = sample['title'].astype(str)
    sample['prefix'] = list(map(str.lower,sample['prefix']))
    sample['title'] = list(map(str.lower,sample['title']))
    gc.collect()
    return sample

def fillnaQuery(sample, na='{}'):
    tempDf = sample[sample.query_prediction!=na].drop_duplicates(['prefix']).set_index('prefix')['query_prediction']
    sample.loc[sample.query_prediction==na, 'query_prediction'] = sample[sample.query_prediction==na]['prefix'].map(lambda x: tempDf[x] if x in tempDf.index else na)
    return sample

def queryNum(sample):
    tempDf = sample[['query_prediction']].drop_duplicates()
    tempDf['query_num'] = tempDf['query_prediction'].map(lambda x: len(eval(x)))
    sample = sample.merge(tempDf, 'left', on=['query_prediction'])
    return sample

def timeFeas(sample):
    sample.sort_values(['prefix','instance_id'], inplace=True)
    sample['temp'] = sample['prefix'].shift(1)
    sample['pre_prefix'] = sample['instance_id'].shift(1)
    sample['pre_prefix'] = sample['instance_id'] - sample['pre_prefix']
    sample.loc[sample.prefix!=sample.temp, 'pre_prefix'] = -1
    sample['temp'] = sample['prefix'].shift(-1)
    sample['next_prefix'] = sample['instance_id'].shift(-1)
    sample['next_prefix'] = sample['next_prefix'] - sample['instance_id']
    sample.loc[sample.prefix!=sample.temp, 'next_prefix'] = -1

    sample.sort_values(['title','instance_id'], inplace=True)
    sample['temp'] = sample['title'].shift(1)
    sample['pre_title'] = sample['instance_id'].shift(1)
    sample['pre_title'] = sample['instance_id'] - sample['pre_title']
    sample.loc[sample.title!=sample.temp, 'pre_title'] = -1
    sample['temp'] = sample['title'].shift(-1)
    sample['next_title'] = sample['instance_id'].shift(-1)
    sample['next_title'] = sample['next_title'] - sample['instance_id']
    sample.loc[sample.title!=sample.temp, 'next_title'] = -1

    sample.drop(['temp'], axis=1, inplace=True)
    sample.sort_values(['instance_id'], inplace=True)
    return sample

def get_ctr(raw, sample, stat_list):
    rate_stat = raw[stat_list+['label']].groupby(stat_list).mean().reset_index()
    rate_stat = rate_stat.rename(columns={'label':'_'.join(stat_list)+'_ctr'})
    sample = pd.merge(sample, rate_stat, on=stat_list, how='left')

    count_stat = raw[stat_list+['label']].groupby(stat_list).count().reset_index()
    count_stat = count_stat.rename(columns={'label':'_'.join(stat_list)+'_count'})
    sample = pd.merge(sample, count_stat, on=stat_list, how='left').fillna(0)
    gc.collect()

    click_stat = raw[stat_list+['label']].groupby(stat_list).sum().reset_index()
    click_stat = click_stat.rename(columns={'label':'_'.join(stat_list)+'_click'})
    sample = pd.merge(sample, click_stat, on=stat_list, how='left').fillna(0)
    gc.collect()

    return sample

def ctr_features(raw, sample):
    stat_ls = [['prefix'],
               ['title'],
               ['tag'],
               # ['prefix_position'],
               ['prefix','title'],
               ['prefix','tag'],
               ['title','tag'],
               ['query_num','tag'],
               # ['prefix', 'prefix_position'],
               # ['title', 'prefix_position'],
               # ['tag', 'prefix_position'],
               ['prefix','title','tag']]
    for l in stat_ls:
        sample = get_ctr(raw, sample, l)
        gc.collect()
    return sample

def get_nunique(raw, sample, c1, c2):
    n_stat = raw[[c1, c2]].drop_duplicates()
    n_stat = n_stat.groupby(c1).count().reset_index()
    n_stat.columns = [c1, c1 + '_' + c2 + '_nunique']
    sample = pd.merge(sample, n_stat, on=c1, how='left').fillna(0)
    return sample

def lake_features(raw, sample):
    c1_list = ['prefix', 'title', 'prefix', 'title']
    c2_list = ['title', 'prefix', 'tag', 'tag']
    for c1, c2 in zip(c1_list,c2_list):
        sample = get_nunique(raw, sample, c1, c2)
        gc.collect()
    return sample

def get_ratio_in_col(raw, sample, col1, col2):
    crossColDf = raw.reset_index().groupby([col1,col2])['index'].agg(len).to_frame('count')
    col2Df = crossColDf.groupby(col2)['count'].sum().to_frame('sum')
    crossColDf = crossColDf.merge(col2Df, 'left', left_on=col2, right_index=True)
    crossColDf['ratio_%s_in_%s'%(col1, col2)] = crossColDf['count'] / crossColDf['sum']
    sample = sample.merge(crossColDf[['ratio_%s_in_%s'%(col1, col2)]], 'left', left_on=[col1,col2], right_index=True)
    return sample

def ratio_col_features(raw, sample):
    stat_ls = [['tag','prefix'],
               ['prefix','tag'],
               ['prefix','title'],
               ['title','prefix'],
               ['title','tag'],
               ['tag','title']]
    for col1, col2 in stat_ls:
        sample = get_ratio_in_col(raw, sample, col1, col2)
        gc.collect()
    return sample

def stat_features(raw, sample):
    sample = str_lower(sample)
    sample = lake_features(raw, sample)
    sample = ctr_features(raw, sample)
    sample = ratio_col_features(raw, sample)
    return sample

def k_fold_stat_features(data, k=5):
    print('-- get 5 fold stat features')
    kf = KFold(n_splits=k)
    samples = []
    for raw_idx, sample_idx in kf.split(data.index):
        raw = data[data.index.isin(raw_idx)].reset_index(drop=True)
        sample = data[data.index.isin(sample_idx)].reset_index(drop=True)
        sample = stat_features(raw, sample)
        samples.append(sample)
        gc.collect()
    samples = pd.concat(samples,ignore_index=True)
    #samples = samples.reset_index(drop=True)
    return samples

def k_fold_stat_features2(data, k=5, newRatio=0.4, random_state=0):
    '''
    构建含一定新数据比例的统计数据集
    '''
    print('-- get 5 fold stat features. plan 2')
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    dfList = []
    for i, (statIdx, taskIdx) in enumerate(kf.split(data.values, data['label'].values, groups=data['prefix'].values)):
        tempDf = stat_features(data.iloc[statIdx].reset_index(drop=True), data.iloc[taskIdx].reset_index(drop=True))
        tempDf.drop(tempDf[(tempDf['prefix_count']==0)|(tempDf['title_count']==0)].index, inplace=True)
        dfList.append(tempDf)
    oldDf = pd.concat(dfList, ignore_index=True)

    kf = GroupKFold(n_splits=k)
    dfList = []
    for i, (statIdx, taskIdx) in enumerate(kf.split(data.values, data['label'].values, groups=data['prefix'].values)):
        tempDf = stat_features(data.iloc[statIdx].reset_index(drop=True), data.iloc[taskIdx].reset_index(drop=True))
        dfList.append(tempDf)
    newDf = pd.concat(dfList, ignore_index=True)

    newSampleNum = int(len(oldDf)/(1-newRatio)*newRatio)
    data = pd.concat([oldDf, newDf.sample(n=newSampleNum, random_state=random_state)], ignore_index=True)
    return data

def map_to_array(func,data1,data2=None,paral=False):
    if paral==False:
        if data2 is not None:
            data = list(map(func,data1,data2))
        else:
            data = list(map(func,data1))

    else:
        if data2 is not None:
            with Pool(processes=2) as pool:
                data = pool.map(func,zip(data1,data2))
        else:
            with Pool(processes=2) as pool:
                data = pool.map(func,data1)

    data = np.array(data)
    return data

# =========================================
def addQueryCache(data, cache):
    print('------ split query and weight', end='')
    start = time()

    tempSeries = data.drop_duplicates()
    querys = []
    weights = []
    for q in tempSeries:
        query_prediction = eval(q)
        query = [key.lower() for key in sorted(query_prediction)][:11]
        weight = [float(query_prediction[key]) for key in sorted(query_prediction)][:11]
        querys.append(query)
        weights.append(weight)
    querys = pd.DataFrame(querys,columns=list(range(11)), index=tempSeries.values).fillna('')
    weights = pd.DataFrame(weights,columns=list(range(11)), index=tempSeries.values).fillna(0)
    norm_weights = weights.div(weights.sum(axis=1)+0.001, axis=0)
    weight_argmax = weights.idxmax(axis=1)
    cache['querys'] = querys.loc[data.values].values
    cache['weights'] = weights.loc[data.values].values
    cache['norm_weights'] = norm_weights.loc[data.values].values
    cache['weight_argmax'] = weight_argmax.loc[data.values].values
    cache['idx'] = tuple(range(cache['weight_argmax'].shape[0]))
    cache['max_w_query'] = cache['querys'][cache['idx'], cache['weight_argmax']]
    print('get query weight:', data.shape, querys.shape, weights.shape, norm_weights.shape)

    del querys, weights, norm_weights, weight_argmax
    print('   cost: %.1f ' %(time()-start))
    return cache

def min_max_mean_std(sample,data,name,func_name, **params):
    data_nw = data * params['norm_weights']
    data_w = data * params['weights']
    sample[name+'_min_'+func_name] = np.min(data,1)
    sample[name+'_max_'+func_name] = np.max(data,1)
    sample[name+'_mean_'+func_name] = np.sum(data_w, 1)
    sample[name+'_norm_mean_'+func_name] = np.sum(data_nw,1)
    sample[name+'_std_'+func_name] = np.sum(np.power(data      \
                                            - np.array(sample[name+'_norm_mean_'+func_name]).reshape(-1,1),2)*params['norm_weights'],1)
    del data_nw,data_w
    return sample

def weight_features(sample, **cache):
    print('------ weight features',end='')
    print('weight fea:',sample.shape, cache['weights'].shape)
    start = time()

    sample['weight_sum'] = np.sum(cache['weights'],1)
    sample = min_max_mean_std(sample,cache['weights'],'weight','', **cache)
    sample['weight_mean'] = np.mean(cache['weights'],1)
    sample['weight_std'] = np.std(cache['weights'],1)

    print('   cost: %.1f ' %(time()-start))
    return sample

def len_features(sample, **cache):
    print('------ len features',end = '')
    start = time()
    def get_query_len(query):
        q_lens = [len(q) for q in query]
        return q_lens

    querys_lens = map_to_array(get_query_len,cache['querys'])
    sample = min_max_mean_std(sample,querys_lens, 'query','len', **cache)

    sample['prefix_len'] = list(map(len,sample['prefix']))
    sample['title_len'] = list(map(len,sample['title']))

    max_w_query_len = querys_lens[cache['idx'],cache['weight_argmax']]

    sample['mx_w_prfx_qry_len_sub'] = max_w_query_len-sample['prefix_len']

    sample['title_prefix_len_sub'] = sample['title_len']-sample['prefix_len']
    sample['query_title_len_sub'] = sample['query_mean_len'] - sample['title_len']
    sample['query_prefix_len_sub'] = sample['query_mean_len'] - sample['prefix_len']
    sample['prefix_query_len_div'] = sample['prefix_len'].div(sample['query_mean_len'])
    sample['prefix_title_len_div'] = sample['prefix_len'].div(sample['title_len'])

    del max_w_query_len, querys_lens
    print('   cost: %.1f ' %(time()-start))
    return sample

def lev_features(sample, **cache):
    print('------ lev features', end='')
    start = time()
    def get_lev_dist_list(query,data):
        query_data_levs = [lev_distance(q,data) for q in query]
        return query_data_levs

    query_title_levs = map_to_array(get_lev_dist_list,cache['querys'],sample['title'])
    sample = min_max_mean_std(sample,query_title_levs,'query_title','lev', **cache)
    query_argmx = query_title_levs.argmin(axis=1)
    sample['query_title_min_lev_weight'] = cache['weights'][cache['idx'], query_argmx]
    sample['mx_w_query_title_lev'] = query_title_levs[cache['idx'],cache['weight_argmax']]
    sample['prefix_title_lev'] = list(map(lev_distance,sample['prefix'],sample['title']))

    sample['mx_w_prefix_query_lev'] = list(map(lev_distance,sample['prefix'],cache['max_w_query']))

    levs = pd.DataFrame(np.sort(query_title_levs, axis=1),columns=['lev_'+str(i) for i in range(11)], index=sample.index)
    sample = pd.concat([sample,levs],axis=1)

    del query_title_levs, query_argmx, levs
    print('   cost: %.1f ' %(time()-start))
    return sample

def jaccard_features(sample, **cache):
    print('------ jaccard features',end='')
    start = time()
    def jaccard(s1,s2):
        inter=len([w for w in s1 if w in s2])
        union = len(s1)+len(s2)-inter
        return inter/(union+0.001)

    def jaccard_dist(querys,data):
        res = [jaccard(q,data) for q in querys]
        return res

    print('jaccard->querys:', cache['querys'].shape)
    querys_title_jac = map_to_array(jaccard_dist,cache['querys'],sample['title'])
    sample = min_max_mean_std(sample,querys_title_jac,'query_title','jac', **cache)
    query_argmax = querys_title_jac.argmax(axis=1)
    sample['query_title_max_jac_weight'] = cache['weights'][cache['idx'], query_argmax]
    sample['mx_w_query_title_jac'] = querys_title_jac[cache['idx'],cache['weight_argmax']]
    sample['prefix_title_jac'] = list(map(jaccard,sample['prefix'],sample['title']))

    jacs = pd.DataFrame(-np.sort(-querys_title_jac,axis=1),columns=['query_title_jac_'+str(i) for i in range(11)], index=sample.index)
    sample = pd.concat([sample,jacs],axis=1)
    sample = sample.fillna(0)

    del querys_title_jac,query_argmax,jacs
    print('   cost: %.1f ' %(time()-start))
    return sample

def get_word_seg(sentence, stop_words, **params):
    sentence = sentence.replace('%2C', ',')
    word_seg = [word for word in jieba.cut(sentence) if (word not in stop_words)]
    return word_seg

def get_sentence_vec(word_seg, w2v_model, **params):
    s_vector = np.zeros((len(w2v_model['我'])))
    if len(word_seg) > 0:
        count=0
        # dct = params['dictionary']
        for word in word_seg :
            try:
                # idf = math.log2(dct.num_docs / dct.dfs[dct.token2id[word]]) if word in dct.token2id else 1
                vec = w2v_model[word]# * idf
                s_vector += vec
                count += 1
            except (KeyError):
                pass
        if count:
            s_vector /= count
    return s_vector

def addSegCahce(sample, cache):
    print('------ get sentence seg',end='')
    start = time()

    tempDf = sample['title'].drop_duplicates()
    temp_segs = np.array(list(map(lambda x: get_word_seg(x, **cache), tempDf)))
    tempDf = pd.Series(temp_segs, index=tempDf)
    cache['title_seg'] = tempDf.loc[sample.title].values

    tempDf = sample['prefix'].drop_duplicates()
    temp_segs = np.array(list(map(lambda x: get_word_seg(x, **cache), tempDf)))
    tempDf = pd.Series(temp_segs, index=tempDf)
    cache['prefix_seg'] = tempDf.loc[sample.prefix].values

#     tempDf = sample['query_prediction'].drop_duplicates()
#     querys = cache['querys'][tempDf.index]
#     temp_segs = np.array(list(map(lambda x: [get_word_seg(q, **cache) for q in x], querys)))
#     tempDf = pd.DataFrame(temp_segs, index=tempDf)
#     cache['query_seg'] = tempDf.loc[sample.query_prediction].values
#     cache['mx_w_query_seg'] = cache['query_seg'][cache['idx'], cache['weight_argmax']]

    tempDf = sample['query_prediction'].drop_duplicates()
    querys = cache['max_w_query'][tempDf.index]
    temp_segs = np.array(list(map(lambda x: get_word_seg(x, **cache), querys)))
    tempDf = pd.Series(temp_segs, index=tempDf)
    cache['mx_w_query_seg'] = tempDf.loc[sample.query_prediction].values

    del temp_segs,tempDf,querys
    print('   cost: %.1f ' %(time()-start))
    return cache

def addEmbedCache(sample, cache):
    print('------ get w2v embed 1',end='')
    start = time()
    cache['title_embed'] = np.array(list(map(lambda x: get_sentence_vec(x, **cache), cache['title_seg'])))
    cache['prefix_embed'] = np.array(list(map(lambda x: get_sentence_vec(x, **cache), cache['prefix_seg'])))
    cache['mx_w_query_embed'] = np.array(list(map(lambda x: get_sentence_vec(x, **cache), cache['mx_w_query_seg'])))
    print('   cost: %.1f ' %(time()-start))
    return cache

def cosine(v1,v2):
    if len(v1.shape)==1:
        multi = np.dot(v1,v2)
        axis=None
    else:
        multi = np.sum(v1*v2,1)
        axis=1
    s1_norm = np.linalg.norm(v1,axis=axis)
    s2_norm = np.linalg.norm(v2,axis=axis)
    cos = multi/(s1_norm*s2_norm+0.001)
    return cos

def calcWmSimilar(strList1, strList2, wvModel):
    '''
    计算文档的词移距离
    '''
    dist = wvModel.wmdistance(strList1, strList2)
    if dist==np.inf:
        return np.nan
    else:
        return dist

def w2v_features(sample, **cache):
    print('------ w2v features 1',end='')
    start = time()
    title_embed = pd.DataFrame(cache['title_embed'],columns=['title_w2v_'+str(i) for i in range(50)], index=sample.index)
    sample = pd.concat([sample,title_embed],axis=1)

    prefix_embed = pd.DataFrame(cache['prefix_embed'],columns=['prefix_w2v_'+str(i) for i in range(50)], index=sample.index)
    sample = pd.concat([sample,prefix_embed],axis=1)

    mx_w_query_embed = pd.DataFrame(cache['mx_w_query_embed'],columns=['mx_w_query_w2v_'+str(i) for i in range(50)], index=sample.index)
    sample = pd.concat([sample,mx_w_query_embed],axis=1)
    print('   cost: %.1f ' %(time()-start))
    return sample

def cos_feature(sample, alias='0', **cache):
    print('------ cos features 1',end='')
    start = time()

    sample['prefix_title_cos_%s'%alias] = cosine(cache['title_embed'],cache['prefix_embed'])
    sample['prefix_mx_query_cos_%s'%alias] = cosine(cache['prefix_embed'],cache['mx_w_query_embed'])
    sample['mx_w_query_title_cos_%s'%alias] = cosine(cache['title_embed'],cache['mx_w_query_embed'])

    print('   cost: %.1f ' %(time()-start))
    return sample

def wm_feature(sample, alias='0', **cache):
    print('------ word_move distance features 1',end='')
    start = time()

    sample['prefix_title_wm_%s'%alias] = list(map(lambda x,y: calcWmSimilar(x,y,cache['w2v_model']), cache['title_seg'],cache['prefix_seg']))
    sample['prefix_mx_query_wm_%s'%alias] = list(map(lambda x,y: calcWmSimilar(x,y,cache['w2v_model']), cache['prefix_seg'],cache['mx_w_query_seg']))
    sample['mx_w_query_title_wm_%s'%alias] = list(map(lambda x,y: calcWmSimilar(x,y,cache['w2v_model']), cache['mx_w_query_seg'],cache['title_seg']))

    print('   cost: %.1f ' %(time()-start))
    return sample

def text_features(sample, w2v_model_1, ext_vec_dirs, **cache):
    tempDf = sample.drop_duplicates(['prefix','query_prediction','title'])
    tempDf.index = list(range(tempDf.shape[0]))
    cache = addQueryCache(tempDf['query_prediction'], cache)
    gc.collect()
    tempDf = weight_features(tempDf, **cache)
    gc.collect()
    tempDf = len_features(tempDf, **cache)
    gc.collect()
    tempDf = lev_features(tempDf, **cache)
    gc.collect()
    tempDf = jaccard_features(tempDf, **cache)
    gc.collect()

    cache['w2v_model'] = w2v_model_1
    cache = addSegCahce(tempDf, cache)
    gc.collect()
    cache = addEmbedCache(tempDf, cache)
    gc.collect()
    tempDf = w2v_features(tempDf, **cache)
    gc.collect()
    tempDf = cos_feature(tempDf, **cache)
    gc.collect()
    tempDf = wm_feature(tempDf, **cache)
    del cache['w2v_model']
    gc.collect()

    for i,dir in enumerate(ext_vec_dirs):
        cache['w2v_model'] = read_w2v_model(dir, persist=False)
        cache = addEmbedCache(tempDf, cache)
        gc.collect()
        tempDf = cos_feature(tempDf, alias=str(i+1), **cache)
        gc.collect()
        tempDf = wm_feature(tempDf, alias=str(i+1), **cache)
        del cache['w2v_model']
        gc.collect()

    del cache
    sample = sample.merge(
        tempDf[['prefix','query_prediction','title'] + np.setdiff1d(tempDf.columns,sample.columns).tolist()],
        how='left',
        on=['prefix','query_prediction','title'])
    return sample

def getPredLabel(predArr, threshold=None, tops=None):
    '''
    根据阈值返回分类预测结果
    '''
    if tops is not None :
        temp = np.sort(np.array(predArr))
        if tops < 1:
            threshold = temp[-1*round(len(temp)*tops)]
        else:
            threshold = temp[-round(tops)]
    if threshold is None:
        print('[Error] could not get threshold value.')
        exit()
    return (predArr>=threshold).astype(int)

def findF1Threshold(predictList, labelList, thrList=None):
    '''
    寻找F1最佳阈值
    '''
    tempDf = pd.DataFrame({'predict':predictList, 'label':labelList})
    trueNum = len(tempDf[tempDf.label==1])
    if thrList is None:
        thrList = np.unique(tempDf['predict'])
    f1List = []
    for thr in thrList:
        tempDf['temp'] = getPredLabel(tempDf['predict'], thr)
        TP = len(tempDf[(tempDf.label==1)&(tempDf.temp==1)])
        if TP==0:
            break
        positiveNum = len(tempDf[tempDf.temp==1])
        precise = TP / positiveNum
        recall = TP / trueNum
        f1 = 2 * precise * recall / (precise + recall)
        f1List.append(f1)
    f1Df = pd.DataFrame({'thr':thrList[:len(f1List)], 'f1':f1List}).sort_values(by=['f1','thr'], ascending=[False,True])
    if thrList is None:
        averThr = f1Df.head(5).sort_values(by=['thr']).head(4)['thr'].mean()    # 取前5，去掉最大阈值后取平均
        return averThr
    else:
        bestThr = thrList[f1List.index(max(f1List))]
        return bestThr

def custom_eval(preds, train_data):
    '''
    自定义F1评价函数
    '''
    labels = train_data.get_label()
    f1List = []
    thr = findF1Threshold(preds, labels, np.array(range(330,460,20)) * 0.001)
    predLabels = getPredLabel(preds, thr)
    f1 = metrics.f1_score(labels, predLabels)
    return 'f1', f1, True

def runLGBCV(train_X, train_y,vali_X=None,vali_y=None, seed_val=2012, num_rounds = 2000, random_state=None):
    def lgb_f1_score(y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat)
        return 'f1', metrics.f1_score(y_true, y_hat), True

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        # 'metric': 'custom',
        'metric': 'binary_logloss',
        'num_leaves': 127,
        'learning_rate': 0.02,
        'feature_fraction': 1,
        'bagging_fraction': 0.95,
    	'bagging_freq': 3,
        'num_threads':-1,
        'is_training_metric':True,
    }

    if random_state is not None:
        params['seed'] = random_state

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=['tag'])

    if vali_y is not None:
        lgb_vali = lgb.Dataset(vali_X,vali_y)
        model = lgb.train(params,lgb_train,num_boost_round=num_rounds,verbose_eval=10,early_stopping_rounds=200,
                          valid_sets=[lgb_vali, lgb_train],valid_names=['val', 'train'])#, feval=custom_eval

    else:
        model = lgb.train(params,lgb_train,num_boost_round=num_rounds,verbose_eval=10,
                          valid_sets=[lgb_train],valid_names=['train'])

    return model,model.best_iteration

def get_x_y(data):
    drop_list = ['prefix','query_prediction','title']
    # drop_list.extend(['pre_prefix','pre_title','next_prefix','next_title'])

    if 'label' in data.columns:
        y = data['label']
        data = data.drop(drop_list+['label'],axis=1)
    else:
        y=None
        data = data.drop(drop_list,axis=1)
    # print('------ ',data.shape)
    return data,y

def train_and_predict(samples,vali_samples,num_rounds=3000, **params):
    print('-- train and predict')
    print('---- get x and y')
    train_x,train_y = get_x_y(samples)
    vali_X,vali_y = get_x_y(vali_samples)

    print('------ train shape: %s, vali shape: %s' % (train_x.shape, vali_X.shape))
    gc.collect()

    print('---- training')
    model,best_iter = runLGBCV(train_x, train_y,vali_X,vali_y,num_rounds=num_rounds, **params)
    print('best_iteration:',best_iter)

    print('---- predict')
    vali_pred = model.predict(vali_X)
    return model,best_iter,vali_pred,vali_y

def result_analysis(res):
    print('mean : ',np.mean(res))

if __name__ == "__main__":
    # 路径
    train_dir = '../data/data_train.txt'
    vali_dir = '../data/data_vali.txt'
    test_dir = '../data/data_test.txt'
    vec_dir_1 = '../data/w2v_model/w2v_total_50wei.model'
    ext_vec_dirs = [
        # '../data/merge_sgns_bigram_char300/merge_sgns_bigram_char300.txt',
        ]
    srop_word_dir = '../data/stop_words.txt'
    dictionary_dir = './dictionary.txt'
    # test_result_dir = './lake_20181122.csv'
    modelName = 'xkl_drop'

    # 导入数据
    train_fea_dir = 'train_time.csv'
    vali_fea_dir = 'vali_time.csv'
    print("-- 导入原始数据", end='')
    if os.path.isfile(train_fea_dir) and os.path.isfile(vali_fea_dir):
    # if False:
        raw_train = importCacheDf(train_fea_dir)
        raw_vali = importCacheDf(vali_fea_dir)
    else:
        start = time()
        raw_train = importDf(train_dir, colNames=['prefix','query_prediction','title','tag','label'])
        raw_vali = importDf(vali_dir, colNames=['prefix','query_prediction','title','tag','label'])
        raw_test = importDf(test_dir, colNames=['prefix','query_prediction','title','tag'])
        raw_train = raw_train.reset_index().rename(columns={'index':'instance_id'})
        raw_vali = raw_vali.reset_index().rename(columns={'index':'instance_id'})
        raw_test = raw_test.reset_index().rename(columns={'index':'instance_id'})
        print('   cost: %.1f ' %(time() - start))

        # 清洗数据
        print("-- 清洗数据", end='')
        start = time()
        raw_train['query_prediction'].replace({'':'{}', np.nan:'{}'}, inplace=True)
        raw_vali['query_prediction'].replace({'':'{}', np.nan:'{}'}, inplace=True)
        raw_test['query_prediction'].replace({'':'{}', np.nan:'{}'}, inplace=True)
        raw_train = str_lower(raw_train)
        raw_vali = str_lower(raw_vali)
        raw_test = str_lower(raw_test)
        gc.collect()
        print('   cost: %.1f ' %(time() - start))

        # 提取全局特征
        print("-- 提取全局特征", end='')
        start = time()
        # raw_train['prefix_position'] = raw_train.apply(get_prefix_position, axis=1)
        # raw_vali['prefix_position'] = raw_vali.apply(get_prefix_position, axis=1)
        raw_train = queryNum(raw_train)
        raw_vali = queryNum(raw_vali)
        raw_test = queryNum(raw_test)
        # raw_train = timeFeas(raw_train)
        # raw_vali = timeFeas(raw_vali)
        # raw_test = timeFeas(raw_test)
        gc.collect()
        print('   cost: %.1f ' %(time() - start))

        ## 提取统计特征
        # 提取训练集统计特征
        print("-- 提取训练集统计特征", end='')
        start = time()
        tempTrain = raw_train
        raw_train = k_fold_stat_features(raw_train)
        gc.collect()
        print('   cost: %.1f ' %(time() - start))

        # 提取验证集统计特征
        print("-- 提取验证集统计特征", end='')
        start = time()
        raw_vali = stat_features(tempTrain, raw_vali)
        raw_test = stat_features(tempTrain, raw_test)
        del tempTrain
        gc.collect()
        print('   cost: %.1f ' %(time() - start))

        ## 提取文本特征
        '''
        1、tag进行labelEncoder
        '''
        print("-- 对tag进行encoder", end='')
        start = time()
        encoder = get_tag_dict(raw_train)
        raw_train['tag'] = encoder.transform(raw_train['tag'])
        raw_vali['tag'] = encoder.transform(raw_vali['tag'])
        raw_test['tag'] = encoder.transform(raw_test['tag'])
        print('   cost: %.1f ' %(time() - start))
        del encoder
        gc.collect()

        '''
        #3、其他
        '''
        # 导入模型和停用词表
        print("-- 导入词模型和停用词表", end='')
        start = time()
        w2v_model_1 = read_w2v_model(vec_dir_1)
        stop_words = read_stop_word(srop_word_dir)
        # dictionary = Dictionary.load_from_text(dictionary_dir)
        print('   cost: %.1f ' %(time() - start))

        # 提取其他文本特征
        print("-- 提取训练集其他文本特征", end='')
        start = time()
        raw_train = text_features(raw_train, w2v_model_1, ext_vec_dirs, stop_words=stop_words)
        gc.collect()
        print("-- 提取验证集其他文本特征", end='')
        raw_vali = text_features(raw_vali, w2v_model_1, ext_vec_dirs, stop_words=stop_words)
        raw_test = text_features(raw_test, w2v_model_1, ext_vec_dirs, stop_words=stop_words)
        del w2v_model_1, stop_words
        gc.collect()

        raw_train = raw_train.sort_values(['instance_id']).drop(['instance_id'],axis=1)
        raw_vali = raw_vali.sort_values(['instance_id']).drop(['instance_id'],axis=1)
        raw_test = raw_test.sort_values(['instance_id']).drop(['instance_id'],axis=1)

        # raw_train.to_csv(train_fea_dir, index=False)
        # raw_vali.to_csv(vali_fea_dir, index=False)
        # gc.collect()

        # dropCols = ['title_w2v_%d'%i for i in range(50)]
        # dropCols.extend(['prefix_w2v_%d'%i for i in range(50)])
        # dropCols.extend(['mx_w_query_w2v_%d'%i for i in range(50)])
        # saveTime = datetime.now()
        # raw_train.drop(dropCols, axis=1).to_csv('./temp/%s_fea_train_%s.csv' % (modelName, saveTime.strftime("%Y%m%d%H%M")), index=False)
        # raw_vali.drop(dropCols, axis=1).to_csv('./temp/%s_fea_vali_%s.csv' % (modelName, saveTime.strftime("%Y%m%d%H%M")), index=False)
        # raw_test.drop(dropCols, axis=1).to_csv('./temp/%s_fea_test_%s.csv' % (modelName, saveTime.strftime("%Y%m%d%H%M")), index=False)
        del raw_test
        gc.collect()

    best_iter_list = []
    best_logloss_list = []
    best_thr_list = []
    best_f1_list = []
    auc_list = []
    for rd in range(5):
        model,best_iter,vali_pred,vali_y = train_and_predict(raw_train, raw_vali, random_state=rd)
        best_iter_list.append(best_iter)
        best_logloss_list.append(metrics.log_loss(vali_y, vali_pred))

        # 计算AUC
        fpr, tpr, thresholds = metrics.roc_curve(vali_y, vali_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        auc_list.append(auc)
        print('-- auc score: %f' % auc)

        scores = []
        print('-- search best split point')
        for thre in range(100):
            thre *=0.01
            score = metrics.f1_score(vali_y,list(map(one_zero2,vali_pred,[thre]*len(vali_pred))))
            scores.append(score)

        scores = np.array(scores)
        best_5 = np.argsort(scores)[-5:]
        best_5_s = scores[best_5]
        best_f1_list.append(np.mean(best_5_s))
        for x,y in zip(best_5,best_5_s):
            print('%.2f  %.4f' %(0.01*x,y))
        max_thre = np.mean(best_5)*0.01
        best_thr_list.append(best_5[-1]*0.01)
    print('iter list:', best_iter_list)
    print('logloss list:', best_logloss_list)
    print('auc list:', auc_list)
    print('thr list:', best_thr_list)
    print('f1 list:', best_f1_list)

    # raw_vali['pred'] = vali_pred
    # raw_vali.to_csv('../result/%s_vali.csv' % modelName, index=False)

    # 特征重要性
    print('feature importance:')
    scoreDf = pd.DataFrame({
        'fea': model.feature_name(),
        'importance': model.feature_importance()
        })
    scoreDf.reset_index().sort_values(['importance'], ascending=False, inplace=True)
    print(scoreDf.head(100))
    # scoreDf.to_csv('./temp/fea_score.csv')
