# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, log_loss
import scipy as sp
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import jieba
from Levenshtein import distance as lev_distance
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from gensim.models import KeyedVectors, Word2Vec
from time import time
from multiprocessing import Pool
import gc, os
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',100)

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
    data_w = data*params['norm_weights']
    sample[name+'_min_'+func_name] = np.min(data,1)
    sample[name+'_max_'+func_name] = np.max(data,1)
    sample[name+'_mean_'+func_name] = np.divide(np.sum(data_w,1),sample['query_num'])
    sample[name+'_std_'+func_name] = np.sum(np.power(data      \
                                            - np.array(sample[name+'_mean_'+func_name]).reshape(-1,1),2)*params['norm_weights'],1)
    return sample

def weight_features(sample, **cache):
    print('------ weight features',end='')
    print('weight fea:',sample.shape, cache['weights'].shape)
    start = time()

    sample['weight_sum'] = np.sum(cache['weights'],1)
    sample = min_max_mean_std(sample,cache['weights'],'weight','', **cache)

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
    word_seg = [word for word in jieba.cut(sentence) if (word not in stop_words)]
    return word_seg

def get_sentence_vec(word_seg, w2v_model, **params):
    s_vector = np.zeros((len(w2v_model['我'])))
    if len(word_seg) > 0:
        count=0
        for word in word_seg :
            try:
                vec = w2v_model[word]
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
#     cache['query_embed'] = np.array(list(map(lambda x: [get_sentence_vec(q, **cache) for q in x], cache['query_seg'])))
#     cache['mx_w_query_embed'] = cache['query_embed'][cache['idx'], cache['weight_argmax']]
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
#     query_cos = np.array([cosine(cache['query_embed'][:,i,:], cache['title_embed']) for i in range(cache['query_embed'].shape[1])]).T
#     sample['query_title_mean_cos_%s'%alias] = np.divide(np.multiply(query_cos, cache['weights']).sum(axis=1), sample['query_num'])
#     sample['query_title_max_cos_%s'%alias] = query_cos.max(axis=1)
#     query_argmax_cos = query_cos.argmax(axis=1)
#     sample['query_title_max_cos_%s_weight'%alias] = cache['weights'][cache['idx'], query_argmax_cos]
#     sample['mx_w_query_title_cos_%s'%alias] = query_cos[cache['idx'], cache['weight_argmax']]
    sample['mx_w_query_title_cos_%s'%alias] = cosine(cache['title_embed'],cache['mx_w_query_embed'])

#     del query_cos,query_argmax_cos
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

def runLGBCV(train_X, train_y,vali_X=None,vali_y=None, seed_val=2012, num_rounds = 2000, random_state=None):
    def lgb_f1_score(y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat)
        return 'f1', f1_score(y_true, y_hat), True

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
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
        lgb_vali = lgb.Dataset(vali_X,vali_y, reference=lgb_train)
        model = lgb.train(params,lgb_train,num_boost_round=num_rounds,verbose_eval=10,early_stopping_rounds=200,
                          valid_sets=[lgb_vali],valid_names=['val'])

    else:
        model = lgb.train(params,lgb_train,num_boost_round=num_rounds,verbose_eval=10,
                          valid_sets=[lgb_train],valid_names=['train'])

    return model,model.best_iteration

def get_x_y(data):
    drop_list = ['prefix','query_prediction','title']
    if 'label' in data.columns:
        y = data['label']
        data.drop(drop_list+['label'],axis=1, inplace=True)
    else:
        y=None
        data.drop(drop_list,axis=1, inplace=True)
    print('------ ',data.shape)
    return data,y

def train_and_predict(samples,vali_samples,num_rounds=3000, **params):
    print('-- train and predict')
    print('---- get x and y')
    train_x,train_y = get_x_y(samples)
    vali_X,vali_y = get_x_y(vali_samples)
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
    train_dir = '../data/oppo_data_ronud2_20181107/data_train.txt'
    vali_dir = '../data/oppo_data_ronud2_20181107/data_vali.txt'
    test_dir = '../data/oppo_data_ronud2_20181107/data_test.txt'
    vec_dir_1 = '../data/w2v_model/w2v_total_50wei.model'
    vec_dir_2 = '../data/merge_sgns_bigram_char300/merge_sgns_bigram_char300.txt'
    ext_vec_dirs = [
        '../data/merge_sgns_bigram_char300/merge_sgns_bigram_char300.txt',
#         '../data/sgns.merge.bigram/sgns.merge.bigram',
#         '../data/sgns.merge.word/sgns.merge.word',
#         '../data/sgns.merge.char/sgns.merge.char',
        ]
    srop_word_dir = '../xkl/stop_words.txt'
    test_result_dir = './result/xkl_fea.csv'

    # 导入数据
    print("-- 导入原始数据", end='')
    start = time()
    raw_train = importDf(train_dir, colNames=['prefix','query_prediction','title','tag','label'])
    raw_vali = importDf(vali_dir, colNames=['prefix','query_prediction','title','tag','label'])
    raw_test = importDf(test_dir, colNames=['prefix','query_prediction','title','tag'])
    raw_train = pd.concat([raw_train, raw_vali], ignore_index=True).reset_index(drop=True)
    del raw_vali
    gc.collect()
    print('   cost: %.1f ' %(time() - start))

    # 清洗数据
    print("-- 清洗数据", end='')
    start = time()
    raw_train['query_prediction'].replace({'':'{}', np.nan:'{}'}, inplace=True)
    raw_test['query_prediction'].replace({'':'{}', np.nan:'{}'}, inplace=True)
    raw_train = str_lower(raw_train)
    raw_test = str_lower(raw_test)
    gc.collect()
    print('   cost: %.1f ' %(time() - start))

    # 提取全局特征
    print("-- 提取全局特征", end='')
    start = time()
    raw_train = queryNum(raw_train)
    raw_test = queryNum(raw_test)
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
    raw_test['tag'] = encoder.transform(raw_test['tag'])
    print('   cost: %.1f ' %(time() - start))
    del encoder
    gc.collect()


    '''
    #2、其他
    '''
    # 导入模型和停用词表
    print("-- 导入词模型和停用词表", end='')
    start = time()
    w2v_model_1 = read_w2v_model(vec_dir_1)
    stop_words = read_stop_word(srop_word_dir)
    print('   cost: %.1f ' %(time() - start))

    # 提取其他文本特征
    print("-- 提取训练集其他文本特征", end='')
    start = time()
    raw_train = text_features(raw_train, w2v_model_1, ext_vec_dirs, stop_words=stop_words)
    gc.collect()
    print("-- 提取验证集其他文本特征", end='')
    raw_test = text_features(raw_test, w2v_model_1, ext_vec_dirs, stop_words=stop_words)
    del w2v_model_1, stop_words
    gc.collect()

    #开始训练
#     best_iter = 663
#     max_thre = 0.37
    best_iter = 958
    max_thre = 0.37
    print('-- final training ')
    train_X,train_y = get_x_y(raw_train)
    model_,best_iter_ = runLGBCV(train_X, train_y,num_rounds=best_iter)
    print('best_iteration:',best_iter)

    print('---- predict')
    predict_start = time()
    test_X,_ = get_x_y(raw_test)
    test_pred = model_.predict(test_X)

    raw_test['pred'] = vali_pred
    raw_test.to_csv('./result/xkl_fea_testa.csv', index=False)

    print('-- process to get result')
    test_y = pd.Series(list(map(one_zero2,test_pred,[max_thre]*len(test_pred))))
    test_y.to_csv(test_result_dir,header=None,index=None)

    print('print result')
    result_analysis(test_pred)
    print('feature importance:')
    scoreDf = pd.DataFrame({'fea': train_X.columns, 'importance': model_.feature_importance()})
    print(scoreDf.sort_values(['importance'], ascending=False).head(100))
    scoreDf.sort_values(['importance'], ascending=False).to_csv('./fea_score.csv')
