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
from gensim.models import KeyedVectors, Word2Vec
from time import time
from multiprocessing import Pool
import gc, os
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
        w2v_model = Word2Vec.load(model_dir)
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

def stat_features(raw, sample):
    sample = str_lower(sample)
    sample = lake_features(raw, sample)
    sample = ctr_features(raw, sample)
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

def text_features(sample, w2v_model_1, ext_vec_dirs, stop_words):
    def get_query_weight(data):
        print('------ split query and weight', end='')
        start = time()
        def split_query_weight(data):
            query_prediction = eval(data)
            query = [key.lower() for key in sorted(query_prediction)][:11]
            weight = [float(query_prediction[key]) for key in sorted(query_prediction)][:11]
            return [query, weight]

        querys = []
        weights = []
        for query, weight in map(split_query_weight, data):
            querys.append(query)
            weights.append(weight)
        print('get query weight:', data.shape, len(querys), len(weights))
        querys = pd.DataFrame(querys,columns=['query_'+str(i) for i in range(11)]).fillna('')
        weights = pd.DataFrame(weights,columns=['weight_'+str(i) for i in range(11)]).fillna(0)
        querys = np.array(querys)
        weights = np.array(weights)

        norm_weights = weights / (np.sum(weights,1).reshape((-1,1))+0.001)
        print('get query weight:', querys.shape, weights.shape, norm_weights.shape)

        print('   cost: %.1f ' %(time()-start))
        return querys, weights, norm_weights

    def min_max_mean_std(sample,data,name,func_name):
        data_w = data*norm_weights
        sample[name+'_min_'+func_name] = np.min(data,1)
        sample[name+'_max_'+func_name] = np.max(data,1)
        sample[name+'_mean_'+func_name] =np.divide(np.sum(data_w,1),sample['query_num'])
        sample[name+'_std_'+func_name] = np.sum(np.power(data      \
                                                - np.array(sample[name+'_mean_'+func_name]).reshape(-1,1),2)*norm_weights,1)
        return sample

    def get_max_weight_idx():
        weight_argmax = tuple(np.argmax(weights,1))
        idx = tuple(range(len(weight_argmax)))
        return idx,weight_argmax

    def split_sentence(s):
        return [w for w in jieba.cut(s) if w not in stop_words]

    def lev_features(sample):
        print('------ lev features', end='')
        start = time()
        def get_lev_dist_list(query,data):
            query_data_levs = [lev_distance(q,data) for q in query]
            return query_data_levs

        query_title_levs = map_to_array(get_lev_dist_list,querys,sample['title'])
        sample = min_max_mean_std(sample,query_title_levs,'query_title','lev')
        sample['mx_w_query_title_lev'] = query_title_levs[idx,weight_argmax]
        sample['prefix_title_lev'] = list(map(lev_distance,sample['prefix'],sample['title']))

        max_w_query = querys[idx,weight_argmax]
        sample['mx_w_prefix_query_lev'] = list(map(lev_distance,sample['prefix'],max_w_query))

        levs = pd.DataFrame(np.sort(query_title_levs, axis=1),columns=['lev_'+str(i) for i in range(11)], index=sample.index)

        sample = pd.concat([sample,levs],axis=1)

        print('   cost: %.1f ' %(time()-start))
        return sample

    def len_features(sample):
        print('------ len features',end = '')
        start = time()
        def get_query_len(query):
            q_lens = [len(q) for q in query]
            return q_lens

        querys_lens = map_to_array(get_query_len,querys)
        sample = min_max_mean_std(sample,querys_lens,'query','len')

        sample['prefix_len'] = list(map(len,sample['prefix']))
        sample['title_len'] = list(map(len,sample['title']))

        max_w_query_len = querys_lens[idx,weight_argmax]

        sample['mx_w_prfx_qry_len_sub'] = max_w_query_len-sample['prefix_len']

        sample['title_prefix_len_sub'] = sample['title_len']-sample['prefix_len']
        sample['query_title_len_sub'] = sample['query_mean_len'] - sample['title_len']
        sample['query_prefix_len_sub'] = sample['query_mean_len'] - sample['prefix_len']
        sample['prefix_query_len_div'] = sample['prefix_len'].div(sample['query_mean_len'])
        sample['prefix_title_len_div'] = sample['prefix_len'].div(sample['title_len'])

        print('   cost: %.1f ' %(time()-start))

        return sample

    def weight_features(sample):
        print('------ weight features',end='')
        print('weight fea:',sample.shape, weights.shape)
        start = time()

        num = weights.copy()
        num[num>0]=1
        sample['query_num'] = np.sum(num,axis=1)

        sample['weight_sum'] = np.sum(weights,1)
        sample = min_max_mean_std(sample,weights,'weight','')

        print('   cost: %.1f ' %(time()-start))
        return sample

    def get_sentence_vec(sentence):
        s_vector = np.zeros((len(w2v_model['我'])))
        if sentence:
            count=0
            for word in jieba.cut(sentence) :
                if word not in stop_words:
                    try:
                        vec = w2v_model[word]
                        s_vector += vec
                        count += 1
                    except (KeyError):
                        pass
            if count:
                s_vector /= count
        return s_vector

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

    def sentence_simi(s1,s2):
        s1_vec = get_sentence_vec(s1)
        s2_vec = get_sentence_vec(s2)
        cos = cosine(s1_vec,s2_vec)
        return cos

    def query_data_cos(query,data):
        q_data_cos = [sentence_simi(q,data) for q in query]
        return q_data_cos

    def word2vec_features_1(sample,cos_feature=False):
        print('------ word2vec features 1',end='')
        start = time()

        title_embed = map_to_array(get_sentence_vec,sample['title'])
        prefix_embed = map_to_array(get_sentence_vec,sample['prefix'])

        max_w_query = querys[idx,weight_argmax]
        mx_w_query_embed = map_to_array(get_sentence_vec,max_w_query)

        if cos_feature:
            querys_title_cos = [cosine(map_to_array(get_sentence_vec,querys[:,i],paral=True),title_embed) for i in range(11)]
            querys_title_cos = np.array(querys_title_cos).T
            sample = min_max_mean_std(sample,querys_title_cos,'querys_title','cos')
            sample['mx_w_query_title_cos'] = querys_title_cos[idx,weight_argmax]

        sample['prefix_title_cos'] = cosine(title_embed,prefix_embed)
        sample['prefix_mx_query_cos'] = cosine(prefix_embed,mx_w_query_embed)
        sample['mx_w_query_title_cos'] = cosine(mx_w_query_embed,title_embed)

        title_embed = pd.DataFrame(title_embed,columns=['title_w2v_'+str(i) for i in range(50)], index=sample.index)
        sample = pd.concat([sample,title_embed],axis=1)

        prefix_embed = pd.DataFrame(prefix_embed,columns=['prefix_w2v_'+str(i) for i in range(50)], index=sample.index)
        sample = pd.concat([sample,prefix_embed],axis=1)

        mx_w_query_embed = pd.DataFrame(mx_w_query_embed,columns=['mx_w_query_w2v_'+str(i) for i in range(50)], index=sample.index)
        sample = pd.concat([sample,mx_w_query_embed],axis=1)

        print('   cost: %.1f ' %(time()-start))
        return sample

    def word2vec_features_2(sample, alias='2'):
        print('------ word2vec features 2',end='')
        start = time()

        def calc_all_cos(s1,s2,s3):
            prefix_embed = get_sentence_vec(s1)
            title_embed = get_sentence_vec(s2)
            mx_w_query_embed = get_sentence_vec(s3)
            cos = [0,0,0]
            cos[0] = cosine(prefix_embed,title_embed)
            cos[1] = cosine(prefix_embed,mx_w_query_embed)
            cos[2] = cosine(title_embed,mx_w_query_embed)
            return cos

        max_w_query = querys[idx,weight_argmax]
        cos = list(map(calc_all_cos,sample['prefix'],sample['title'],max_w_query))

        cos = pd.DataFrame(cos,columns=['prefix_title_cos_%s'%alias,'prefix_mx_query_cos_%s'%alias,'mx_w_query_title_cos_%s'%alias], index=sample.index)

        sample = pd.concat([sample,cos],axis=1)
        print('   cost: %.1f ' %(time()-start))
        return sample

    def jaccard_features(sample):
        print('------ jaccard features',end='')
        start = time()
        def jaccard(s1,s2):
            inter=len([w for w in s1 if w in s2])
            union = len(s1)+len(s2)-inter
            return inter/(union+0.001)

        def jaccard_dist(querys,data):
            res = [jaccard(q,data) for q in querys]
            return res

        print('jaccard->querys:', querys.shape)
        querys_title_jac = map_to_array(jaccard_dist,querys,sample['title'])
        sample = min_max_mean_std(sample,querys_title_jac,'query_title','jac')
        sample['mx_w_query_title_jac'] = querys_title_jac[idx,weight_argmax]
        sample['prefix_title_jac'] = list(map(jaccard,sample['prefix'],sample['title']))

        jacs = pd.DataFrame(-np.sort(-querys_title_jac,axis=1),columns=['query_title_jac_'+str(i) for i in range(11)], index=sample.index)

        sample = pd.concat([sample,jacs],axis=1)

        sample = sample.fillna(0)
        print('   cost: %.1f ' %(time()-start))
        return sample


    sample = str_lower(sample)
    tempDf = sample.drop_duplicates(['prefix','query_prediction','title'])
    # tempDf['old_prefix'] = tempDf['prefix']
    # tempDf['old_title'] = tempDf['title']
    querys,weights,norm_weights = get_query_weight(tempDf['query_prediction'])
#    tempDf.drop(['query_prediction'],axis=1,inplace=True)
    gc.collect()
    idx,weight_argmax = get_max_weight_idx()
    tempDf = weight_features(tempDf)
    gc.collect()
    tempDf = len_features(tempDf)
    gc.collect()
    tempDf = lev_features(tempDf)
    gc.collect()
    tempDf = jaccard_features(tempDf)
    gc.collect()
    w2v_model = w2v_model_1
    tempDf = word2vec_features_1(tempDf)
    gc.collect()

    # w2v_model = w2v_model_2
    for i,dir in enumerate(ext_vec_dirs):
        w2v_model = read_w2v_model(dir, persist=False)#
        tempDf = word2vec_features_2(tempDf,str(i+2))
        del w2v_model
        gc.collect()

    sample = sample.merge(
        tempDf[['prefix','query_prediction','title'] + np.setdiff1d(tempDf.columns,sample.columns).tolist()],
        how='left',
        on=['prefix','query_prediction','title'])
    return sample

def runLGBCV(train_X, train_y,vali_X=None,vali_y=None, seed_val=2012, num_rounds = 2000, random_state=None):
    def lgb_f1_score(y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat)
        return 'f1', metrics.f1_score(y_true, y_hat), True

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 127,
        'learning_rate': 0.02,
        'feature_fraction': 1,
        'bagging_fraction': 0.95,
    	'bagging_freq': 5,
        'num_threads':-1,
        'is_training_metric':True,
    }

    if random_state is not None:
        params['seed'] = random_state

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=['tag'])#

    if vali_y is not None:
        lgb_vali = lgb.Dataset(vali_X,vali_y)
        model = lgb.train(params,lgb_train,num_boost_round=num_rounds,verbose_eval=10,early_stopping_rounds=200,
                          valid_sets=[lgb_vali, lgb_train],valid_names=['val', 'train'])

    else:
        model = lgb.train(params,lgb_train,num_boost_round=num_rounds,verbose_eval=10,
                          valid_sets=[lgb_train],valid_names=['train'])

    return model,model.best_iteration

def get_x_y(data):
    drop_list = ['prefix','query_prediction','title']
    # drop_list.extend(['title_w2v_'+str(i) for i in range(50)])
    # drop_list.extend(['prefix_w2v_'+str(i) for i in range(50)])
    # drop_list.extend(['mx_w_query_w2v_'+str(i) for i in range(50)])

    # drop_list.extend(['prefix_position'])
    # drop_list.extend(['%s_ctr'%x for x in ['prefix_position','prefix_prefix_position','title_prefix_position','tag_prefix_position']])
    # drop_list.extend(['%s_count'%x for x in ['prefix_position','prefix_prefix_position','title_prefix_position','tag_prefix_position']])
    # drop_list.extend(['%s_click'%x for x in ['prefix_position','prefix_prefix_position','title_prefix_position','tag_prefix_position']])

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

    # train_x, extra_x, train_y, extra_y = train_test_split(train_x, train_y, train_size=0.95, random_state=params['random_state'])
    # vali_X = pd.concat([extra_x, vali_X], copy=False)
    # vali_y = pd.concat([extra_y, vali_y], copy=False)
    # del extra_x, extra_y

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
    # vec_dir_2 = '../data/merge_sgns_bigram_char300/merge_sgns_bigram_char300.txt'
    ext_vec_dirs = [
        # '../data/w2v_model/w2v_total_50wei.model',
        # '../data/w2v_model/w2v_total_50wei.model',
        # '../data/'
        ]
    srop_word_dir = '../data/stop_words.txt'
    test_result_dir = './lake_20181122.csv'

    # 导入数据
    train_fea_dir = 'train_re_new.csv'
    vali_fea_dir = 'vali_re_new.csv'
    print("-- 导入原始数据", end='')
    # if os.path.isfile(train_fea_dir) and os.path.isfile(vali_fea_dir):
    if False:
        raw_train = importCacheDf(train_fea_dir)
        raw_vali = importCacheDf(vali_fea_dir)
    else:
        start = time()
        raw_train = importDf(train_dir, colNames=['prefix','query_prediction','title','tag','label'], nrows=100000)
        raw_vali = importDf(vali_dir, colNames=['prefix','query_prediction','title','tag','label'])
        print('   cost: %.1f ' %(time() - start))

        # 清洗数据
        print("-- 清洗数据", end='')
        start = time()
        raw_train['query_prediction'].replace({'':'{}', np.nan:'{}'}, inplace=True)
        raw_vali['query_prediction'].replace({'':'{}', np.nan:'{}'}, inplace=True)
        raw_train = str_lower(raw_train)
        raw_vali = str_lower(raw_vali)
        # raw_train = fillnaQuery(raw_train, na='{}')
        # raw_vali = fillnaQuery(raw_vali, na='{}')
        gc.collect()
        print('   cost: %.1f ' %(time() - start))

        # # 提取全局特征
        # print("-- 提取全局特征", end='')
        # start = time()
        # raw_train['prefix_position'] = raw_train.apply(get_prefix_position, axis=1)
        # raw_vali['prefix_position'] = raw_vali.apply(get_prefix_position, axis=1)
        # gc.collect()
        # print('   cost: %.1f ' %(time() - start))

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
        print('   cost: %.1f ' %(time() - start))

        # 提取其他文本特征
        print("-- 提取训练集其他文本特征", end='')
        start = time()
        raw_train = text_features(raw_train, w2v_model_1, ext_vec_dirs, stop_words)
        gc.collect()
        print("-- 提取验证集其他文本特征", end='')
        raw_vali = text_features(raw_vali, w2v_model_1, ext_vec_dirs, stop_words)
        del w2v_model_1, stop_words
        gc.collect()

        # raw_train.to_csv(train_fea_dir, index=False)
        # raw_vali.to_csv(vali_fea_dir, index=False)

        temp_train = raw_train.describe().T
        temp_vali = raw_vali.describe().T
        temp_train.to_csv(train_fea_dir)
        temp_vali.to_csv(vali_fea_dir)
        exit()

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
        best_thr_list.append(max_thre)
    print('iter list:', best_iter_list)
    print('logloss list:', best_logloss_list)
    print('auc list:', auc_list)
    print('thr list:', best_thr_list)
    print('f1 list:', best_f1_list)
