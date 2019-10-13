#!/usr/bin/env python
# coding: utf-8

# In[45]:


import os
import pandas as pd
import numpy as np
import time
import gc
import random
import jieba
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,KFold,train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD,PCA,LatentDirichletAllocation,NMF
from collections import Counter
from gensim.models import Word2Vec
import gensim
import sklearn


# In[47]:


"""版本信息"""
print('xgb:',xgb.__version__, 'pandas:', pd.__version__, 'numpy:',np.__version__, 'sklearn:',sklearn.__version__, 'gensim:',gensim.__version__)
# xgb: 0.90 pandas: 0.23.4 numpy: 1.16.4 sklearn: 0.21.2 gensim: 3.8.0


# In[2]:


def cc_label(i):
    if i=='传输系统-传输设备':
        return 0
    elif i=='传输系统-光缆故障':
        return 1
    elif i=='传输系统-其他原因':
        return 2
    elif i=='动力环境-UPS':
        return 3
    elif i=='动力环境-电力部门供电':
        return 4
    elif i=='动力环境-电源线路故障':
        return 5
    elif i=='动力环境-动环监控系统':
        return 6
    elif i=='动力环境-动力环境故障':
        return 7
    elif i=='动力环境-高低压设备':
        return 8
    elif i=='动力环境-环境':
        return 9
    elif i=='动力环境-开关电源':
        return 10
    elif i=='其他-误告警或自动恢复':
        return 11
    elif i=='人为操作-告警测试':
        return 12
    elif i=='人为操作-工程施工':
        return 13
    elif i=='人为操作-物业原因':
        return 14
    elif i=='主设备-参数配置异常':
        return 15
    elif i=='主设备-其他':
        return 16
    elif i=='主设备-软件故障':
        return 17
    elif i=='主设备-设备复位问题':
        return 18
    elif i=='主设备-设备连线故障':
        return 19
    elif i=='主设备-天馈线故障':
        return 20
    elif i=='主设备-信源问题':
        return 21
    elif i=='主设备-硬件故障':
        return 22


# In[3]:


path = '../data/'
"""读取数据,要在这里修改路径"""
train_label= pd.read_csv(open(path+'训练故障工单.csv'))
test_label= pd.read_csv(open(path+'测试故障工单.csv'))
print(train_label.shape)
train_label=train_label.drop_duplicates(['故障发生时间','涉及告警基站或小区名称',\
                   '故障原因定位（大类）']).reset_index(drop=True) ##去重
print(train_label.shape)
traing= pd.read_csv(open(path+'训练告警.csv'))
testg= pd.read_csv(open(path+'测试告警.csv'))


# In[4]:


label=pd.concat([train_label,test_label],axis=0).reset_index(drop=True)
label['guzhang_time'] = label['故障发生时间']
label['故障发生时间'] = pd.to_datetime(label['故障发生时间'])
label=label.sort_values(by=['涉及告警基站或小区名称','故障发生时间']).reset_index(drop=True)

data=pd.concat([traing,testg],axis=0).reset_index(drop=True)
data=data[~data['涉及告警基站或小区名称'].isnull()]
data=data.drop_duplicates().reset_index(drop=True)

data['告警发生时间'] = data['告警发生时间'].apply(lambda x:x.replace('FEB','02'))
data['告警发生时间'] = data['告警发生时间'].apply(lambda x:x.replace('MAR','03'))
data['告警发生时间'] = pd.to_datetime(data['告警发生时间'],format="%d-%m-%Y %H:%M:%S")


# In[5]:


"""故障发生时间w2v"""
def guzhang_time_w2v(label):
    temp = label[['涉及告警基站或小区名称', 'guzhang_time']].copy()
    temp = temp.groupby(['涉及告警基站或小区名称'])['guzhang_time'].apply(list).reset_index(name='qwer')
    temp = temp.sample(frac=1.0, random_state=666)

    sen = temp['qwer'].values.tolist()
    try:
        fastmodel = Word2Vec.load('model_guzhangqwer300.txt')
    except:
        fastmodel =Word2Vec(sen,size=300,window=8,min_count=1,sg=1,workers=4,iter=10)
        fastmodel.save('model_guzhangqwer300.txt')
    w2v = []
    sen = label['guzhang_time'].values.tolist()
    for i in range(len(sen)):
        w2v.append(np.mean(fastmodel.wv[sen[i]],axis=0))
    del sen;gc.collect()
    w2v_time_df = pd.DataFrame(w2v)
    del w2v;gc.collect()
    w2v_time_df.columns = ['故障时间_'+str(i+1) for i in w2v_time_df.columns]
    return  w2v_time_df


# In[6]:


w2v_time_df = guzhang_time_w2v(label)
label = pd.concat([label, w2v_time_df], axis=1)
del label['guzhang_time']


# In[7]:


"""告警标题tfidf"""
feat = data[['涉及告警基站或小区名称', '告警标题']].copy()
feat = feat.groupby(['涉及告警基站或小区名称'])['告警标题'].agg(lambda x: ' '.join(x)).reset_index(name='Product_list')
tf_idf = TfidfVectorizer(max_features=100)
tf_vec = tf_idf.fit_transform(feat['Product_list'].values.tolist())
tf_df = pd.DataFrame(tf_vec.toarray())
tf_df['涉及告警基站或小区名称'] = feat['涉及告警基站或小区名称'].values
tf_df.columns = ['idf_'+str(i+1) for i in range(100)]+ ['涉及告警基站或小区名称']
label = label.merge(tf_df, on='涉及告警基站或小区名称', how='left')


# In[8]:


data['告警发生时间_mon'] = data['告警发生时间'].dt.month
data['告警发生时间_day'] = data['告警发生时间'].dt.day
data['告警发生时间_dayofyear'] = data['告警发生时间'].dt.dayofyear
data['告警发生时间_hour'] = data['告警发生时间'].dt.hour
data['告警发生时间_weekday'] = data['告警发生时间'].dt.weekday
data['告警发生时间_wy'] = data['告警发生时间'].dt.weekofyear
data['告警发生时间_是否周末'] = data['告警发生时间_weekday'].apply(lambda x:1 if x >= 5 else 0)


# In[9]:


label['故障发生时间_mon'] = label['故障发生时间'].dt.month
label['故障发生时间_day'] = label['故障发生时间'].dt.day
label['故障发生时间_dayofyear'] = label['故障发生时间'].dt.dayofyear
label['故障发生时间_hour'] = label['故障发生时间'].dt.hour
label['故障发生时间_weekday'] = label['故障发生时间'].dt.weekday
label['故障发生时间_wy'] = label['故障发生时间'].dt.weekofyear
label['故障发生时间_是否周末'] = label['故障发生时间_weekday'].apply(lambda x:1 if x >= 5 else 0)


# In[10]:


data['告警发生时间_int'] = data['告警发生时间'].apply(lambda x: x.value//10**9)
data['告警发生时间_diff'] = data.groupby(['涉及告警基站或小区名称'])['告警发生时间_int'].diff()
del data['告警发生时间_int']


# In[11]:


"""历史统计特征"""
def make_feature(data,aggs,name):
    agg_data = data.groupby('涉及告警基站或小区名称').agg(aggs)
    agg_data.columns = agg_data.columns = ['_'.join(i).strip()+name for i in agg_data.columns.tolist()]
    return agg_data.reset_index()
aggs = {
    '告警标题': ['count', 'nunique'],
    '告警发生时间_是否周末': ['mean'],
    '告警发生时间_hour': ['nunique'],
    '告警发生时间_weekday': ['nunique'],
    '告警发生时间_day':['nunique'],
    '告警发生时间_wy':['nunique'],
    '告警发生时间_dayofyear':['nunique', 'max', 'min', np.ptp],
    '告警发生时间_diff':['min', 'max', 'mean', 'std'],
}
agg_df=make_feature(data,aggs,"_告警")
label = label.merge(agg_df, on='涉及告警基站或小区名称', how='left')


# In[12]:


"""告警标题按周w2v"""
data = data.sort_values('告警发生时间')
feat = data.groupby(['涉及告警基站或小区名称', '告警发生时间_wy'])['告警标题'].apply(list).reset_index()
feat = feat.sample(frac=1.0, random_state=112)
sen = feat['告警标题'].values.tolist()
try:
    fastmodel = Word2Vec.load('model_fusaigaojing.txt')
except:
    fastmodel =Word2Vec(sen,size=100,window=8,min_count=1,workers=4,iter=10)
    fastmodel.save('model_fusaigaojing.txt')
w2v = []
for i in range(len(sen)):
    w2v.append(np.mean(fastmodel.wv[sen[i]],axis=0))
del sen;gc.collect()
w2v_df = pd.DataFrame(w2v)
del w2v;gc.collect()
w2v_df.columns = ['告警标题_'+str(i+1) for i in w2v_df.columns]
w2v_df['涉及告警基站或小区名称'] = feat['涉及告警基站或小区名称'].values
w2v_df['告警发生时间_wy'] = feat['告警发生时间_wy'].values
w2v_df = w2v_df.rename(columns={'告警发生时间_wy': '故障发生时间_wy'})


# In[13]:


label['涉及告警基站或小区名称_code'] = preprocessing.LabelEncoder().fit_transform(label['涉及告警基站或小区名称'])


# In[14]:


label['涉及告警基站或小区名称_count'] = label.groupby('涉及告警基站或小区名称')['故障发生时间'].transform('count')


# In[15]:


"""小时展开表特征"""
pivot = pd.pivot_table(label, index = '涉及告警基站或小区名称', columns= '故障发生时间_hour',values=['工单编号'],aggfunc='count').reset_index().fillna(0)
pivot.columns = ['涉及告警基站或小区名称'] + pivot['工单编号'].columns.tolist()
label = label.merge(pivot, on='涉及告警基站或小区名称', how='left')


# In[16]:


"""历史统计特征"""
label['故障发生时间_day_count'] = label.groupby(['涉及告警基站或小区名称', '故障发生时间_dayofyear'])['工单编号'].transform('count')
label['故障发生时间_week_count'] = label.groupby(['涉及告警基站或小区名称', '故障发生时间_wy'])['工单编号'].transform('count')
label['故障发生时间_hour_count'] = label.groupby(['涉及告警基站或小区名称', '故障发生时间_hour'])['工单编号'].transform('count')
label['故障发生时间_周末_count'] = label.groupby(['涉及告警基站或小区名称', '故障发生时间_是否周末'])['工单编号'].transform('count')
label['故障发生时间_weekday_count'] = label.groupby(['涉及告警基站或小区名称', '故障发生时间_weekday'])['工单编号'].transform('count')
label['故障发生时间_故障发生时间_count'] = label.groupby(['涉及告警基站或小区名称', '故障发生时间'])['工单编号'].transform('count')


# In[17]:


label['小区故障时间_min'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_dayofyear'].transform('min')
label['小区故障时间_max'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_dayofyear'].transform('max')
label['小区故障时间_ptp'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_dayofyear'].transform(np.ptp)


# In[18]:


label['故障发生时间_int'] = label['故障发生时间'].apply(lambda x: x.value//10**9)
label['故障发生时间_diff'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_int'].diff()
label['故障发生时间_diff_min'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_diff'].transform('min')
label['故障发生时间_diff_max'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_diff'].transform('max')
label['故障发生时间_diff_mean'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_diff'].transform('mean')
label['故障发生时间_diff_std'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_diff'].transform('std')
del label['故障发生时间_int']


# In[19]:


def make_feature(data,aggs,name):
    agg_data = data.groupby('涉及告警基站或小区名称').agg(aggs)
    agg_data.columns = agg_data.columns = ['_'.join(i).strip()+name for i in agg_data.columns.tolist()]
    return agg_data.reset_index()
aggs = {
    '故障发生时间_是否周末': ['mean'],
    '故障发生时间_hour': ['nunique'],
    '故障发生时间_weekday': ['nunique'],
    '故障发生时间_day':['nunique'],
    '故障发生时间_wy':['nunique'],
    '故障发生时间_dayofyear':['nunique', 'max', 'min', np.ptp]
}
agg_df=make_feature(label,aggs,"_故障")
label = label.merge(agg_df, on='涉及告警基站或小区名称', how='left')


# In[20]:


"""比率特征"""
for i in ['告警标题_count_告警', '告警标题_nunique_告警']:
    label[i+'_ratio'] = label['涉及告警基站或小区名称_count'] / (label[i]+1)
for i in ['hour_nunique', 'weekday_nunique', 'day_nunique', 'wy_nunique', 'dayofyear_nunique']:
    label[i+'_ratio'] = label['故障发生时间_'+i+'_故障'] / (label['告警发生时间_'+i+'_告警']+1)


# In[21]:


"""告警表特征"""
print(label.shape)
for i in ['_dayofyear', '_wy', '_hour', '_是否周末', '_weekday']:
    haha = data.groupby(['涉及告警基站或小区名称', '告警发生时间'+i])['告警标题'].nunique().reset_index(name='告警发生时间'+i+'_nunique')
    haha.columns = ['涉及告警基站或小区名称', '故障发生时间'+i, '告警发生时间'+i+'_nunique']
    label = label.merge(haha,on = ['涉及告警基站或小区名称', '故障发生时间'+i], how='left')
print(label.shape)


# In[22]:


"""告警表特征"""
print(label.shape)
for i in ['_dayofyear', '_wy', '_hour', '_是否周末', '_weekday']:
    haha = data.groupby(['涉及告警基站或小区名称', '告警发生时间'+i])['告警标题'].count().reset_index(name='告警发生时间'+i+'_count')
    haha.columns = ['涉及告警基站或小区名称', '故障发生时间'+i, '告警发生时间'+i+'_count']
    label = label.merge(haha,on = ['涉及告警基站或小区名称', '故障发生时间'+i], how='left')
print(label.shape)


# In[23]:


label['rank'] = label.groupby('涉及告警基站或小区名称')['故障发生时间'].rank(method='dense')


# In[24]:


label['故障原因定位（大类）'] = label['故障原因定位（大类）'].map(cc_label)


# In[25]:


label = label.merge(w2v_df, on=['涉及告警基站或小区名称', '故障发生时间_wy'], how='left')


# In[ ]:


def accua(df1, df_oof):
    rr = list(np.argsort(df_oof,axis=1)[:,-3:])
    acc = 0
    for i in range(len(df1)):
        acc += len(set([df1.values[i]])&set(list(rr[i])))
    return acc/len(df1)


# In[27]:


train = label[~label['故障原因定位（大类）'].isnull()]
test = label[label['故障原因定位（大类）'].isnull()]


# In[28]:


col = [i for i in train.columns if i not in ['涉及告警基站或小区名称', '故障发生时间','guzhang_time',
                                             '工单编号','告警发生时间','故障原因定位5','故障发生时间_minute',
                                             '故障原因定位（大类）']]
X_train = train[col].copy().reset_index(drop=True)
y_train = train['故障原因定位（大类）'].copy().reset_index(drop=True).astype(int)
X_test = test[col].copy()
print(X_train.shape, X_test.shape)
#(59758, 254) (6696, 254)


# In[29]:


K = 5
seed = 2021
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
xgb_pred_te_all = 0
xgb_auc_mean = 0
xgb_auc_mean2 = 0
f1 = []
logsocre = []
oof_xgb = np.zeros((len(X_train), 23))
for i, (train_index, test_index) in enumerate(skf.split(X_train,y_train)):
        y_tr, y_val = y_train.iloc[train_index].copy(), y_train.iloc[test_index].copy()
        X_tr, X_val= X_train.iloc[train_index,:].copy(), X_train.iloc[test_index,:].copy()
        print( "\nFold ", i+1)

        xgb_tr = xgb.DMatrix(X_tr, y_tr)
        xgb_val = xgb.DMatrix(X_val, y_val)
        xgb_te = xgb.DMatrix(X_test)
        xgb_params = {"objective": 'multi:softprob',
                      "booster" : "gbtree",
                      "eta": 0.07,
                      "max_depth":9,
                      "subsample": 0.85,
                      'eval_metric':'mlogloss',
                      'num_class': 23, 
                      "colsample_bylevel":0.7,
                      'tree_method':'gpu_hist',
                      'lambda':6,                                 
                      "thread":12,
                      "seed": 666
                      }
        watchlist = [(xgb_tr, 'train'), (xgb_val, 'eval')]
        xgb_model =xgb.train(xgb_params,
                     xgb_tr,
                     num_boost_round = 2666,
                     evals =watchlist,
                     verbose_eval=200,
                     early_stopping_rounds=100)

        pred = xgb_model.predict(xgb_val, ntree_limit=xgb_model.best_ntree_limit)
        oof_xgb[test_index] = pred
        val_pred = [np.argmax(x) for x in pred]
        acc = accuracy_score(y_val, val_pred)
        logsocre.append(xgb_model.best_score)
        print( "                       acc_score = ", acc )
        print('auc :', roc_auc_score(pd.get_dummies(y_val).values, pred,average='weighted'))
        print('acc_3 :', accua(y_val, pred))
        f1.append(acc)
        print("*"*100)
        pred_te = xgb_model.predict(xgb_te,ntree_limit=xgb_model.best_ntree_limit)
        xgb_pred_te_all = xgb_pred_te_all + pred_te / K
print("="*50+'result'+"="*50)
print( " mean_f1 = ", np.mean(f1) ,np.std(f1))
print( " mean_mlog = ", np.mean(logsocre) ,np.std(logsocre))


# In[ ]:


"""本地最优线下输出结果"""
#  mean_f1 =  0.7044736457258982 0.003106892835853258
#  mean_mlog =  0.974696 0.008362968013809447


# In[39]:


res=pd.DataFrame(xgb_pred_te_all,
                 columns=['传输系统-传输设备','传输系统-光缆故障','传输系统-其他原因',
                         '动力环境-UPS','动力环境-电力部门供电','动力环境-电源线路故障',
                         '动力环境-动环监控系统','动力环境-动力环境故障','动力环境-高低压设备',
                         '动力环境-环境','动力环境-开关电源','其他-误告警或自动恢复',
                         '人为操作-告警测试','人为操作-工程施工','人为操作-物业原因',
                         '主设备-参数配置异常','主设备-其他','主设备-软件故障',
                         '主设备-设备复位问题','主设备-设备连线故障','主设备-天馈线故障',
                         '主设备-信源问题','主设备-硬件故障'])
res['工单编号'] = test['工单编号'].values
res = res.sort_values('工单编号')


# In[57]:


res[['工单编号','传输系统-传输设备','传输系统-光缆故障','传输系统-其他原因',
                         '动力环境-UPS','动力环境-电力部门供电','动力环境-电源线路故障',
                         '动力环境-动环监控系统','动力环境-动力环境故障','动力环境-高低压设备',
                         '动力环境-环境','动力环境-开关电源','其他-误告警或自动恢复',
                         '人为操作-告警测试','人为操作-工程施工','人为操作-物业原因',
                         '主设备-参数配置异常','主设备-其他','主设备-软件故障',
                         '主设备-设备复位问题','主设备-设备连线故障','主设备-天馈线故障',
                         '主设备-信源问题','主设备-硬件故障']].to_csv('../xgb_704test.csv',index=None,encoding='gbk')


# In[ ]:




