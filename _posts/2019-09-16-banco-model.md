---
title: "Banco Data Model"
date: 2019-09-16T15:34:30-04:00
categories:
  - blog
tags:
  - machine learning
  - python
  - scikit-learn
---

```python
import pandas as pd
from pandas import np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn
```


```python
X_train = pd.read_csv('data/train.csv', compression="gzip").drop('USER_ID', axis=1)
X_test = pd.read_csv('data/test.csv', compression="gzip").drop('USER_ID', axis=1)
y_train = X_train.pop('target')
```


```python
X_train.shape, X_test.shape, y_train.shape
```




    ((11529, 1961), (11676, 1961), (11529,))




```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAGE_46</th>
      <th>PAGE_386</th>
      <th>PAGE_556</th>
      <th>PAGE_601</th>
      <th>ON_SITE_SEARCH_TERM_112</th>
      <th>PAGE_962</th>
      <th>PAGE_1099</th>
      <th>PAGE_263</th>
      <th>CONTENT_CATEGORY_57</th>
      <th>PAGE_615</th>
      <th>...</th>
      <th>PAGE_77</th>
      <th>PAGE_143</th>
      <th>PAGE_1081</th>
      <th>PAGE_8</th>
      <th>PAGE_1281</th>
      <th>ON_SITE_SEARCH_TERM_29</th>
      <th>PAGE_561</th>
      <th>ON_SITE_SEARCH_TERM_287</th>
      <th>PAGE_767</th>
      <th>PAGE_730</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.000204</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000204</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.000000</td>
      <td>0.001165</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.004077</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.000000</td>
      <td>0.000404</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000404</td>
      <td>0.000807</td>
      <td>0.0</td>
      <td>0.000404</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1961 columns</p>
</div>




```python
X_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAGE_46</th>
      <th>PAGE_386</th>
      <th>PAGE_556</th>
      <th>PAGE_601</th>
      <th>ON_SITE_SEARCH_TERM_112</th>
      <th>PAGE_962</th>
      <th>PAGE_1099</th>
      <th>PAGE_263</th>
      <th>CONTENT_CATEGORY_57</th>
      <th>PAGE_615</th>
      <th>...</th>
      <th>PAGE_77</th>
      <th>PAGE_143</th>
      <th>PAGE_1081</th>
      <th>PAGE_8</th>
      <th>PAGE_1281</th>
      <th>ON_SITE_SEARCH_TERM_29</th>
      <th>PAGE_561</th>
      <th>ON_SITE_SEARCH_TERM_287</th>
      <th>PAGE_767</th>
      <th>PAGE_730</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.000155</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000155</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000845</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.002956</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000422</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.000000</td>
      <td>0.000322</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000322</td>
      <td>0.000644</td>
      <td>0.0</td>
      <td>0.000322</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1961 columns</p>
</div>




```python
cols = []
found = 0
for c in X_train.columns:
    # print(c)
    val = X_train[c].value_counts(normalize=True).values[0]
    if val >= 0.90:
        found += 1
        print(found)
        cols.append(c)
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27
    28
    29
    30
    31
    32
    33
    34
    35
    36
    37
    38
    39
    40
    41
    42
    43
    44
    45
    46
    47
    48
    49
    50
    51
    52
    53
    54
    55
    56
    57
    58
    59
    60
    61
    62
    63
    64
    65
    66
    67
    68
    69
    70
    71
    72
    73
    74
    75
    76
    77
    78
    79
    80
    81
    82
    83
    84
    85
    86
    87
    88
    89
    90
    91
    92
    93
    94
    95
    96
    97
    98
    99
    100
    101
    102
    103
    104
    105
    106
    107
    108
    109
    110
    111
    112
    113
    114
    115
    116
    117
    118
    119
    120
    121
    122
    123
    124
    125
    126
    127
    128
    129
    130
    131
    132
    133
    134
    135
    136
    137
    138
    139
    140
    141
    142
    143
    144
    145
    146
    147
    148
    149
    150
    151
    152
    153
    154
    155
    156
    157
    158
    159
    160
    161
    162
    163
    164
    165
    166
    167
    168
    169
    170
    171
    172
    173
    174
    175
    176
    177
    178
    179
    180
    181
    182
    183
    184
    185
    186
    187
    188
    189
    190
    191
    192
    193
    194
    195
    196
    197
    198
    199
    200
    201
    202
    203
    204
    205
    206
    207
    208
    209
    210
    211
    212
    213
    214
    215
    216
    217
    218
    219
    220
    221
    222
    223
    224
    225
    226
    227
    228
    229
    230
    231
    232
    233
    234
    235
    236
    237
    238
    239
    240
    241
    242
    243
    244
    245
    246
    247
    248
    249
    250
    251
    252
    253
    254
    255
    256
    257
    258
    259
    260
    261
    262
    263
    264
    265
    266
    267
    268
    269
    270
    271
    272
    273
    274
    275
    276
    277
    278
    279
    280
    281
    282
    283
    284
    285
    286
    287
    288
    289



```python
cols_to_keep = X_train.columns.difference(cols)
```


```python
X_train = X_train[cols_to_keep]
X_test = X_test[cols_to_keep]
```


```python
X_train.shape, X_train.shape
```




    ((11529, 476), (11529, 476))




```python
from sklearn.ensemble import RandomForestClassifier
```


```python
bst = RandomForestClassifier()
```


```python
bst.fit(X_train, y_train)
```

    /anaconda3/envs/pyenv/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)





    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)




```python
plt.figure(figsize=(16,6))
plt.plot(bst.feature_importances_)
plt.xticks(np.arange(X_train.shape[1]), X_train.columns.tolist(), rotation=90);
```


![png](/blog/lgb_files/output_12_0.png)



```python
plt.figure(figsize=(16,6))
plt.plot(X_train.PAGE_286, '.')
```




    [<matplotlib.lines.Line2D at 0x1a25d1c320>]




![png](/blog/lgb_files/output_13_1.png)



```python
plt.figure(figsize=(16,6))
plt.scatter(range(len(X_train)), X_train.PAGE_286, c=y_train)
```




    <matplotlib.collections.PathCollection at 0x1a27aff630>




![png](/blog/lgb_files/output_14_1.png)



```python
x1 = np.random.randn(100)
x2 = 1 - x1
```


```python
plt.scatter(x1, x2)
```




    <matplotlib.collections.PathCollection at 0x1a255e5780>




![png](/blog/lgb_files/output_16_1.png)



```python
plt.figure(figsize=(16,6))
X_train.mean().sort_values().plot(style='.')
plt.xticks(rotation = 90)
```




    (array([   0.,  250.,  500.,  750., 1000., 1250., 1500., 1750., 2000.,
            2250.]), <a list of 10 Text xticklabel objects>)




![png](/blog/lgb_files/output_17_1.png)



```python
import lightgbm as lgb
from sklearn import preprocessing, metrics, ensemble, neighbors, linear_model, tree, model_selection
from sklearn.model_selection import KFold, StratifiedKFold
```


```python
def runLGB(train_X, train_y, test_X, test_y=None, test_X2=None, feature_names=None, seed_val=0, rounds=500, dep=6, eta=0.05):
	params = {}
	params["objective"] = "binary"
	params['metric'] = 'auc'
	params["max_depth"] = dep
	params["min_data_in_leaf"] = 20
	params["learning_rate"] = eta
	params["bagging_fraction"] = 0.8
	params["feature_fraction"] = 0.8
	params["bagging_freq"] = 5
	params["bagging_seed"] = seed_val
	params["verbosity"] = 0
	num_rounds = rounds

	plst = list(params.items())
	lgtrain = lgb.Dataset(train_X, label=train_y)

	if test_y is not None:
		lgtest = lgb.Dataset(test_X, label=test_y)
		model = lgb.train(params, lgtrain, num_rounds, valid_sets=[lgtest], early_stopping_rounds=100, verbose_eval=20)
	else:
		lgtest = lgb.DMatrix(test_X)
		model = lgb.train(params, lgtrain, num_rounds)

	pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
	pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)

	loss = 0
	if test_y is not None:
		loss = metrics.roc_auc_score(test_y, pred_test_y)
		print (loss)
		return pred_test_y, loss, pred_test_y2
	else:
		return pred_test_y, loss, pred_test_y2
```


```python
print ("Model building..")

for model_name in ["LGB1"]:

    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2018)
    cv_scores = []
    pred_test_full = 0
    pred_val_full = np.zeros(X_train.shape[0])
    for dev_index, val_index in kf.split(X_train):
        dev_X, val_X = X_train.iloc[dev_index,:], X_train.iloc[val_index,:]
        dev_y, val_y = y_train[dev_index], y_train[val_index]

        if model_name == "XGB1":
            pred_val, loss, pred_test = runXGB(dev_X, dev_y, val_X, val_y, X_test, rounds=5000, dep=8, feature_names=dev_X.columns.tolist())
        elif model_name == "LGB1":
            pred_val, loss, pred_test = runLGB(dev_X, dev_y, val_X, val_y, X_test, rounds=5000, dep=8)
        pred_val_full[val_index] = pred_val
        pred_test_full = pred_test_full + pred_test
        cv_scores.append(loss)
        print ('cv scores:', cv_scores)
    pred_test_full /= 5.
    print ('roc:', metrics.roc_auc_score(y_train, pred_val_full))
```

    Model building..
    Training until validation scores don't improve for 100 rounds.
    [20]	valid_0's auc: 0.822282
    [40]	valid_0's auc: 0.826048
    [60]	valid_0's auc: 0.826306
    [80]	valid_0's auc: 0.826656
    [100]	valid_0's auc: 0.824796
    [120]	valid_0's auc: 0.822485
    [140]	valid_0's auc: 0.816789
    Early stopping, best iteration is:
    [52]	valid_0's auc: 0.83123
    0.8312300760720277
    cv scores: [0.8312300760720277]
    Training until validation scores don't improve for 100 rounds.
    [20]	valid_0's auc: 0.842594
    [40]	valid_0's auc: 0.844892
    [60]	valid_0's auc: 0.840735
    [80]	valid_0's auc: 0.841112
    [100]	valid_0's auc: 0.842307
    [120]	valid_0's auc: 0.840006
    Early stopping, best iteration is:
    [28]	valid_0's auc: 0.851708
    0.8517077543702376
    cv scores: [0.8312300760720277, 0.8517077543702376]
    Training until validation scores don't improve for 100 rounds.
    [20]	valid_0's auc: 0.845353
    [40]	valid_0's auc: 0.848135
    [60]	valid_0's auc: 0.855476
    [80]	valid_0's auc: 0.855558
    [100]	valid_0's auc: 0.853304
    [120]	valid_0's auc: 0.853924
    [140]	valid_0's auc: 0.853877
    [160]	valid_0's auc: 0.849528
    Early stopping, best iteration is:
    [67]	valid_0's auc: 0.856007
    0.8560066084493745
    cv scores: [0.8312300760720277, 0.8517077543702376, 0.8560066084493745]
    Training until validation scores don't improve for 100 rounds.
    [20]	valid_0's auc: 0.834798
    [40]	valid_0's auc: 0.841903
    [60]	valid_0's auc: 0.839563
    [80]	valid_0's auc: 0.838975
    [100]	valid_0's auc: 0.841619
    [120]	valid_0's auc: 0.84141
    Early stopping, best iteration is:
    [34]	valid_0's auc: 0.845089
    0.8450892857142858
    cv scores: [0.8312300760720277, 0.8517077543702376, 0.8560066084493745, 0.8450892857142858]
    Training until validation scores don't improve for 100 rounds.
    [20]	valid_0's auc: 0.84436
    [40]	valid_0's auc: 0.856457
    [60]	valid_0's auc: 0.860475
    [80]	valid_0's auc: 0.8633
    [100]	valid_0's auc: 0.860432
    [120]	valid_0's auc: 0.858523
    [140]	valid_0's auc: 0.857216
    [160]	valid_0's auc: 0.855915
    [180]	valid_0's auc: 0.855254
    Early stopping, best iteration is:
    [83]	valid_0's auc: 0.863848
    0.8638478305661891
    cv scores: [0.8312300760720277, 0.8517077543702376, 0.8560066084493745, 0.8450892857142858, 0.8638478305661891]
    roc: 0.838741675243339



```python
out_df = pd.DataFrame({"UCIC_ID":test_id})
out_df["Responders"] = pred_test_full
out_df.to_csv("./meta_models/test/pred_test_v5_"+model_name+".csv", index=False)

out_df = pd.DataFrame({"UCIC_ID":train_id})
out_df["Responders"] = pred_val_full
out_df.to_csv("./meta_models/val/pred_val_v5_"+model_name+".csv", index=False)
```


```python
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
```


```python
fi = []
test_probs = []
i = 0
categorical_features = []
for train_idx, valid_idx in model_selection.KFold(n_splits=3, shuffle=True).split(X_train):
    i += 1
    Xt = X_train.iloc[train_idx]
    yt = y_train.iloc[train_idx]

    Xv = X_train.iloc[valid_idx]
    yv = y_train.iloc[valid_idx]
    
    learner = LGBMClassifier(n_estimators=10000)
    learner.fit(Xt, yt,  early_stopping_rounds=50, eval_metric="auc",
                eval_set=[(Xt, yt), (Xv, yv)])
    
    test_probs.append(pd.Series(learner.predict_proba(X_test)[:, -1],
                                index=X_test.index, name="fold_" + str(i)))
    fi.append(pd.Series(learner.feature_importances_ / learner.feature_importances_.sum(), index=Xt.columns))
    print('*'*100)

test_probs = pd.concat(test_probs, axis=1).mean(axis=1)
test_probs.index.name="USER_ID"
test_probs.name="SCORE"
fi = pd.concat(fi, axis=1).mean(axis=1)
```

    [1]	training's binary_logloss: 0.126145	training's auc: 0.858384	valid_1's binary_logloss: 0.139452	valid_1's auc: 0.719383
    Training until validation scores don't improve for 50 rounds.
    [2]	training's binary_logloss: 0.115389	training's auc: 0.902577	valid_1's binary_logloss: 0.136807	valid_1's auc: 0.791728
    [3]	training's binary_logloss: 0.106853	training's auc: 0.913021	valid_1's binary_logloss: 0.134199	valid_1's auc: 0.800686
    [4]	training's binary_logloss: 0.0998989	training's auc: 0.927987	valid_1's binary_logloss: 0.132153	valid_1's auc: 0.796186
    [5]	training's binary_logloss: 0.0940303	training's auc: 0.934045	valid_1's binary_logloss: 0.130979	valid_1's auc: 0.796161
    [6]	training's binary_logloss: 0.0890072	training's auc: 0.940559	valid_1's binary_logloss: 0.129556	valid_1's auc: 0.806731
    [7]	training's binary_logloss: 0.0844893	training's auc: 0.944346	valid_1's binary_logloss: 0.128602	valid_1's auc: 0.80852
    [8]	training's binary_logloss: 0.0806133	training's auc: 0.949222	valid_1's binary_logloss: 0.127904	valid_1's auc: 0.807907
    [9]	training's binary_logloss: 0.0768307	training's auc: 0.959412	valid_1's binary_logloss: 0.127236	valid_1's auc: 0.817692
    [10]	training's binary_logloss: 0.0732959	training's auc: 0.968213	valid_1's binary_logloss: 0.126305	valid_1's auc: 0.821131
    [11]	training's binary_logloss: 0.0702006	training's auc: 0.969658	valid_1's binary_logloss: 0.125961	valid_1's auc: 0.817581
    [12]	training's binary_logloss: 0.0673004	training's auc: 0.97194	valid_1's binary_logloss: 0.125426	valid_1's auc: 0.819899
    [13]	training's binary_logloss: 0.0647311	training's auc: 0.974895	valid_1's binary_logloss: 0.125178	valid_1's auc: 0.820049
    [14]	training's binary_logloss: 0.0621246	training's auc: 0.977829	valid_1's binary_logloss: 0.124847	valid_1's auc: 0.820033
    [15]	training's binary_logloss: 0.0596625	training's auc: 0.980717	valid_1's binary_logloss: 0.124584	valid_1's auc: 0.822919
    [16]	training's binary_logloss: 0.0571292	training's auc: 0.98688	valid_1's binary_logloss: 0.124134	valid_1's auc: 0.823306
    [17]	training's binary_logloss: 0.0548319	training's auc: 0.991661	valid_1's binary_logloss: 0.12378	valid_1's auc: 0.823833
    [18]	training's binary_logloss: 0.0527287	training's auc: 0.994754	valid_1's binary_logloss: 0.12391	valid_1's auc: 0.821777
    [19]	training's binary_logloss: 0.0507124	training's auc: 0.995772	valid_1's binary_logloss: 0.124074	valid_1's auc: 0.822309
    [20]	training's binary_logloss: 0.0490284	training's auc: 0.997262	valid_1's binary_logloss: 0.124065	valid_1's auc: 0.822455
    [21]	training's binary_logloss: 0.047192	training's auc: 0.998439	valid_1's binary_logloss: 0.12451	valid_1's auc: 0.820813
    [22]	training's binary_logloss: 0.0454424	training's auc: 0.998923	valid_1's binary_logloss: 0.124892	valid_1's auc: 0.819708
    [23]	training's binary_logloss: 0.0437388	training's auc: 0.999402	valid_1's binary_logloss: 0.125297	valid_1's auc: 0.818569
    [24]	training's binary_logloss: 0.0421955	training's auc: 0.999647	valid_1's binary_logloss: 0.125505	valid_1's auc: 0.82001
    [25]	training's binary_logloss: 0.0406365	training's auc: 0.999796	valid_1's binary_logloss: 0.125293	valid_1's auc: 0.821076
    [26]	training's binary_logloss: 0.0393583	training's auc: 0.999844	valid_1's binary_logloss: 0.125782	valid_1's auc: 0.819628
    [27]	training's binary_logloss: 0.0381513	training's auc: 0.999882	valid_1's binary_logloss: 0.1258	valid_1's auc: 0.819997
    [28]	training's binary_logloss: 0.0369115	training's auc: 0.999908	valid_1's binary_logloss: 0.126033	valid_1's auc: 0.818347
    [29]	training's binary_logloss: 0.0358091	training's auc: 0.999925	valid_1's binary_logloss: 0.1265	valid_1's auc: 0.817562
    [30]	training's binary_logloss: 0.0345994	training's auc: 0.999949	valid_1's binary_logloss: 0.126369	valid_1's auc: 0.819908
    [31]	training's binary_logloss: 0.0335735	training's auc: 0.99996	valid_1's binary_logloss: 0.126647	valid_1's auc: 0.819705
    [32]	training's binary_logloss: 0.0324522	training's auc: 0.999974	valid_1's binary_logloss: 0.126458	valid_1's auc: 0.821713
    [33]	training's binary_logloss: 0.03138	training's auc: 0.999985	valid_1's binary_logloss: 0.12658	valid_1's auc: 0.821923
    [34]	training's binary_logloss: 0.0304929	training's auc: 0.99999	valid_1's binary_logloss: 0.126826	valid_1's auc: 0.824159
    [35]	training's binary_logloss: 0.0294763	training's auc: 0.999993	valid_1's binary_logloss: 0.127328	valid_1's auc: 0.824307
    [36]	training's binary_logloss: 0.0285311	training's auc: 0.999997	valid_1's binary_logloss: 0.127341	valid_1's auc: 0.824803
    [37]	training's binary_logloss: 0.0275564	training's auc: 0.999998	valid_1's binary_logloss: 0.127536	valid_1's auc: 0.82557
    [38]	training's binary_logloss: 0.0266944	training's auc: 0.999998	valid_1's binary_logloss: 0.127763	valid_1's auc: 0.826581
    [39]	training's binary_logloss: 0.0258317	training's auc: 1	valid_1's binary_logloss: 0.128331	valid_1's auc: 0.825271
    [40]	training's binary_logloss: 0.0250127	training's auc: 1	valid_1's binary_logloss: 0.128897	valid_1's auc: 0.825966
    [41]	training's binary_logloss: 0.0242378	training's auc: 1	valid_1's binary_logloss: 0.129387	valid_1's auc: 0.825159
    [42]	training's binary_logloss: 0.0234938	training's auc: 1	valid_1's binary_logloss: 0.129848	valid_1's auc: 0.824617
    [43]	training's binary_logloss: 0.022696	training's auc: 1	valid_1's binary_logloss: 0.130131	valid_1's auc: 0.823775
    [44]	training's binary_logloss: 0.0219963	training's auc: 1	valid_1's binary_logloss: 0.130387	valid_1's auc: 0.825231
    [45]	training's binary_logloss: 0.0212978	training's auc: 1	valid_1's binary_logloss: 0.131033	valid_1's auc: 0.824203
    [46]	training's binary_logloss: 0.0206478	training's auc: 1	valid_1's binary_logloss: 0.131483	valid_1's auc: 0.823324
    [47]	training's binary_logloss: 0.0199815	training's auc: 1	valid_1's binary_logloss: 0.132179	valid_1's auc: 0.821955
    [48]	training's binary_logloss: 0.0193163	training's auc: 1	valid_1's binary_logloss: 0.132791	valid_1's auc: 0.822688
    [49]	training's binary_logloss: 0.0187318	training's auc: 1	valid_1's binary_logloss: 0.133021	valid_1's auc: 0.823913
    [50]	training's binary_logloss: 0.0181585	training's auc: 1	valid_1's binary_logloss: 0.133582	valid_1's auc: 0.823439
    [51]	training's binary_logloss: 0.017649	training's auc: 1	valid_1's binary_logloss: 0.134033	valid_1's auc: 0.823854
    [52]	training's binary_logloss: 0.0171129	training's auc: 1	valid_1's binary_logloss: 0.134959	valid_1's auc: 0.822176
    [53]	training's binary_logloss: 0.0165774	training's auc: 1	valid_1's binary_logloss: 0.135236	valid_1's auc: 0.822595
    [54]	training's binary_logloss: 0.0160674	training's auc: 1	valid_1's binary_logloss: 0.135685	valid_1's auc: 0.822905
    [55]	training's binary_logloss: 0.0156756	training's auc: 1	valid_1's binary_logloss: 0.136363	valid_1's auc: 0.822299
    [56]	training's binary_logloss: 0.0151693	training's auc: 1	valid_1's binary_logloss: 0.136792	valid_1's auc: 0.822417
    [57]	training's binary_logloss: 0.0147126	training's auc: 1	valid_1's binary_logloss: 0.136809	valid_1's auc: 0.823663
    [58]	training's binary_logloss: 0.0142422	training's auc: 1	valid_1's binary_logloss: 0.137147	valid_1's auc: 0.824515
    [59]	training's binary_logloss: 0.0138362	training's auc: 1	valid_1's binary_logloss: 0.137844	valid_1's auc: 0.823763
    [60]	training's binary_logloss: 0.0134268	training's auc: 1	valid_1's binary_logloss: 0.138406	valid_1's auc: 0.822932
    [61]	training's binary_logloss: 0.0130646	training's auc: 1	valid_1's binary_logloss: 0.139329	valid_1's auc: 0.821019
    [62]	training's binary_logloss: 0.0126739	training's auc: 1	valid_1's binary_logloss: 0.139533	valid_1's auc: 0.821862
    [63]	training's binary_logloss: 0.0123332	training's auc: 1	valid_1's binary_logloss: 0.140541	valid_1's auc: 0.820716
    [64]	training's binary_logloss: 0.0119706	training's auc: 1	valid_1's binary_logloss: 0.140784	valid_1's auc: 0.821955
    [65]	training's binary_logloss: 0.0115689	training's auc: 1	valid_1's binary_logloss: 0.141384	valid_1's auc: 0.822311
    [66]	training's binary_logloss: 0.0112473	training's auc: 1	valid_1's binary_logloss: 0.141913	valid_1's auc: 0.821868
    [67]	training's binary_logloss: 0.010899	training's auc: 1	valid_1's binary_logloss: 0.142749	valid_1's auc: 0.820317
    Early stopping, best iteration is:
    [17]	training's binary_logloss: 0.0548319	training's auc: 0.991661	valid_1's binary_logloss: 0.12378	valid_1's auc: 0.823833
    ****************************************************************************************************
    [1]	training's binary_logloss: 0.126331	training's auc: 0.86006	valid_1's binary_logloss: 0.141773	valid_1's auc: 0.688795
    Training until validation scores don't improve for 50 rounds.
    [2]	training's binary_logloss: 0.115392	training's auc: 0.894527	valid_1's binary_logloss: 0.138845	valid_1's auc: 0.775107
    [3]	training's binary_logloss: 0.107267	training's auc: 0.909139	valid_1's binary_logloss: 0.136317	valid_1's auc: 0.800862
    [4]	training's binary_logloss: 0.100222	training's auc: 0.919017	valid_1's binary_logloss: 0.134387	valid_1's auc: 0.802316
    [5]	training's binary_logloss: 0.0939713	training's auc: 0.934957	valid_1's binary_logloss: 0.133776	valid_1's auc: 0.803204
    [6]	training's binary_logloss: 0.0887733	training's auc: 0.942664	valid_1's binary_logloss: 0.132936	valid_1's auc: 0.802072
    [7]	training's binary_logloss: 0.0842822	training's auc: 0.948386	valid_1's binary_logloss: 0.13204	valid_1's auc: 0.802987
    [8]	training's binary_logloss: 0.0802062	training's auc: 0.958432	valid_1's binary_logloss: 0.131325	valid_1's auc: 0.814213
    [9]	training's binary_logloss: 0.0763598	training's auc: 0.960917	valid_1's binary_logloss: 0.130487	valid_1's auc: 0.816669
    [10]	training's binary_logloss: 0.0729061	training's auc: 0.967657	valid_1's binary_logloss: 0.129227	valid_1's auc: 0.822097
    [11]	training's binary_logloss: 0.0695367	training's auc: 0.970097	valid_1's binary_logloss: 0.129342	valid_1's auc: 0.822155
    [12]	training's binary_logloss: 0.0667035	training's auc: 0.974941	valid_1's binary_logloss: 0.128444	valid_1's auc: 0.828191
    [13]	training's binary_logloss: 0.0641782	training's auc: 0.979961	valid_1's binary_logloss: 0.127316	valid_1's auc: 0.830489
    [14]	training's binary_logloss: 0.0612904	training's auc: 0.98538	valid_1's binary_logloss: 0.127367	valid_1's auc: 0.825965
    [15]	training's binary_logloss: 0.0590564	training's auc: 0.985914	valid_1's binary_logloss: 0.126906	valid_1's auc: 0.830161
    [16]	training's binary_logloss: 0.056681	training's auc: 0.990071	valid_1's binary_logloss: 0.127065	valid_1's auc: 0.827053
    [17]	training's binary_logloss: 0.0545205	training's auc: 0.993111	valid_1's binary_logloss: 0.126752	valid_1's auc: 0.827281
    [18]	training's binary_logloss: 0.0526384	training's auc: 0.994904	valid_1's binary_logloss: 0.126751	valid_1's auc: 0.830441
    [19]	training's binary_logloss: 0.0505455	training's auc: 0.996805	valid_1's binary_logloss: 0.126315	valid_1's auc: 0.83561
    [20]	training's binary_logloss: 0.0485214	training's auc: 0.998347	valid_1's binary_logloss: 0.126179	valid_1's auc: 0.838604
    [21]	training's binary_logloss: 0.0467807	training's auc: 0.99891	valid_1's binary_logloss: 0.125962	valid_1's auc: 0.839363
    [22]	training's binary_logloss: 0.0451167	training's auc: 0.999246	valid_1's binary_logloss: 0.125816	valid_1's auc: 0.840019
    [23]	training's binary_logloss: 0.0435827	training's auc: 0.999483	valid_1's binary_logloss: 0.12592	valid_1's auc: 0.839787
    [24]	training's binary_logloss: 0.0420736	training's auc: 0.999704	valid_1's binary_logloss: 0.1259	valid_1's auc: 0.84227
    [25]	training's binary_logloss: 0.040651	training's auc: 0.99977	valid_1's binary_logloss: 0.125857	valid_1's auc: 0.841908
    [26]	training's binary_logloss: 0.0393096	training's auc: 0.999827	valid_1's binary_logloss: 0.125906	valid_1's auc: 0.842037
    [27]	training's binary_logloss: 0.0380601	training's auc: 0.999875	valid_1's binary_logloss: 0.125896	valid_1's auc: 0.842431
    [28]	training's binary_logloss: 0.0367597	training's auc: 0.999905	valid_1's binary_logloss: 0.126139	valid_1's auc: 0.842332
    [29]	training's binary_logloss: 0.0355762	training's auc: 0.999939	valid_1's binary_logloss: 0.126352	valid_1's auc: 0.842761
    [30]	training's binary_logloss: 0.0345102	training's auc: 0.999949	valid_1's binary_logloss: 0.126577	valid_1's auc: 0.841375
    [31]	training's binary_logloss: 0.0334145	training's auc: 0.999967	valid_1's binary_logloss: 0.12685	valid_1's auc: 0.841117
    [32]	training's binary_logloss: 0.0321739	training's auc: 0.999984	valid_1's binary_logloss: 0.1271	valid_1's auc: 0.842203
    [33]	training's binary_logloss: 0.0310936	training's auc: 0.999988	valid_1's binary_logloss: 0.127382	valid_1's auc: 0.842118
    [34]	training's binary_logloss: 0.0300559	training's auc: 0.999994	valid_1's binary_logloss: 0.127792	valid_1's auc: 0.84107
    [35]	training's binary_logloss: 0.0290591	training's auc: 0.999996	valid_1's binary_logloss: 0.128144	valid_1's auc: 0.841678
    [36]	training's binary_logloss: 0.0281319	training's auc: 0.999998	valid_1's binary_logloss: 0.128487	valid_1's auc: 0.841169
    [37]	training's binary_logloss: 0.0272297	training's auc: 0.999999	valid_1's binary_logloss: 0.128513	valid_1's auc: 0.841538
    [38]	training's binary_logloss: 0.0262928	training's auc: 1	valid_1's binary_logloss: 0.128527	valid_1's auc: 0.842143
    [39]	training's binary_logloss: 0.0254499	training's auc: 1	valid_1's binary_logloss: 0.128628	valid_1's auc: 0.842218
    [40]	training's binary_logloss: 0.024647	training's auc: 1	valid_1's binary_logloss: 0.129111	valid_1's auc: 0.841093
    [41]	training's binary_logloss: 0.0238641	training's auc: 1	valid_1's binary_logloss: 0.129163	valid_1's auc: 0.843065
    [42]	training's binary_logloss: 0.0231215	training's auc: 1	valid_1's binary_logloss: 0.129411	valid_1's auc: 0.842445
    [43]	training's binary_logloss: 0.0224215	training's auc: 1	valid_1's binary_logloss: 0.130148	valid_1's auc: 0.841227
    [44]	training's binary_logloss: 0.0217507	training's auc: 1	valid_1's binary_logloss: 0.130677	valid_1's auc: 0.840508
    [45]	training's binary_logloss: 0.0210707	training's auc: 1	valid_1's binary_logloss: 0.130872	valid_1's auc: 0.841115
    [46]	training's binary_logloss: 0.0204001	training's auc: 1	valid_1's binary_logloss: 0.131421	valid_1's auc: 0.840817
    [47]	training's binary_logloss: 0.0198442	training's auc: 1	valid_1's binary_logloss: 0.131893	valid_1's auc: 0.840883
    [48]	training's binary_logloss: 0.019278	training's auc: 1	valid_1's binary_logloss: 0.132044	valid_1's auc: 0.841188
    [49]	training's binary_logloss: 0.0187957	training's auc: 1	valid_1's binary_logloss: 0.132464	valid_1's auc: 0.839949
    [50]	training's binary_logloss: 0.0181738	training's auc: 1	valid_1's binary_logloss: 0.133156	valid_1's auc: 0.839483
    [51]	training's binary_logloss: 0.0175961	training's auc: 1	valid_1's binary_logloss: 0.133437	valid_1's auc: 0.839274
    [52]	training's binary_logloss: 0.0170688	training's auc: 1	valid_1's binary_logloss: 0.134146	valid_1's auc: 0.838895
    [53]	training's binary_logloss: 0.0166069	training's auc: 1	valid_1's binary_logloss: 0.134403	valid_1's auc: 0.838677
    [54]	training's binary_logloss: 0.0161648	training's auc: 1	valid_1's binary_logloss: 0.135108	valid_1's auc: 0.837525
    [55]	training's binary_logloss: 0.0156756	training's auc: 1	valid_1's binary_logloss: 0.135223	valid_1's auc: 0.837784
    [56]	training's binary_logloss: 0.0151777	training's auc: 1	valid_1's binary_logloss: 0.135692	valid_1's auc: 0.837208
    [57]	training's binary_logloss: 0.0146566	training's auc: 1	valid_1's binary_logloss: 0.136196	valid_1's auc: 0.83633
    [58]	training's binary_logloss: 0.0142951	training's auc: 1	valid_1's binary_logloss: 0.136514	valid_1's auc: 0.837212
    [59]	training's binary_logloss: 0.0138408	training's auc: 1	valid_1's binary_logloss: 0.137236	valid_1's auc: 0.836587
    [60]	training's binary_logloss: 0.0133977	training's auc: 1	valid_1's binary_logloss: 0.13777	valid_1's auc: 0.835942
    [61]	training's binary_logloss: 0.0130507	training's auc: 1	valid_1's binary_logloss: 0.138179	valid_1's auc: 0.836514
    [62]	training's binary_logloss: 0.0126757	training's auc: 1	valid_1's binary_logloss: 0.13865	valid_1's auc: 0.836986
    [63]	training's binary_logloss: 0.0122888	training's auc: 1	valid_1's binary_logloss: 0.139294	valid_1's auc: 0.836705
    [64]	training's binary_logloss: 0.0119334	training's auc: 1	valid_1's binary_logloss: 0.139803	valid_1's auc: 0.836276
    [65]	training's binary_logloss: 0.0116717	training's auc: 1	valid_1's binary_logloss: 0.140277	valid_1's auc: 0.836133
    [66]	training's binary_logloss: 0.0113593	training's auc: 1	valid_1's binary_logloss: 0.140845	valid_1's auc: 0.835938
    [67]	training's binary_logloss: 0.0109989	training's auc: 1	valid_1's binary_logloss: 0.141565	valid_1's auc: 0.836087
    [68]	training's binary_logloss: 0.0106599	training's auc: 1	valid_1's binary_logloss: 0.141738	valid_1's auc: 0.836669
    [69]	training's binary_logloss: 0.0103586	training's auc: 1	valid_1's binary_logloss: 0.142621	valid_1's auc: 0.83611
    [70]	training's binary_logloss: 0.0100796	training's auc: 1	valid_1's binary_logloss: 0.143238	valid_1's auc: 0.836141
    [71]	training's binary_logloss: 0.00982507	training's auc: 1	valid_1's binary_logloss: 0.143421	valid_1's auc: 0.837332
    [72]	training's binary_logloss: 0.00954214	training's auc: 1	valid_1's binary_logloss: 0.143905	valid_1's auc: 0.837463
    Early stopping, best iteration is:
    [22]	training's binary_logloss: 0.0451167	training's auc: 0.999246	valid_1's binary_logloss: 0.125816	valid_1's auc: 0.840019
    ****************************************************************************************************
    [1]	training's binary_logloss: 0.126282	training's auc: 0.848134	valid_1's binary_logloss: 0.144133	valid_1's auc: 0.75482
    Training until validation scores don't improve for 50 rounds.
    [2]	training's binary_logloss: 0.114914	training's auc: 0.871369	valid_1's binary_logloss: 0.140947	valid_1's auc: 0.774032
    [3]	training's binary_logloss: 0.106599	training's auc: 0.889167	valid_1's binary_logloss: 0.138631	valid_1's auc: 0.814759
    [4]	training's binary_logloss: 0.0997407	training's auc: 0.914177	valid_1's binary_logloss: 0.137023	valid_1's auc: 0.808746
    [5]	training's binary_logloss: 0.0941895	training's auc: 0.930399	valid_1's binary_logloss: 0.135949	valid_1's auc: 0.821575
    [6]	training's binary_logloss: 0.0886509	training's auc: 0.946937	valid_1's binary_logloss: 0.134401	valid_1's auc: 0.830349
    [7]	training's binary_logloss: 0.0841347	training's auc: 0.958235	valid_1's binary_logloss: 0.133283	valid_1's auc: 0.828593
    [8]	training's binary_logloss: 0.0799436	training's auc: 0.967969	valid_1's binary_logloss: 0.132694	valid_1's auc: 0.832146
    [9]	training's binary_logloss: 0.0762107	training's auc: 0.970817	valid_1's binary_logloss: 0.13251	valid_1's auc: 0.832498
    [10]	training's binary_logloss: 0.0725432	training's auc: 0.976609	valid_1's binary_logloss: 0.131188	valid_1's auc: 0.841586
    [11]	training's binary_logloss: 0.0695386	training's auc: 0.978622	valid_1's binary_logloss: 0.130446	valid_1's auc: 0.842788
    [12]	training's binary_logloss: 0.0664893	training's auc: 0.981263	valid_1's binary_logloss: 0.129943	valid_1's auc: 0.843686
    [13]	training's binary_logloss: 0.063499	training's auc: 0.985564	valid_1's binary_logloss: 0.129263	valid_1's auc: 0.843438
    [14]	training's binary_logloss: 0.0609046	training's auc: 0.988434	valid_1's binary_logloss: 0.128301	valid_1's auc: 0.845644
    [15]	training's binary_logloss: 0.0585628	training's auc: 0.989409	valid_1's binary_logloss: 0.128031	valid_1's auc: 0.845241
    [16]	training's binary_logloss: 0.0557697	training's auc: 0.99496	valid_1's binary_logloss: 0.128023	valid_1's auc: 0.845236
    [17]	training's binary_logloss: 0.0536368	training's auc: 0.995358	valid_1's binary_logloss: 0.127602	valid_1's auc: 0.848545
    [18]	training's binary_logloss: 0.051504	training's auc: 0.996457	valid_1's binary_logloss: 0.127456	valid_1's auc: 0.850324
    [19]	training's binary_logloss: 0.0495199	training's auc: 0.996878	valid_1's binary_logloss: 0.127402	valid_1's auc: 0.849346
    [20]	training's binary_logloss: 0.0475493	training's auc: 0.997296	valid_1's binary_logloss: 0.127012	valid_1's auc: 0.850232
    [21]	training's binary_logloss: 0.0457628	training's auc: 0.998147	valid_1's binary_logloss: 0.126746	valid_1's auc: 0.849746
    [22]	training's binary_logloss: 0.044046	training's auc: 0.998634	valid_1's binary_logloss: 0.126805	valid_1's auc: 0.848831
    [23]	training's binary_logloss: 0.0424157	training's auc: 0.99921	valid_1's binary_logloss: 0.126627	valid_1's auc: 0.848772
    [24]	training's binary_logloss: 0.0409892	training's auc: 0.999473	valid_1's binary_logloss: 0.126997	valid_1's auc: 0.847413
    [25]	training's binary_logloss: 0.0395634	training's auc: 0.999688	valid_1's binary_logloss: 0.126953	valid_1's auc: 0.847239
    [26]	training's binary_logloss: 0.038262	training's auc: 0.999783	valid_1's binary_logloss: 0.12665	valid_1's auc: 0.847107
    [27]	training's binary_logloss: 0.0369473	training's auc: 0.999871	valid_1's binary_logloss: 0.12676	valid_1's auc: 0.847103
    [28]	training's binary_logloss: 0.0357513	training's auc: 0.999929	valid_1's binary_logloss: 0.127088	valid_1's auc: 0.846456
    [29]	training's binary_logloss: 0.0346801	training's auc: 0.999938	valid_1's binary_logloss: 0.127242	valid_1's auc: 0.846537
    [30]	training's binary_logloss: 0.0336486	training's auc: 0.99997	valid_1's binary_logloss: 0.127418	valid_1's auc: 0.846343
    [31]	training's binary_logloss: 0.0325476	training's auc: 0.999991	valid_1's binary_logloss: 0.127741	valid_1's auc: 0.845251
    [32]	training's binary_logloss: 0.0313824	training's auc: 0.999998	valid_1's binary_logloss: 0.127947	valid_1's auc: 0.847329
    [33]	training's binary_logloss: 0.0303287	training's auc: 1	valid_1's binary_logloss: 0.128155	valid_1's auc: 0.847719
    [34]	training's binary_logloss: 0.0293654	training's auc: 1	valid_1's binary_logloss: 0.128578	valid_1's auc: 0.847521
    [35]	training's binary_logloss: 0.0283718	training's auc: 1	valid_1's binary_logloss: 0.129013	valid_1's auc: 0.846986
    [36]	training's binary_logloss: 0.0273982	training's auc: 1	valid_1's binary_logloss: 0.129661	valid_1's auc: 0.846978
    [37]	training's binary_logloss: 0.0265119	training's auc: 1	valid_1's binary_logloss: 0.129987	valid_1's auc: 0.846909
    [38]	training's binary_logloss: 0.0256151	training's auc: 1	valid_1's binary_logloss: 0.130576	valid_1's auc: 0.845966
    [39]	training's binary_logloss: 0.0248319	training's auc: 1	valid_1's binary_logloss: 0.130833	valid_1's auc: 0.846156
    [40]	training's binary_logloss: 0.0240933	training's auc: 1	valid_1's binary_logloss: 0.131051	valid_1's auc: 0.845674
    [41]	training's binary_logloss: 0.0233588	training's auc: 1	valid_1's binary_logloss: 0.131771	valid_1's auc: 0.844286
    [42]	training's binary_logloss: 0.0226336	training's auc: 1	valid_1's binary_logloss: 0.132106	valid_1's auc: 0.844235
    [43]	training's binary_logloss: 0.0219562	training's auc: 1	valid_1's binary_logloss: 0.132606	valid_1's auc: 0.844469
    [44]	training's binary_logloss: 0.0212872	training's auc: 1	valid_1's binary_logloss: 0.132841	valid_1's auc: 0.844792
    [45]	training's binary_logloss: 0.0206507	training's auc: 1	valid_1's binary_logloss: 0.133052	valid_1's auc: 0.845451
    [46]	training's binary_logloss: 0.0200644	training's auc: 1	valid_1's binary_logloss: 0.133745	valid_1's auc: 0.844882
    [47]	training's binary_logloss: 0.0194828	training's auc: 1	valid_1's binary_logloss: 0.134543	valid_1's auc: 0.844067
    [48]	training's binary_logloss: 0.0188788	training's auc: 1	valid_1's binary_logloss: 0.135237	valid_1's auc: 0.842354
    [49]	training's binary_logloss: 0.0183189	training's auc: 1	valid_1's binary_logloss: 0.135366	valid_1's auc: 0.843063
    [50]	training's binary_logloss: 0.0178593	training's auc: 1	valid_1's binary_logloss: 0.135914	valid_1's auc: 0.841836
    [51]	training's binary_logloss: 0.0174614	training's auc: 1	valid_1's binary_logloss: 0.136431	valid_1's auc: 0.841281
    [52]	training's binary_logloss: 0.0168703	training's auc: 1	valid_1's binary_logloss: 0.136852	valid_1's auc: 0.841444
    [53]	training's binary_logloss: 0.0163364	training's auc: 1	valid_1's binary_logloss: 0.137472	valid_1's auc: 0.841109
    [54]	training's binary_logloss: 0.0158803	training's auc: 1	valid_1's binary_logloss: 0.137835	valid_1's auc: 0.841187
    [55]	training's binary_logloss: 0.0154185	training's auc: 1	valid_1's binary_logloss: 0.13813	valid_1's auc: 0.840791
    [56]	training's binary_logloss: 0.0150188	training's auc: 1	valid_1's binary_logloss: 0.138555	valid_1's auc: 0.839817
    [57]	training's binary_logloss: 0.0146288	training's auc: 1	valid_1's binary_logloss: 0.139285	valid_1's auc: 0.837782
    [58]	training's binary_logloss: 0.0141671	training's auc: 1	valid_1's binary_logloss: 0.139772	valid_1's auc: 0.837804
    [59]	training's binary_logloss: 0.0137083	training's auc: 1	valid_1's binary_logloss: 0.14019	valid_1's auc: 0.837416
    [60]	training's binary_logloss: 0.0133052	training's auc: 1	valid_1's binary_logloss: 0.140814	valid_1's auc: 0.836796
    [61]	training's binary_logloss: 0.0128988	training's auc: 1	valid_1's binary_logloss: 0.141344	valid_1's auc: 0.836663
    [62]	training's binary_logloss: 0.0125285	training's auc: 1	valid_1's binary_logloss: 0.141935	valid_1's auc: 0.836195
    [63]	training's binary_logloss: 0.0121711	training's auc: 1	valid_1's binary_logloss: 0.142322	valid_1's auc: 0.836814
    [64]	training's binary_logloss: 0.0118152	training's auc: 1	valid_1's binary_logloss: 0.143164	valid_1's auc: 0.836208
    [65]	training's binary_logloss: 0.0114889	training's auc: 1	valid_1's binary_logloss: 0.143674	valid_1's auc: 0.835646
    [66]	training's binary_logloss: 0.0111595	training's auc: 1	valid_1's binary_logloss: 0.144268	valid_1's auc: 0.835663
    [67]	training's binary_logloss: 0.0108488	training's auc: 1	valid_1's binary_logloss: 0.145316	valid_1's auc: 0.833615
    [68]	training's binary_logloss: 0.0105621	training's auc: 1	valid_1's binary_logloss: 0.146085	valid_1's auc: 0.832956
    Early stopping, best iteration is:
    [18]	training's binary_logloss: 0.051504	training's auc: 0.996457	valid_1's binary_logloss: 0.127456	valid_1's auc: 0.850324
    ****************************************************************************************************



```python
test_probs.shape
```




    (11676,)




```python
X_train.shape
```




    (11529, 476)




```python
test_probs.to_csv("benchmark_03-09-2019_remove_constant_columns_098.zip", header=True, compression="zip")
```


```python
test_probs.shape
```




    (11676,)



* With 0.98 gives 0.839999 score, better than 0.99 but still lower than vanilla baseline

### Using XGB


```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
```


```python
train = X_train.copy()
test = X_test.copy()
target = y_train.copy()
```


```python
XX_train, XX_valid, yy_train, yy_valid = train_test_split(train, target, test_size = 0.2, stratify = target)



```


```python
clf = xgb.XGBClassifier(max_depth = 4,
                n_estimators=10000,
                learning_rate=0.1, 
                # nthread=4,
                subsample=0.8,
                colsample_bytree=0.8,
                #min_child_weight = 3,
                # scale_pos_weight = ratio,
                #reg_alpha=50,
                #reg_lambda = 50,
                seed=42)
```


```python
clf.fit(XX_train, yy_train, early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(XX_train, yy_train), (XX_valid, yy_valid)])
```

    [0]	validation_0-auc:0.818687	validation_1-auc:0.787636
    Multiple eval metrics have been passed: 'validation_1-auc' will be used for early stopping.
    
    Will train until validation_1-auc hasn't improved in 50 rounds.
    [1]	validation_0-auc:0.826944	validation_1-auc:0.787118
    [2]	validation_0-auc:0.844102	validation_1-auc:0.788626
    [3]	validation_0-auc:0.859109	validation_1-auc:0.793387
    [4]	validation_0-auc:0.884856	validation_1-auc:0.804994
    [5]	validation_0-auc:0.886563	validation_1-auc:0.806786
    [6]	validation_0-auc:0.89115	validation_1-auc:0.803584
    [7]	validation_0-auc:0.893021	validation_1-auc:0.803023
    [8]	validation_0-auc:0.899585	validation_1-auc:0.820018
    [9]	validation_0-auc:0.899996	validation_1-auc:0.820392
    [10]	validation_0-auc:0.902817	validation_1-auc:0.819983
    [11]	validation_0-auc:0.909219	validation_1-auc:0.829231
    [12]	validation_0-auc:0.912691	validation_1-auc:0.830154
    [13]	validation_0-auc:0.91381	validation_1-auc:0.830514
    [14]	validation_0-auc:0.914642	validation_1-auc:0.829668
    [15]	validation_0-auc:0.915714	validation_1-auc:0.829863
    [16]	validation_0-auc:0.917183	validation_1-auc:0.828793
    [17]	validation_0-auc:0.920056	validation_1-auc:0.829277
    [18]	validation_0-auc:0.919797	validation_1-auc:0.828048
    [19]	validation_0-auc:0.920644	validation_1-auc:0.833989
    [20]	validation_0-auc:0.92418	validation_1-auc:0.83323
    [21]	validation_0-auc:0.924859	validation_1-auc:0.832721
    [22]	validation_0-auc:0.926895	validation_1-auc:0.832942
    [23]	validation_0-auc:0.926908	validation_1-auc:0.832467
    [24]	validation_0-auc:0.92797	validation_1-auc:0.833348
    [25]	validation_0-auc:0.929412	validation_1-auc:0.832013
    [26]	validation_0-auc:0.931584	validation_1-auc:0.831662
    [27]	validation_0-auc:0.93228	validation_1-auc:0.831944
    [28]	validation_0-auc:0.932891	validation_1-auc:0.833814
    [29]	validation_0-auc:0.933905	validation_1-auc:0.835739
    [30]	validation_0-auc:0.935516	validation_1-auc:0.83428
    [31]	validation_0-auc:0.93572	validation_1-auc:0.834084
    [32]	validation_0-auc:0.939081	validation_1-auc:0.834032
    [33]	validation_0-auc:0.939693	validation_1-auc:0.832349
    [34]	validation_0-auc:0.940579	validation_1-auc:0.833575
    [35]	validation_0-auc:0.941515	validation_1-auc:0.833074
    [36]	validation_0-auc:0.943071	validation_1-auc:0.830683
    [37]	validation_0-auc:0.944694	validation_1-auc:0.828782
    [38]	validation_0-auc:0.946662	validation_1-auc:0.829187
    [39]	validation_0-auc:0.947113	validation_1-auc:0.832085
    [40]	validation_0-auc:0.948034	validation_1-auc:0.834349
    [41]	validation_0-auc:0.95002	validation_1-auc:0.832205
    [42]	validation_0-auc:0.951958	validation_1-auc:0.837822
    [43]	validation_0-auc:0.953975	validation_1-auc:0.837249
    [44]	validation_0-auc:0.955158	validation_1-auc:0.836133
    [45]	validation_0-auc:0.956943	validation_1-auc:0.838357
    [46]	validation_0-auc:0.958296	validation_1-auc:0.837289
    [47]	validation_0-auc:0.959455	validation_1-auc:0.836541
    [48]	validation_0-auc:0.960748	validation_1-auc:0.836823
    [49]	validation_0-auc:0.961504	validation_1-auc:0.836941
    [50]	validation_0-auc:0.962513	validation_1-auc:0.836803
    [51]	validation_0-auc:0.963195	validation_1-auc:0.835655
    [52]	validation_0-auc:0.964788	validation_1-auc:0.83722
    [53]	validation_0-auc:0.965143	validation_1-auc:0.836841
    [54]	validation_0-auc:0.966637	validation_1-auc:0.835391
    [55]	validation_0-auc:0.967604	validation_1-auc:0.836
    [56]	validation_0-auc:0.96896	validation_1-auc:0.83455
    [57]	validation_0-auc:0.970339	validation_1-auc:0.833587
    [58]	validation_0-auc:0.971178	validation_1-auc:0.833529
    [59]	validation_0-auc:0.971747	validation_1-auc:0.831601
    [60]	validation_0-auc:0.972856	validation_1-auc:0.834668
    [61]	validation_0-auc:0.973241	validation_1-auc:0.833903
    [62]	validation_0-auc:0.974562	validation_1-auc:0.832539
    [63]	validation_0-auc:0.97559	validation_1-auc:0.830583
    [64]	validation_0-auc:0.976187	validation_1-auc:0.831555
    [65]	validation_0-auc:0.976592	validation_1-auc:0.833909
    [66]	validation_0-auc:0.976878	validation_1-auc:0.832942
    [67]	validation_0-auc:0.977256	validation_1-auc:0.83304
    [68]	validation_0-auc:0.977434	validation_1-auc:0.834041
    [69]	validation_0-auc:0.977497	validation_1-auc:0.833961
    [70]	validation_0-auc:0.978304	validation_1-auc:0.832108
    [71]	validation_0-auc:0.979064	validation_1-auc:0.830911
    [72]	validation_0-auc:0.97966	validation_1-auc:0.829576
    [73]	validation_0-auc:0.980408	validation_1-auc:0.8293
    [74]	validation_0-auc:0.981182	validation_1-auc:0.82797
    [75]	validation_0-auc:0.981661	validation_1-auc:0.82816
    [76]	validation_0-auc:0.982451	validation_1-auc:0.828356
    [77]	validation_0-auc:0.983024	validation_1-auc:0.827671
    [78]	validation_0-auc:0.983281	validation_1-auc:0.827798
    [79]	validation_0-auc:0.984147	validation_1-auc:0.828603
    [80]	validation_0-auc:0.984323	validation_1-auc:0.827516
    [81]	validation_0-auc:0.98481	validation_1-auc:0.826445
    [82]	validation_0-auc:0.98517	validation_1-auc:0.826555
    [83]	validation_0-auc:0.985361	validation_1-auc:0.827124
    [84]	validation_0-auc:0.985883	validation_1-auc:0.82835
    [85]	validation_0-auc:0.986529	validation_1-auc:0.827153
    [86]	validation_0-auc:0.986805	validation_1-auc:0.826825
    [87]	validation_0-auc:0.987011	validation_1-auc:0.827637
    [88]	validation_0-auc:0.987264	validation_1-auc:0.826814
    [89]	validation_0-auc:0.987837	validation_1-auc:0.826267
    [90]	validation_0-auc:0.988159	validation_1-auc:0.827101
    [91]	validation_0-auc:0.988428	validation_1-auc:0.826388
    [92]	validation_0-auc:0.988677	validation_1-auc:0.827522
    [93]	validation_0-auc:0.988862	validation_1-auc:0.826463
    [94]	validation_0-auc:0.989129	validation_1-auc:0.826405
    [95]	validation_0-auc:0.989665	validation_1-auc:0.82568
    Stopping. Best iteration:
    [45]	validation_0-auc:0.956943	validation_1-auc:0.838357
    





    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bytree=0.8, gamma=0, learning_rate=0.1,
                  max_delta_step=0, max_depth=4, min_child_weight=1, missing=None,
                  n_estimators=10000, n_jobs=1, nthread=None,
                  objective='binary:logistic', random_state=0, reg_alpha=0,
                  reg_lambda=1, scale_pos_weight=1, seed=42, silent=True,
                  subsample=0.8)




```python
# model with replies features
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,X_train.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 30))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(50))
plt.title('LightGBM Features')
plt.show()
```


![png](/blog/lgb_files/output_35_0.png)



```python
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
```


```python
def score(params):
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    bst = xgb.train(params, 
                      dtrain, 
                      num_round,
                      evals=watchlist,
                      verbose_eval=False)
    predictions = bst.predict(dvalid, ntree_limit=bst.best_iteration)
    roc = roc_auc_score(y_valid, np.array(predictions))
    print(roc)
    return {'loss': roc, 'status': STATUS_OK}
```


```python
def optimize(evals, cores, trials, optimizer=tpe.suggest, random_state=42):
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 600, 1),
        'eta': hp.quniform('eta', 0.025, 0.25, 0.025), # A problem with max_depth casted to float instead of int with the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
        'alpha' :  hp.quniform('alpha', 0, 10, 1),
        'lambda': hp.quniform('lambda', 1, 2, 0.1),
        'nthread': cores,
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'seed': random_state
    }
    best = fmin(score, space, algo=tpe.suggest, max_evals=evals, trials = trials)
    return best
```


```python
trials = Trials()
cores = 8
n= 10
best_param = optimize(evals = n,
                      optimizer=tpe.suggest,
                      cores = cores,
                      trials = trials)
```

    0.8020687645687645                                  
     10%|█         | 1/10 [01:37<14:41, 97.91s/it, best loss: 0.8020687645687645]

    /anaconda3/envs/pyenv/lib/python3.7/site-packages/xgboost/core.py:587: FutureWarning: Series.base is deprecated and will be removed in a future version
      if getattr(data, 'base', None) is not None and \
    


    0.8221853146853146                                                           
     20%|██        | 2/10 [02:53<12:09, 91.17s/it, best loss: 0.8020687645687645]

    /anaconda3/envs/pyenv/lib/python3.7/site-packages/xgboost/core.py:587: FutureWarning: Series.base is deprecated and will be removed in a future version
      if getattr(data, 'base', None) is not None and \
    


    0.8257983682983683                                                           
     30%|███       | 3/10 [07:44<17:39, 151.29s/it, best loss: 0.8020687645687645]

    /anaconda3/envs/pyenv/lib/python3.7/site-packages/xgboost/core.py:587: FutureWarning: Series.base is deprecated and will be removed in a future version
      if getattr(data, 'base', None) is not None and \
    


    0.8397843822843822                                                            
     40%|████      | 4/10 [09:50<14:21, 143.50s/it, best loss: 0.8020687645687645]

    /anaconda3/envs/pyenv/lib/python3.7/site-packages/xgboost/core.py:587: FutureWarning: Series.base is deprecated and will be removed in a future version
      if getattr(data, 'base', None) is not None and \
    


    0.8381002331002331                                                            
     50%|█████     | 5/10 [15:14<16:28, 197.69s/it, best loss: 0.8020687645687645]

    /anaconda3/envs/pyenv/lib/python3.7/site-packages/xgboost/core.py:587: FutureWarning: Series.base is deprecated and will be removed in a future version
      if getattr(data, 'base', None) is not None and \
    


    0.7970920745920745                                                            
     60%|██████    | 6/10 [18:06<12:39, 189.97s/it, best loss: 0.7970920745920745]

    /anaconda3/envs/pyenv/lib/python3.7/site-packages/xgboost/core.py:587: FutureWarning: Series.base is deprecated and will be removed in a future version
      if getattr(data, 'base', None) is not None and \
    


    0.839965034965035                                                             
     70%|███████   | 7/10 [19:11<07:37, 152.44s/it, best loss: 0.7970920745920745]

    /anaconda3/envs/pyenv/lib/python3.7/site-packages/xgboost/core.py:587: FutureWarning: Series.base is deprecated and will be removed in a future version
      if getattr(data, 'base', None) is not None and \
    


    0.7940209790209791                                                            
     80%|████████  | 8/10 [21:04<04:41, 140.79s/it, best loss: 0.7940209790209791]

    /anaconda3/envs/pyenv/lib/python3.7/site-packages/xgboost/core.py:587: FutureWarning: Series.base is deprecated and will be removed in a future version
      if getattr(data, 'base', None) is not None and \
    


    0.8001165501165503                                                            
     90%|█████████ | 9/10 [23:49<02:27, 147.95s/it, best loss: 0.7940209790209791]

    /anaconda3/envs/pyenv/lib/python3.7/site-packages/xgboost/core.py:587: FutureWarning: Series.base is deprecated and will be removed in a future version
      if getattr(data, 'base', None) is not None and \
    


    0.8047552447552447                                                            
    100%|██████████| 10/10 [25:20<00:00, 152.09s/it, best loss: 0.7940209790209791]



```python
best_param
```




    {'alpha': 0.0,
     'colsample_bytree': 0.7000000000000001,
     'eta': 0.2,
     'gamma': 0.9,
     'lambda': 1.0,
     'max_depth': 6,
     'min_child_weight': 8.0,
     'n_estimators': 231.0,
     'subsample': 0.75}



### Logistic Regression



```python
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC
```


```python
lr = LinearSVC()
```


```python
XX_train, XX_valid, yy_train, yy_valid = train_test_split(train, target, test_size = 0.2, stratify = target)
```


```python
lr.fit(XX_train, yy_train)
```




    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=1000,
              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
              verbose=0)




```python
vals = lr.predict(XX_valid)
```


```python
metrics.roc_auc_score(yy_valid, vals)
```




    0.5




```python
print ("Model building..")

for model_name in ["logreg"]:

    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2018)
    cv_scores = []
    pred_test_full = 0
    pred_val_full = np.zeros(X_train.shape[0])
    for dev_index, val_index in kf.split(X_train):
        dev_X, val_X = X_train.iloc[dev_index,:], X_train.iloc[val_index,:]
        dev_y, val_y = y_train[dev_index], y_train[val_index]

        if model_name == "logreg":
            lr.fit(dev_X, dev_y)
            pred_val = lr.predict(val_X)
            pred_test = lr.predict(X_test)
            loss = metrics.roc_auc_score(val_y, pred_val)
        
        pred_val_full[val_index] = pred_val
        pred_test_full = pred_test_full + pred_test
        cv_scores.append(loss)
        print ('cv scores:', cv_scores)
    pred_test_full /= 5.
    print ('roc:', metrics.roc_auc_score(y_train, pred_val_full))
```

    Model building..


    /anaconda3/envs/pyenv/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)


    cv scores: [0.5]
    cv scores: [0.5, 0.5]


    /anaconda3/envs/pyenv/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /anaconda3/envs/pyenv/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)


    cv scores: [0.5, 0.5, 0.5]
    cv scores: [0.5, 0.5, 0.5, 0.5]


    /anaconda3/envs/pyenv/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /anaconda3/envs/pyenv/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)


    cv scores: [0.5, 0.5, 0.5, 0.5, 0.5]



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-190-a5111c6ea899> in <module>
         21         cv_scores.append(loss)
         22         print ('cv scores:', cv_scores)
    ---> 23     pred_test_full /= 5.
         24     print ('roc:', metrics.roc_auc_score(y_train, pred_val_full))


    TypeError: ufunc 'true_divide' output (typecode 'd') could not be coerced to provided output parameter (typecode 'l') according to the casting rule ''same_kind''

