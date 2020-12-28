# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 09:12:02 2020

@author: Abhinaya
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import multiprocessing
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import gc
from time import time
import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import sys
import lightgbm as lgb
from sklearn.model_selection import KFold,GroupKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats
from scipy.stats import skew,norm
from scipy.stats.stats import pearsonr
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
#conda install -c conda-forge xgboost
#conda install -c conda-forge xgboost
#pip install xgboost
import xgboost as xgb
#import lightgbm as lgb

###############################################
id  = pd.read_csv("C:/Users/SM/Desktop/train_identity.csv")
train = pd.read_csv("C:/Users/SM/Desktop/train.csv")
test = pd.read_csv("C:/Users/SM/Desktop/test.csv")

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

# Dropping  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# Lets have a look at our datasets Train and Test
print("Train Dataset has "+ str(train.shape[1])+ " Columns")
print("Train Dataset has "+ str(train.shape[0])+" Rows")

print("Test Dataset has "+ str(test.shape[1])+ " Columns")
print("Test Dataset has "+ str(test.shape[0])+ " Rows")

# Lets look at the target variable
train['SalePrice'].describe()

print(train['SalePrice'].describe())


# Kernel Density Plot
sns.distplot(train.SalePrice,fit= norm, color ='r');
plt.ylabel('Frequency Distribution')
plt.title('SalePrice Distribution');

# We notice a positive skew in Sale Price.Since linear model fits better
# on normally distributed data, we will correct skewed 
#numeric features by taking log(feature + 1) - to make features more normal



# Scatterplot to study a few Attributes 

TotalBasement = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis=1)
TotalBasement.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000), color ='y')
plt.show();


GrLivAreascatter = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)
GrLivAreascatter.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000), color ='r');

#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#box plot overallqual/saleprice
xar = 'OverallQual'
overallvssaleprice = pd.concat([train['SalePrice'], train[xar]], axis=1)
f, ax = plt.subplots(figsize=(10, 6))
fig = sns.boxplot(x=xar, y="SalePrice", data= overallvssaleprice)
fig.axis(ymin=0, ymax=800000);


xar = 'YearBuilt'
yrbuiltvssaleprice = pd.concat([train['SalePrice'], train[xar]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=xar, y="SalePrice", data= yrbuiltvssaleprice)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

#correlation matrix
corrmatrix = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmatrix, vmax=.9, square=True);


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmatrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#According to our crystal ball, 
#these are the variables most correlated with 'SalePrice'.
#'OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'.
#'GarageCars' and 'GarageArea' are also some of the most strongly correlated variables. However, as we discussed in the last sub-point, the number of cars that fit into the garage is a consequence of the garage area. 'GarageCars' and 'GarageArea' are like twin brothers. You'll never be able to distinguish them.
# Therefore, we just need one of these variables in our analysis (we can keep 'GarageCars' since its correlation with 'SalePrice' is higher).
#'TotalBsmtSF' and '1stFloor' also seem to be twin brothers.
# We can keep 'TotalBsmtSF' just to say that our first guess was right
# (re-read 'So... What can we expect?').'TotRmsAbvGrd' and 'GrLivArea', same again.
#It seems that 'YearBuilt' is slightly correlated with 'SalePrice'. Honestly, it scares me to think about 'YearBuilt' because I start feeling that we should do a little bit of time-series analysis to get this right.


#skewness and kurtosis
print("Skew of Saleprice in train is " + str(train['SalePrice'].skew()))
print("Kurtosis of SalePrice is  "+ str(train['SalePrice'].kurt()))

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])
#SalePrice_t = np.log1p(train.pop('SalePrice'))

train_test_concat = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

#Coutning missing values in cpncatenated train and test datasets
Train_test_null =  1 - train_test_concat.count()/len(train_test_concat.index)
Train_test_topnull = Train_test_null.sort_values(ascending=False)[:30]
#print(Train_test_topnull)

missing_percent = pd.DataFrame({'Missing %' :Train_test_topnull})

f, ax = plt.subplots(figsize=(7, 4))
plt.xticks(rotation='90')
sns.barplot(x=Train_test_topnull.index, y=Train_test_topnull, color = "green")
plt.xlabel('Features', fontsize=15)
plt.ylabel('% Missing', fontsize=15)
plt.title('Missing %', fontsize=15)
print(missing_percent)
#Going to replace nulls with 0 in certain fields related to garage 
# and Basements bec null means 0 garage
for i in ('GarageYrBlt','MasVnrArea', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    train_test_concat[i] = train_test_concat[i].fillna(0)
    

# Categorical features such as basement quality NaN means that there is no basement.
for i in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train_test_concat[i] = train_test_concat[i].fillna('None')

# Fill in median LotFrontage of all the neighborhood after group by neighbourhood
train_test_concat["LotFrontage"] =  train_test_concat.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
# Setting mode value for missing entries 

#MSZoning classification : 'RL' replace missing with mode
train_test_concat['MSZoning'] = train_test_concat['MSZoning'].fillna(train_test_concat['MSZoning'].mode()[0])

# Functional : NA = typical replace missing with mode
train_test_concat["Functional"] = train_test_concat["Functional"].fillna("Typ")

# Electrical replace missing with mode
train_test_concat['Electrical'] = train_test_concat['Electrical'].fillna(train_test_concat['Electrical'].mode()[0])

# KitchenQual replace missing with mode
train_test_concat['KitchenQual'] = train_test_concat['KitchenQual'].fillna(train_test_concat['KitchenQual'].mode()[0])

# Exterior1st and Exterior2nd replace missing with mode
train_test_concat['Exterior1st'] = train_test_concat['Exterior1st'].fillna(train_test_concat['Exterior1st'].mode()[0])
train_test_concat['Exterior2nd'] = train_test_concat['Exterior2nd'].fillna(train_test_concat['Exterior2nd'].mode()[0])

#SaleType replace missing with mode
train_test_concat['SaleType'] = train_test_concat['SaleType'].fillna(train_test_concat['SaleType'].mode()[0])

print(train_test_concat.Utilities.value_counts())
# It looks likefor column Utilitites all the values except one are "AllPub" so lets remove column

#  Dropping as same value 'AllPub' for all records except 2 NA and 1 'NoSeWa'
train_test_concat = train_test_concat.drop(['Utilities'], axis=1)

# Numerical to categorical 

train_test_concat['MSSubClass'] = train_test_concat['MSSubClass'].apply(str)
train_test_concat['OverallCond'] = train_test_concat['OverallCond'].astype(str)
train_test_concat['YrSold'] = train_test_concat['YrSold'].astype(str)
train_test_concat['MoSold'] = train_test_concat['MoSold'].astype(str)

#Label Encoding some categorical variables 
# for information in their ordering set

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train_test_concat[c].values)) 
    train_test_concat[c] = lbl.transform(list(train_test_concat[c].values))

# shape        
print('Shape train_test_concat: {}'.format(train_test_concat.shape))

# Adding Total surface area as 'TotalSF'= basement+firstflr+secondflr

train_test_concat['TotalSF'] = train_test_concat['TotalBsmtSF'] + train_test_concat['1stFlrSF'] + train_test_concat['2ndFlrSF']
# Finding skew and log transform (https://medium.com/@ODSC/transforming-skewed-data-for-machine-learning-90e6cc364b0)
numeric_train_test = train_test_concat.dtypes[train_test_concat.dtypes != "object"].index
#numeric_train_test =train_test_concat[numeric_train_test].skew().sort_values(ascending = False)
print(numeric_train_test)
numeric_train_test = train_test_concat[['MSSubClass', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape',  'LandSlope', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd','MasVnrArea', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',   'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',  'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC', 'CentralAir','1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',   'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',   'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',  'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageCars',   'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',   'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',   'PoolQC', 'Fence', 'MiscVal', 'MoSold', 'YrSold', 'TotalSF']]

# skewness along the index axis 
skewed_numeric_train_test = numeric_train_test.skew(axis = 0, skipna = True) 
skewofnumeric = pd.DataFrame({'Skew' : skewed_numeric_train_test})


skewgreaterthan75 = skewofnumeric[abs(skewofnumeric) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewgreaterthan75.shape[0]))
print(skewgreaterthan75)
#https://machinelearningmastery.com/power-transforms-with-scikit-learn/
#  https://stackoverflow.com/questions/63105754/how-do-i-calculate-lambda-to-use-scipy-special-boxcox1p-function-for-my-entire-d
import numpy as np
from sklearn.preprocessing import PowerTransformer
#pt = PowerTransformer(method='yeo-johnson')
skewed_features = skewgreaterthan75.index
#data = train_test_concat[['MSSubClass', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape',
      # 'LandSlope', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
      ##'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       #'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC', 'CentralAir',
       #'1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
       #'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       #'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
       #'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
       #'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
       #'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       #'PoolQC', 'Fence', 'MiscVal', 'MoSold', 'YrSold', 'TotalSF']]
#data = pt.fit_transform(data)
#dataset = pd.DataFrame(data)
#print("lambda is")
#print(pt.lambdas_)
from scipy.special import boxcox1p
#skewed_features = skewgreaterthan75.index

lam = 0.15
for feat in skewed_features:
    train_test_concat[feat] = boxcox1p(train_test_concat[feat], lam)
#train_test_concat[skewed_features] = np.log1p(train_test_concat[skewed_features])

#Getting categprocal dummy 

train_test_concat = pd.get_dummies(train_test_concat)
print(train_test_concat.shape)

ntrain = train.shape[0]
ntest = test.shape[0]
y_train= train.SalePrice.values
train = pd.DataFrame(train_test_concat[:ntrain])
test = pd.DataFrame(train_test_concat[ntrain:])
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
#1
lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005, random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

n_folds = 10

def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=1234).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


lasso_alpha = [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]
lasso_rmse = []

for value in lasso_alpha:
    lasso = make_pipeline(RobustScaler(), Lasso(alpha = value, max_iter=3000, random_state = 1234))
    lasso_rmse.append(rmse_cv(lasso).mean())

lasso_score_table = pd.DataFrame(lasso_rmse,lasso_alpha,columns=['RMSE'])
print(lasso_score_table.transpose())

plt.semilogx(lasso_alpha, lasso_rmse)
plt.xlabel('alpha')
plt.ylabel('score')
plt.show()

print("\nLasso Score is: {:.4f} (alpha = {:.5f})\n".format(min(lasso_score_table['RMSE']), lasso_score_table.idxmin()[0]))

# Grid Search
#param_grid = [{ 'alpha' : [1.0, 0.1, .001, 0.005]
 #             }
  #           ]

#lasso_grid = Lasso()

#grid_search_L = GridSearchCV(lasso_grid, param_grid, cv = 5, n_jobs=-1,
                          #scoring = 'neg_mean_squared_error')
#grid_search_L.fit(train, y_train)

#print(grid_search_L.best_params_)

#2
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


#3
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#4
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#5
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,seed=7, nthread = -1)

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#6
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


#Average Based models class

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


    
# Averaged base models score

averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# Defining rmsle evaluation function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


#Final Training and Prediction

#StackedRegressor:

averaged_models.fit(train.values, y_train)
stacked_train_pred = averaged_models.predict(train.values)
stacked_pred = np.expm1(averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))


# XGBoost

model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))

# LightGBM

model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))

'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))

# Ensembled Predictions:

ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)
