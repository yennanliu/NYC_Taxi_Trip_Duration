# -*- coding: utf-8 -*-

import pandas as pd, numpy as np
import calendar
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.grid_search import RandomizedSearchCV
from xgboost import XGBRegressor
import xgboost as xgb
import scipy.stats as st



def tune_model_1(X,y):
    params = {  "min_child_weight": st.randint(8, 10),
                "eta": st.randint(0.02, 0.05),
                "n_estimators": st.randint(1000, 1010),
                "max_depth": st.randint(10, 13),
                "subsample" : st.randint(0.5, 0.75),
                "learning_rate": st.uniform(0.05, 0.4),
                'booster' : 'gbtree',
                'eval_metric': 'rmse', 
                'objective': 'reg:linear'
                'nthread': -1}
    xgbreg = XGBRegressor(nthreads=-1) 
    # random search 
    gs = RandomizedSearchCV(xgbreg, params, n_jobs=1)  
    #gs.fit(X_train, y_train) 
    gs.fit(X, y) 
    print (gs.best_model_)
    return gs.best_model_



def xgb_model_1(X_train,y_train,X_test,params=None):

	xgb = XGBRegressor(n_estimators=1000, max_depth=13, min_child_weight=150, 
                   subsample=0.7, colsample_bytree=0.3)
    y_test = np.zeros(len(X_test))
	for i, (train_ind, val_ind) in enumerate(KFold(n_splits=2, shuffle=True, 
                                            random_state=1989).split(X_train)):
        print('----------------------')
        print('Training model #%d' % i)
        print('----------------------')
        # XGBRegressor.fit 
        xgb.fit(X_train[train_ind], y_train[train_ind],
                eval_set=[(X_train[val_ind], y_train[val_ind])],
                early_stopping_rounds=10, verbose=25)
        
        y_test += xgb.predict(X_test, ntree_limit=xgb.best_ntree_limit)
    y_test /= 2

    return y_test



def xgb_model_2(X_train,y_train,X_test,y_test,params=None):
	# transform data to DMatrix form for fasting fitting process 
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    dtest = xgb.DMatrix(X_test)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
	xgb_pars = {'min_child_weight': 10, 'eta': 0.04,
				'colsample_bytree': 0.8, 'max_depth': 15,
            	'subsample': 0.75, 'lambda': 2, 'nthread': -1,
             	'booster' : 'gbtree', 'silent': 1, 'gamma' : 0,
            	'eval_metric': 'rmse', 'objective': 'reg:linear'}    
    # xgb.train   	
    model = xgb.train(xgb_pars, dtrain, 500, watchlist, early_stopping_rounds=250,
                  maximize=False, verbose_eval=15)
    ytest = model.predict(dtest)
    return ytest









