import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn import preprocessing


class Data_loader:

    df_dataset=None
    df_train=None
    df_test=None
    y_train = None
    y_test = None
    num_feature = None
    X_train = None
    X_test = None
    X_train_pca = None
    X_test_pca = None

    def __init__(self,path,split,num_feature):
        self.df_dataset = self.read_data(path)
        self.df_test = self.df_dataset.iloc[-int(len(self.df_dataset) / split):]
        self.df_train = self.df_dataset.iloc[:-int(len(self.df_dataset) / split)]

        # DELETE NAN
        self.df_train[np.isinf(self.df_train)] = np.nan
        self.df_test[np.isinf(self.df_test)] = np.nan
        self.df_train.dropna(inplace=True)
        self.df_test.dropna(inplace=True)

        self.num_feature=num_feature
        self.y_train = self.df_train[self.df_train.columns[0]].values * 10000
        self.y_test = self.df_test[self.df_test.columns[0]].values * 10000
        self.X_train = self.df_train[self.df_train.columns[1:1 + num_feature]].values
        self.X_test = self.df_test[self.df_test.columns[1:1 + num_feature]].values

    def read_data(self,path):
        df_dataset=pd.DataFrame()
        print('loading dataset...')
        for file_name in os.listdir(path):
            if len(df_dataset) != 0:
                df = pd.read_csv(path + file_name, index_col=0)
                df_dataset = pd.concat([df_dataset, df])
            else:
                df_dataset = pd.read_csv(path + file_name, index_col=0)
            print(file_name)
        return df_dataset

    def corr_plot(self,k=15):
        CORR = self.df_train[self.df_train.columns[:k]].corr()
        plt.subplots(figsize=(20, 20))
        sns.heatmap(CORR, annot=False, vmax=1, center=0, cmap="RdBu_r")
        plt.show()

    def pca_processor(self):
        X = np.vstack((self.X_train, self.X_test))
        scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
        pca = PCA(n_components=self.num_feature, random_state=42)
        X_pca = pca.fit_transform(X)
        X_pca = scaler.fit_transform(X_pca)
        self.X_train_pca = X_pca[:self.X_train.shape[0], :]
        self.X_test_pca = X_pca[self.X_train.shape[0]:, :]


class gbdt_learner:
    dataset = None
    param = None

    def __init__(self,dataset,param):
        self.dataset = dataset
        self.param = param

    def fit(self,use_pca=True):
        if use_pca:
            print('using PCA...')
            self.dataset.pca_processor()
            lgb_train = lgb.Dataset(self.dataset.X_train_pca, self.dataset.y_train)
            lgb_eval = lgb.Dataset(self.dataset.X_test_pca, self.dataset.y_test, reference=lgb_train)
        else:
            lgb_train = lgb.Dataset(self.dataset.X_train, self.dataset.y_train)
            lgb_eval = lgb.Dataset(self.dataset.X_test, self.dataset.y_test, reference=lgb_train)

        evals_result = {}
        gbm = lgb.train(self.param,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_eval],
                        evals_result=evals_result,
                        feature_name=list(self.dataset.df_dataset.columns[1:])[:self.dataset.num_feature]
                        )

        print('training loss...')
        ax = lgb.plot_metric(evals_result, metric='l2')
        plt.show()

        # train
        if use_pca:
            y_train_pred = gbm.predict(self.dataset.X_train_pca, num_iteration=gbm.best_iteration)
            y_pred = gbm.predict(self.dataset.X_test_pca, num_iteration=gbm.best_iteration)
        else:
            y_pred = gbm.predict(self.dataset.X_test, num_iteration=gbm.best_iteration)
            y_train_pred = gbm.predict(self.dataset.X_train, num_iteration=gbm.best_iteration)

            ax = lgb.plot_tree(gbm, tree_index=self.param['num_boost_round'] - 1, figsize=(20, 8),
                               show_info=['split_gain'])
            plt.show()
            print('feature importance...')
            ax = lgb.plot_importance(gbm, max_num_features=10)
            plt.show()
        # eval
        print('result for GBDT: ')
        print('test r2: ',r2_score(self.dataset.y_test, y_pred))
        print('train r2: ',r2_score(self.dataset.y_train, y_train_pred))
        print('mse for test: ', mean_squared_error(self.dataset.y_test, y_pred))
        print('mse for train: ',mean_squared_error(self.dataset.y_train, y_train_pred))
        print('mae for test: ', mean_absolute_error(self.dataset.y_test, y_pred))
        print('mae for train: ',mean_absolute_error(self.dataset.y_train, y_train_pred))
        return gbm

    def grid_search(self, params, param_grid, use_pca=False):
        estimator = lgb.LGBMRegressor(**params)
        gbm = GridSearchCV(estimator, param_grid)

        if use_pca:
            print('using PCA...')
            self.dataset.pca_processor()
            gbm.fit(self.dataset.X_train_pca, self.dataset.y_train,
                    eval_set=[(self.dataset.X_test_pca, self.dataset.y_test)],
                    eval_metric='l2')
        else:
            gbm.fit(self.dataset.X_train, self.dataset.y_train,
                    eval_set=[(self.dataset.X_test, self.dataset.y_test)],
                    eval_metric='l2')

        print('best params:')
        print(gbm.best_params_)


class LR_learner:
    dataset = None

    def __init__(self,dataset):
        self.dataset = dataset

    def fit(self, Ridge=True,alpha=0.5):
        scaler = preprocessing.StandardScaler().fit(self.dataset.X_train)
        x_train_scaled = scaler.transform(self.dataset.X_train)
        x_test_scaled = scaler.transform(self.dataset.X_test)
        if Ridge:
            reg = linear_model.Ridge(alpha=alpha)
        else:
            reg = LinearRegression()

        reg.fit(x_train_scaled, self.dataset.y_train)
        # print(reg.coef_)
        # print(reg.intercept_)
        y_train_predict = reg.predict(x_train_scaled)
        y_test_predict = reg.predict(x_test_scaled)
        print('result for OLS/Ridge: ')
        print('test r2: ', r2_score(self.dataset.y_test, y_test_predict))
        print('train r2: ',r2_score(self.dataset.y_train, y_train_predict))
        print('mse for test: ', mean_squared_error(self.dataset.y_test, y_test_predict))
        print('mse for train: ',mean_squared_error(self.dataset.y_train, y_train_predict))
        print('mae for test: ', mean_absolute_error(self.dataset.y_test, y_test_predict))
        print('mae for train: ',mean_absolute_error(self.dataset.y_train, y_train_predict))
