from learner import Data_loader,gbdt_learner,LR_learner
from factor_generator import Data_generator

# 1. gen data

gen=Data_generator(file_dir = '../data/sz_level3/000069/000069_20200110.csv',
                   freq = 1000 , num_price=10, year=2020,month=1,day=10)
gen.gen_csv('../data/ex1/000069_1s_extra/000069_20200110.csv')

# 2. train

params = {
    'task': 'train',
    'boosting_type': 'dart',  # 设置提升类型
    'objective': 'regression', # 目标函数
    'metric': {'l2'},  # 评估函数
    'num_leaves': 31,   # 叶子节点数
    'min_data_in_leaf': 2000,
    'max_depth':5,
    'learning_rate':  0.01,  # 学习速率
    'feature_fraction': 0.9, # 建树的特征选择比例
    'bagging_fraction': 0.8, # 建树的样本采样比例
    'verbose': 10, # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    'early_stopping_rounds':30,
    "random_seed":2,
    'num_boost_round':1000 #
}

param_grid = {
            'num_leaves': [31, 63, 127],  # 叶子节点数
            'min_data_in_leaf': [200, 500, 2000, 5000],
            'max_depth': [5, 6, 10],
        }

# load dataset
test_dataloder=Data_loader('../data/ex1/000069_1s_extra/',split=10,num_feature=110)

# plot corr matrix
test_dataloder.corr_plot()

# define learner ---gbdt
test_gbdt=gbdt_learner(test_dataloder,params)

# grid search
test_gbdt.grid_search(params=params, param_grid=param_grid, use_pca=False)

# fit gbdt
model=test_gbdt.fit(use_pca=True)
# save
model.save_model('model.txt')

# define learner --- OLS or Ridge
test_LR=LR_learner(test_dataloder)
test_LR.fit(Ridge=False)
test_LR.fit(Ridge=True)

