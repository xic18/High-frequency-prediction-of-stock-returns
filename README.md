# README

res目录下为每个个股的撮合结果，`directrion.csv`为个股000069在20200110的1s频率next return的预测结果，可通过`backtest.py`脚本执行回测；

script目录下为各种脚本文件，其中`get_orderbook.py`文件生成orderbook撮合结果，保存至res文件夹下的txt，`factor_generator.py`脚本生成相关数据集，示例见`gen_data_example.py`，`learner.py`定义模型以及相关方法，示例见`example.py`

data目录储存数据文件，由于文件较多，只保留了撮合20200110日orderbook的相关文件。
