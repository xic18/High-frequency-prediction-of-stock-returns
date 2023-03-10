import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def back_test(file,window=120,threshold=2,n=3600,fee=10):
    """
    Conduct backtest for certain stock represented by file name.
    Summarize winning rate, win to loss ratio and other metrics.
    Plot execution signal for both buy and sell on price series.

    Parameters
    ----------
    file : str
        file name of certain stock with prediction information
    window : int
        window length to check historical signal value
    threshold : int or float
        threshold to determine signal
    n: int
        number of data point in plot
    fee: int or float
        transaction fee (default*10000)
    """

    df_backtest=pd.read_csv(file,index_col=0)
    cnt_1 = 0
    right_1 = 0
    wrong_1 = 0
    cnt_0 = 0
    right_0 = 0
    wrong_0 = 0
    win = 0
    loss = 0
    df_backtest['mean'] = df_backtest['y_pred'].rolling(window).mean()
    df_backtest['std'] = df_backtest['y_pred'].rolling(window).std()

    buy_list = []
    sell_list = []
    for i, row in tqdm(df_backtest.iterrows(), total=len(df_backtest)):
        buy = np.nan
        sell = np.nan
        if (row['y_pred'] - row['mean']) / row['std'] > threshold:
            cnt_1 += 1
            buy = row['price']
            if row['y_test'] > 0:
                right_1 += 1
                win += row['y_test'] - fee
            elif row['y_test'] < 0:
                wrong_1 += 1
                loss += -row['y_test'] + fee
            else:
                loss += fee
        elif (row['y_pred'] - row['mean']) / row['std'] < -threshold:
            cnt_0 += 1
            sell = row['price']
            if row['y_test'] < 0:
                right_0 += 1
                win += -row['y_test'] - fee
            elif row['y_test'] > 0:
                wrong_0 += 1
                loss += row['y_test'] + fee
            else:
                loss += fee

        buy_list.append(buy)
        sell_list.append(sell)
    df_backtest['buy'] = buy_list
    df_backtest['sell'] = sell_list
    print('buy times: ',cnt_1)
    print('sell times: ', cnt_0)
    print('buy correct/false rate: ',right_1 / cnt_1, wrong_1 / cnt_1)
    print('sell correct/false rate: ',right_0 / cnt_0, wrong_0 / cnt_0)
    print('win_loss rate: ', win / loss)

    # plot
    fig, ax = plt.subplots(figsize=(30, 6))

    plt.plot(df_backtest.iloc[:n]['price'])
    plt.scatter(range(n),df_backtest.iloc[:n]['buy'],c='r')
    plt.scatter(range(n),df_backtest.iloc[:n]['sell'],c='g')
    ax.grid()
    plt.show()


back_test('./direction.csv')