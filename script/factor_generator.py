import pandas as pd
import numpy as np
import collections
import datetime
from tqdm import tqdm


class Order:
    def __init__(self, appl_seq_num, price, order_qty, side, order_type, transact_time):
        self.appl_seq_num = appl_seq_num
        self.price = price
        self.order_qty = order_qty
        self.side = side
        self.order_type = order_type
        self.transact_time = transact_time


class Orderbook_tick:
    buy_price = None
    sell_price = None
    buy_qty = None
    sell_qty = None
    latest_price = None
    acc_qty = None
    transact_time = None

    def __init__(self, latest_price=None, acc_qty=None,transact_time=None,
                 buy_price=[], sell_price=[], buy_qty=[], sell_qty=[]):
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.buy_qty = buy_qty
        self.sell_qty = sell_qty
        self.latest_price = latest_price
        self.acc_qty = acc_qty
        self.transact_time = transact_time


class Factors:
    buy_price = None
    sell_price = None
    buy_qty = None
    sell_qty = None
    spread = None
    depth = None
    relative_spread = None
    slope = None
    latest_price = None
    acc_qty = None
    transact_time = None

    buy_price_log = None
    sell_price_log = None
    buy_qty_log = None
    sell_qty_log = None

    VOI = None
    QR = None
    HR = None
    MID = None
    PRESS = None

    WP_2_4 = None
    WP_5_10 = None

    def __init__(self, orderbook_tick,orderbook_tick_last,num_price):
        self.buy_price = orderbook_tick.buy_price
        self.sell_price = orderbook_tick.sell_price
        self.buy_qty = orderbook_tick.buy_qty
        self.sell_qty = orderbook_tick.sell_qty
        self.latest_price = orderbook_tick.latest_price
        self.acc_qty = orderbook_tick.acc_qty
        self.transact_time = orderbook_tick.transact_time

        buy_price = orderbook_tick.buy_price
        last_buy_price = orderbook_tick_last.buy_price
        self.buy_price_log = [np.log(x) - np.log(y) for x, y in zip(buy_price, last_buy_price)]
        sell_price = orderbook_tick.sell_price
        last_sell_price = orderbook_tick_last.sell_price
        self.sell_price_log = [np.log(x) - np.log(y) for x, y in zip(sell_price, last_sell_price)]
        buy_qty = orderbook_tick.buy_qty
        last_buy_qty = orderbook_tick_last.buy_qty
        self.buy_qty_log = [np.log(x) - np.log(y) for x, y in zip(buy_qty, last_buy_qty)]
        sell_qty = orderbook_tick.sell_qty
        last_sell_qty = orderbook_tick_last.sell_qty
        self.sell_qty_log = [np.log(x) - np.log(y) for x, y in zip(sell_qty, last_sell_qty)]

        self.spread = sell_price[0] - buy_price[0]
        self.depth = (sell_qty[1] +buy_qty[1]) / 2
        self.relative_spread = self.spread / (sell_price[0] + buy_price[0]) * 2
        self.slope = self.spread / self.depth

        if buy_price[0]<last_buy_price[0]:
            VOI_buy = 0
        elif buy_price[0]==last_buy_price[0]:
            VOI_buy = buy_qty[0]-last_buy_qty[0]
        else:
            VOI_buy = buy_qty[0]
        if sell_price[0]>last_sell_price[0]:
            VOI_sell = 0
        elif sell_price[0]==last_sell_price[0]:
            VOI_sell = sell_qty[0]-last_sell_qty[0]
        else:
            VOI_sell = sell_qty[0]
        self.VOI= VOI_buy-VOI_sell

        self.QR=[]
        self.HR=[]
        for i in range(num_price):
            self.QR.append((buy_qty[i]-sell_qty[i])/(buy_qty[i]+sell_qty[i]))
            if i!=0:
                self.HR.append((buy_price[i-1] - buy_price[i]+sell_price[i-1]-sell_price[i]) / (buy_price[i-1] - buy_price[i]+sell_price[i]-sell_price[i-1]))

        self.MID=(buy_price[0]+sell_price[0])/2

        buy_weights=[self.MID/(x-self.MID) for x in buy_price]
        buy_weights=[x/sum(buy_weights) for x in buy_weights]
        buy_press=0
        sell_weights = [self.MID / (x - self.MID) for x in sell_price]
        sell_weights = [x / sum(sell_weights) for x in sell_weights]
        sell_press = 0
        for i in range(num_price):
            buy_press+=buy_weights[i]*buy_qty[i]
            sell_press += sell_weights[i] * sell_qty[i]

        self.PRESS= np.log(buy_press)-np.log(sell_press)

        self.WP_2_4 = \
            sum([buy_qty[i]*buy_price[i]+sell_qty[i]*sell_price[i] for i in range(num_price)])/(sum(buy_qty[1:4])+sum(sell_qty[1:4]))

        self.WP_5_10 = \
            sum([buy_qty[i]*buy_price[i]+sell_qty[i]*sell_price[i] for i in range(num_price)])/(sum(buy_qty[4:])+sum(sell_qty[4:]))


class Data_generator:
    file_dir = None
    freq = None  # 1000ms
    num_price = None
    year = None
    month = None
    day = None
    dataset_df = None

    def __init__(self,file_dir = None, freq = 1000 , num_price=10, year=2020,month=1,day=10):
        self.file_dir = file_dir
        self.freq = freq  # 1000ms
        self.num_price = num_price
        self.year=year
        self.month=month
        self.day=day

    def gen_orderbook_tick(self,order_book,latest_price,acc_qty,row):
        # gen factors
        oderbook_tick = Orderbook_tick(buy_price=[], sell_price=[], buy_qty=[], sell_qty=[])
        buy_dict = collections.OrderedDict(sorted(order_book['buy'].items(), key=lambda t: -t[1].price))
        sell_dict = collections.OrderedDict(sorted(order_book['sell'].items(), key=lambda t: t[1].price))
        buy_id = 0
        sell_id = 0
        buy_price = latest_price-100
        sell_price = latest_price+100
        for j in range(self.num_price):
            if buy_id>=len(list(buy_dict.keys())):
                buy_qty=0
            else:
                buy_price = buy_dict[list(buy_dict.keys())[buy_id]].price
                buy_qty = buy_dict[list(buy_dict.keys())[buy_id]].order_qty
                buy_id+=1
                if buy_id<len(list(buy_dict.keys())):
                    while buy_dict[list(buy_dict.keys())[buy_id]].price==buy_price:
                        buy_qty += buy_dict[list(buy_dict.keys())[buy_id]].order_qty
                        buy_id += 1
                        if buy_id>=len(list(buy_dict.keys())):
                            break

            if sell_id>=len(list(sell_dict.keys())):
                sell_qty=0
            else:
                sell_price = sell_dict[list(sell_dict.keys())[sell_id]].price
                sell_qty = sell_dict[list(sell_dict.keys())[sell_id]].order_qty
                sell_id += 1
                if sell_id < len(list(sell_dict.keys())):
                    while sell_dict[list(sell_dict.keys())[sell_id]].price==sell_price:
                        sell_qty += sell_dict[list(sell_dict.keys())[sell_id]].order_qty
                        sell_id += 1
                        if sell_id >= len(list(sell_dict.keys())):
                            break

            oderbook_tick.buy_price.append(buy_price)
            oderbook_tick.sell_price.append(sell_price)
            oderbook_tick.buy_qty.append(buy_qty)
            oderbook_tick.sell_qty.append(sell_qty)
        oderbook_tick.latest_price = latest_price
        oderbook_tick.acc_qty = acc_qty
        oderbook_tick.transact_time = row['transact_time']
        return oderbook_tick

    def gen_csv(self, out_dir = None):

        print('loading price data: ',self.file_dir)
        df = pd.read_csv(self.file_dir)
        order_book = {'buy': {}, 'sell': {}}
        acc_qty = 0
        latest_price= np.nan
        print('generate order book info...')
        start_time = datetime.datetime(self.year, self.month, self.day, 9, 30)
        end_time = datetime.datetime(self.year, self.month, self.day, 14, 57)
        x = start_time
        time_list = []
        while x <= end_time:
            time_list.append(int(x.strftime('%Y%m%d%H%M%S000')))
            if x== datetime.datetime(self.year, self.month, self.day, 11, 30):
                x=datetime.datetime(self.year, self.month, self.day, 13, 0)
            x += datetime.timedelta(seconds=self.freq / 1000)
        time_id = 0
        orderbook_list = []

        for i, row in tqdm(df.iterrows(),total=df.shape[0]):
            # update order book
            # transaction
            if row['order_type'] == 'F':
                bid_appl_seq_num = int(row['bid_appl_seq_num'])  # buy
                offer_appl_seq_num = int(row['offer_appl_seq_num'])  # sell

                order_book['buy'][bid_appl_seq_num].order_qty -= row['order_qty']
                if order_book['buy'][bid_appl_seq_num].order_qty == 0:
                    del order_book['buy'][bid_appl_seq_num]
                order_book['sell'][offer_appl_seq_num].order_qty -= row['order_qty']
                if order_book['sell'][offer_appl_seq_num].order_qty == 0:
                    del order_book['sell'][offer_appl_seq_num]
                latest_price = row['price']
                acc_qty += row['order_qty']
            # cancel order
            elif row['order_type'] == '4':
                if row['bid_appl_seq_num'] != 0:
                    del order_book['buy'][int(row['bid_appl_seq_num'])]
                elif row['offer_appl_seq_num'] != 0:
                    del order_book['sell'][int(row['offer_appl_seq_num'])]
            # new order
            else:
                order = Order(row['appl_seq_num'], row['price'], row['order_qty'], row['side'], row['order_type'],
                              row['transact_time'])
                if order.side == 1:
                    order_book['buy'][order.appl_seq_num] = order
                if order.side == 2:
                    order_book['sell'][order.appl_seq_num] = order

            if df.iloc[min(i+1,len(df)-1)]['transact_time'] < 20200110093000000:
                continue

            if i==len(df)-1:
                # gen factors
                orderbook_tick = self.gen_orderbook_tick(order_book,latest_price,acc_qty,row)
                while time_id < len(time_list):
                    orderbook_list.append(orderbook_tick)
                    time_id += 1
            elif row['transact_time'] == df.iloc[i + 1]['transact_time']:
                continue
            elif row['transact_time'] <= time_list[time_id] and df.iloc[i + 1]['transact_time'] > time_list[time_id]:
                # gen factors
                orderbook_tick = self.gen_orderbook_tick(order_book,latest_price,acc_qty,row)
                while df.iloc[i + 1]['transact_time'] > time_list[time_id]:
                    orderbook_list.append(orderbook_tick)
                    time_id += 1


        print('processing features...')
        dataset_df=pd.DataFrame(data={'time':time_list,'orderbook':orderbook_list})
        price_list=[x.latest_price for x in orderbook_list]
        dataset_df['price']=price_list
        factor_list=[]

        for i, row in tqdm(dataset_df.iterrows(), total=dataset_df.shape[0]):
            if i == 0:
                continue
            factors=Factors(row['orderbook'],dataset_df.iloc[i - 1]['orderbook'],self.num_price)
            factor_list.append(factors)

        dataset_df['label'] = dataset_df['price'].pct_change().shift(-1)
        dataset_df = dataset_df.iloc[1:] 

        dataset_df['spread'] = [x.spread for x in factor_list]
        dataset_df['depth'] = [x.depth for x in factor_list]
        dataset_df['relative_spread'] = [x.relative_spread for x in factor_list]
        dataset_df['slope'] = [x.slope for x in factor_list]
        dataset_df['latest_price'] = [x.latest_price for x in factor_list]
        dataset_df['acc_qty'] = [x.acc_qty for x in factor_list]
        dataset_df['VOI'] = [x.VOI for x in factor_list]
        dataset_df['MID'] = [x.MID for x in factor_list]
        dataset_df['PRESS'] = [x.PRESS for x in factor_list]
        dataset_df['WP_2_4'] = [x.WP_2_4 for x in factor_list]
        dataset_df['WP_5_10'] = [x.WP_5_10 for x in factor_list]
        dataset_df['spread'] = [x.spread for x in factor_list]


        for i in range(self.num_price):
            dataset_df['buy_price_'+str(i+1)]=[x.buy_price[i] for x in factor_list]
            dataset_df['sell_price_'+str(i+1)]=[x.sell_price[i] for x in factor_list]
            dataset_df['buy_qty_'+str(i+1)]=[x.buy_qty[i] for x in factor_list]
            dataset_df['sell_qty_'+str(i+1)]=[x.sell_qty[i] for x in factor_list]
            dataset_df['buy_price_log_'+str(i+1)]=[x.buy_price_log[i] for x in factor_list]
            dataset_df['sell_price_log_'+str(i+1)]=[x.sell_price_log[i] for x in factor_list]
            dataset_df['buy_qty_log_'+str(i+1)]=[x.buy_qty_log[i] for x in factor_list]
            dataset_df['sell_qty_log_'+str(i+1)]=[x.sell_qty_log[i] for x in factor_list]
            dataset_df['QR_' + str(i + 1)] = [x.QR[i] for x in factor_list]
            if i != 0:
                dataset_df['HR_' + str(i + 1)] = [x.HR[i-1] for x in factor_list]

        dataset_df = dataset_df.iloc[:-1] 
        dataset_df.set_index(keys="time", inplace=True)
        print('output csv: ',out_dir)
        dataset_df[dataset_df.columns[2:]].to_csv(out_dir)
        print(dataset_df[dataset_df.columns[2:]])
        

