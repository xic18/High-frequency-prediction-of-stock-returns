import pandas as pd
import numpy as np
import collections
from tqdm import tqdm


class Order:
    def __init__(self, appl_seq_num, price, order_qty, side, order_type, transact_time):
        self.appl_seq_num = appl_seq_num
        self.price = price
        self.order_qty = order_qty
        self.side = side
        self.order_type = order_type
        self.transact_time = transact_time


def print_orderbook(order_book,log):
    '''
    compute and print orderbook data to file

    Parameters
    ----------
    order_book : Order
        Order object needed to be output
    log: 
        output file
    '''

    sell_list = collections.OrderedDict(sorted(order_book['sell'].items(), key=lambda t: t[1].price))
    sell_price_list = []
    qty = 0
    for i, x in enumerate(sell_list.values()):
        if i == len(sell_list) - 1 or x.price != sell_list[list(sell_list.keys())[min(len(sell_list) - 1, i + 1)]].price:
            price = x.price
            qty += x.order_qty
            sell_price_list.append((price, qty))
            qty = 0
        else:
            price = x.price
            qty += x.order_qty
        if len(sell_price_list) >= 10:
            break
    while len(sell_price_list) < 10:
        sell_price_list.append(((np.nan, 0)))

    print('sell list:',file = log)
    for x in sell_price_list[::-1]:
        print(x,file = log)
    print('-----------------------------------------------------------------',file = log)

    buy_list = collections.OrderedDict(sorted(order_book['buy'].items(), key=lambda t: -t[1].price))
    buy_price_list = []
    qty = 0
    for i, x in enumerate(buy_list.values()):
        if i == len(buy_list) - 1 or x.price != buy_list[list(buy_list.keys())[min(len(buy_list) - 1, i + 1)]].price:
            price = x.price
            qty += x.order_qty
            buy_price_list.append((price, qty))
            qty = 0
        else:
            price = x.price
            qty += x.order_qty
        if len(buy_price_list) >= 10:
            break
    while len(buy_price_list) < 10:
        buy_price_list.append(((np.nan, 0)))

    print('buy list:',file = log)
    for x in buy_price_list:
        print(x,file = log)

    print(file = log)


def gen_oderbook_from_file(filename,outdir):
    '''
    Generate orderbook at certain time.

    Parameters
    ----------
    filename: str
        input file path
    outdir: str
        output file path
    '''

    df = pd.read_csv(filename)
    order_book = {'buy': {}, 'sell': {}}
    temp = 0
    log = open(outdir, mode="a", encoding="utf-8")
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):

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

        # print order book
        if row['transact_time'] >= 20200110093000000 and temp < 20200110093000000:
            print('order book at:', row['transact_time'],file = log)
            print_orderbook(order_book,log=log)
        if row['transact_time'] >= 20200110103000000 and temp < 20200110103000000:
            print('order book at:', row['transact_time'],file = log)
            print_orderbook(order_book,log=log)
        if row['transact_time'] >= 20200110133000000 and temp < 20200110133000000:
            print('order book at:', row['transact_time'],file = log)
            print_orderbook(order_book,log=log)

        temp = row['transact_time']
    log.close()


gen_oderbook_from_file('../data/000069_20200110.csv','../res/000069_20200110.txt')
gen_oderbook_from_file('../data/000566_20200110.csv','../res/000566_20200110.txt')
gen_oderbook_from_file('../data/000876_20200110.csv','../res/000876_20200110.txt')
gen_oderbook_from_file('../data/002304_20200110.csv','../res/002304_20200110.txt')
gen_oderbook_from_file('../data/002841_20200110.csv','../res/002841_20200110.txt')
gen_oderbook_from_file('../data/002918_20200110.csv','../res/002918_20200110.txt')

