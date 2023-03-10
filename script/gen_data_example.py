from factor_generator import Data_generator
import os

# generate 1s data for stock 000069
path = '../data/sz_level3/000566/'
for file_name in os.listdir(path):

    stock=file_name[:6]
    time=file_name[7:15]
    print(file_name)
    gen=Data_generator(file_dir = path+file_name, freq = 1000 , num_price=10,
                       year=int(time[:4]),month=int(time[4:6]),day=int(time[6:]))
    gen.gen_csv('../data/ex1/000566_1s_extra/'+file_name)

