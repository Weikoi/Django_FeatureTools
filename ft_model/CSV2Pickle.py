"""
由 csv 转成 pickle 文件，由 django 来load

其中保存的是dict结构

dict 的 key 是 表名也就是文件名
dict 的 value 是由表转成的 dataframe 结构

"""
import pandas as pd
import pickle as pk
import os
import re


def csv2pickle():
    os.chdir(os.getcwd() + "demo_data")

    regex = re.compile("csv")
    pickle_dict = {}
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if re.search(regex, file):
                pickle_dict[file.split(".")[0]] = pd.read_csv(file)

    pk.dump(pickle_dict, file=open("data_pickle.pkl", "wb"))

