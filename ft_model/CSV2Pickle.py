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
    if not os.path.isdir(os.getcwd() + "\\demo_data"):
        os.mkdir(os.getcwd() + "\\demo_data")
    os.chdir(os.getcwd() + "\\demo_data")
    print(os.getcwd() + "\\demo_data")
    regex = re.compile("csv")
    raw_dict = {}
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if re.search(regex, file):
                raw_dict[file.split(".")[0]] = pd.read_csv(file)
    # print(raw_dict)


if __name__ == '__main__':
    os.chdir(os.getcwd() + "\\demo_data")
    print(os.mkdir("demo_test"))
    # for root, dirs, files in os.walk(os.getcwd()):
    #     print(files)
    for file in os.listdir(os.getcwd()):
        print(file)
    # csv2pickle()
    os.chdir("..")
    print(os.getcwd())
