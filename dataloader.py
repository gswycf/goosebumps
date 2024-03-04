import torch
import pandas

def read_xlsx(path):
    wb = pandas.read_excel(path, sheet_name = 'Sheet1')
    head_list = wb.head()
    data={}
    for line in wb.head():
        data[line]=wb[line]
    print(len(data['rthigh']))
    print(type(data['rthigh']))


def read_txt(path):
    data=[]
    with open(path) as f:
        for line in f.readlines():
            line= line.strip("\t").strip("\n").split("\t")
            data.append(line)
    return data



def split_data(data):
    data_len = len(data)


if __name__ == '__main__':
    data = read_txt('data.txt')