import torch
import pandas

def read_xlsx(path):
    wb = pandas.read_excel(path, sheet_name = 'Sheet1')
    head_list = wb.head()
    data={}
    for line in wb.head():
        data[line]=wb[line]
    print(data['rthigh'])


if __name__ == '__main__':
    read_xlsx('103.xlsx')