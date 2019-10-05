"""
读取数据类型
"""

import pandas as pd
import  matplotlib.pyplot as plt
from  statsmodels.graphics.mosaicplot import  mosaic

def readDataFromCSV(path):
    data = pd.read_csv(path,header=0)
    cols = ['age', 'education_num','capital_gain','capital_loss','hours_per_week','label']

    data =data[cols]
    data["label_code"] = pd.Categorical(data["label"]).codes
    return data
if __name__ == '__main__':
    path = 'adult.csv'
    data = readDataFromCSV(path)
    #print(data.head(8))
    data['label_code'] = pd.Categorical(data["label"]).codes
    #print(data[["label","label_code"]])
    data[['age','hours_per_week','education_num','label_code']].hist()
    plt.show(block=False)
   # print(data.describe())# 数据统计的基本信息
    #计算交叉报表
    cross1 = pd.crosstab(pd.qcut(data['education_num'],[0,0.25,0.5,.75,1]),data['label'])
    #print(cross1.head())
    print(cross1.stack())
    mosaic(cross1.stack())
    plt.show()