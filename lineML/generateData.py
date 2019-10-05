#用于模型分析
import statsmodels.api as sm
#用于矩阵操作
import numpy as np

import pandas as pd


'''

检验假设分析为了可以解决无关变量的排出


'''
def generateData():
    """
    生成z的随机矩阵
    :return:
    测试用例
    """
    np.random.seed(5320) #设定随机种子，实际操作中把时间设置随机种子
    x = np.array(range(0,20))/2

    '''
    np.round() 四舍五入的算法
    第二个参数：保留的小数点的位数
    mp.random.randn()  产生标准正态分布
    第一参数：随机的个数
    '''
    error = np.round(np.random.randn(20),2) #四舍五入小数
    print(error)
    y = 0.05*x+error
    z = np.zeros(20)+1
    return pd.DataFrame({"x":x,"z":z,"y":y})

def wrongCoef():
    """

    :return:
    """
    features =["x","z"] # 输入特征列表
    labels = ["y"]
    data = generateData()
    X = data[features]
    Y = data[labels]

    model = sm.OLS(Y,X["x"])
    res = model.fit()
    print("没有加入z")
    print(res.summary())
    model1 = sm.OLS(Y,X)
    res1 = model1.fit()
    print("添加z 之后 ")
    print(res1.summary())
if __name__ == '__main__':
    wrongCoef()
