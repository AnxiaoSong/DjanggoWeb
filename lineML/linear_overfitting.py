import  os
import sys

import numpy as np

import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


def evaluateModel(model, testData, features,labels, featurizer):


    """
    计算模型的均方差和决定系数
    :param model:   训练完成的线性模型
    :param testData:  测试数据
    :param features:  特征名列表
    :param labels: 标签列表
    :param featurizer:  多项式
    :return:
    error: np.float64 均方差
    score: np.float64 决定系数
    """
    #均方差
    error = np.mean((model.predict(featurizer.fit_transform(testData[features]))-testData[labels])**2)
    #决定系数
    score = model.score(featurizer.fit_transform(testData[features]),testData[labels])

    return error, score

def trainModel(trainData, features, labels ,featurizer):
    """
    模型训练
    :param trainData: 训练数据
    :param features:  特征列表
    :param labels:    标签列表
    :param featurizer: 多项式集合
    :return:

    model： 线性回归模型
    """
    #创建线性回归模型
    model = linear_model.LinearRegression(fit_intercept=False);

    #训练模型，估计模型参数
    """
    model.fit(x,y):有监督学习算法 ，model.fit(x）; 无监督学习算法：降维 特征提取，标准化
    
    transform 可以替换fit_transform()，反之不可以
    
    fit_transform() 先拟合数据，然后在转成标准形式
    每一个transform都需要先fit,比如把数据转为（0，1）分布，需要均值和标准差，
    fit_transform和transform的区别就是前者是先计算均值和标准差再转换，
    而直接transform则是用之前数据计算的参数转换。所以如果之前没有fit，是不能直接transform的
    
    transform() 通过找中心和缩放等实现标准化
    fit():可以求取训练X的均值，方差，最大值，最小值，这些训练集X固有的属性
    transform():在fit的基础上，进行标准化，降维，归一化等操作
    fit_transform :既包含训练又包含转换
    
    
    
    
    """
    #转换成标准数据然后再拟合训练
    model.fit(featurizer.fit_transform(trainData[features]), trainData[labels])
    return model

def visualizeModel(model,featurizer,data,features,labels,evaluation):
    """
    模型可视化化
    :param model:
    :param featurizer:
    :param data:
    :param features:
    :param labels:
    :param evaluation:
    :return:
    """
    #设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei']

    fig = plt.figure(figsize=(6,6),dpi=80)
    for i in  range(4):
        ax = fig.add_subplot(2,2,i+1)
        _visualization(ax, data, model[i], featurizer[i], evaluation[i], features, labels)
    plt.show()
def _visualization(ax,data,model,featurizer,evaluation,features,labels):
    '''
    画图
    :param ax:
    :param data:
    :param model:
    :param featurizer:
    :param evaluation:
    :param features:
    :param labels:
    :return:
    '''
    #画圆点，用蓝色的圆点画原始数据
    ax .scatter(data[features],data[labels],color='b') #画出离散图
    #画线图，使用红色表示模型结果
    ax.plot(data[features],model.predict(featurizer.fit_transform(data[features])),color='r') #画出连续图
    if sys.version_info[0] == 3:
        ax.text(0.01, 0.99,
                u'%s%.3f\n%s%.3f' \
                % ("均方差：", evaluation[0], "决定系数：", evaluation[1]),
                style="italic", verticalalignment="top", horizontalalignment="left",
                transform=ax.transAxes, color="m", fontsize=13)
    else:
        ax.text(0.01, 0.99,
                u'%s%.3f\n%s%.3f' \
                % ("均方差：".decode("utf-8"), evaluation[0],
                   "决定系数：".decode("utf-8"), evaluation[1]),
                style="italic", verticalalignment="top", horizontalalignment="left",
                transform=ax.transAxes, color="m", fontsize=13)
def overfitting(data):
    """
    过拟合多项式
    :param data:
    :return:
    """
    features =['x']
    labels= ['y']
    trainData= data[:15]
    testData =data[15:]

    featurizer = []
    overfittingModel = []
    overfittingEvaluation = []
    model = []
    evaluation = []
    for i in range(1, 11, 3):
        featurizer.append(PolynomialFeatures(degree=i))
        # 产生并训练模型
        overfittingModel.append(trainModel(trainData, features, labels, featurizer[-1]))
        model.append(trainModel(data, features, labels, featurizer[-1])) #不会出现过拟合的现象是由于0
        # 评价模型效果
        overfittingEvaluation.append(evaluateModel(overfittingModel[-1], testData, features, labels, featurizer[-1]))
        evaluation.append(evaluateModel(model[-1], data, features, labels, featurizer[-1]))
        # 图形化模型结果
    visualizeModel(model, featurizer, data, features, labels, evaluation)
    visualizeModel(overfittingModel, featurizer, data, features, labels, overfittingEvaluation)

if __name__ == '__main__':
    path="./x_y_data.csv"
    data = pd.read_csv(path,header=0)
    print([i for i in range(1,11,3)])
    overfitting(data)
