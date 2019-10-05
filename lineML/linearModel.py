from sklearn import linear_model
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


from lineML.read_csv import read_csv_file


def linerModel(data):
    '''
    线性回归模型建模步骤展示

    :param data:  DataFram ,建模数据
    :return:
    '''

    features = ["x"]
    labels = ["y"]

    #前十五列数据训练
    trainData = data[:15]

    #后15列数据测试
    testData = data[15:]
    model = trainModel(trainData,features,labels)
    error,score = evaluateModel(model,testData,features,labels)
    print("error:")
  #  print(error)
    print("score:")
   # print(score)
    visualizeModel(model,data,features,labels,error,score)

def  trainModel(trainData,features, labels):
    """
    利用训练数据，估计模型的参数
    :param trainData:  训练数据
    :param features:  特征名列表
    :param labels:  标签列表
    :return:  model
    """
#创建数据模型
    model = linear_model.LinearRegression()
#训练数据模型，估计数据模型参数
    model.fit(trainData[features],trainData[labels])
    return model



def evaluateModel(model, testData, features, labels):
    """
    计算线性模型的均方差和决定系数
    :param model: 线性回归模型
    :param testData: 测试数据
    :param features: 特征名标签
    :param labels: 标标签
    :return:
    error : 均方差

    score: 决定系数
    """
    print((model.predict(testData[features])-testData[labels])**2)
    error = np.mean((model.predict(testData[features])-testData[labels])**2)
    score = model.score(testData[features],testData[labels])
    return error, score

def visualizeModel(model, data, features, labels, error, score):
    """
    模型可视化
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams['font.sans-serif']=['SimHei']
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(111)
    # 在Matplotlib中显示中文，需要使用unicode
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.set_title(u'%s' % "线性回归示例")
    else:
        ax.set_title(u'%s' % "线性回归示例".decode("utf-8"))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    # 画点图，用蓝色圆点表示原始数据
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:   #如果python 3
        ax.scatter(data[features], data[labels], color='b',
            label=u'%s: $y = x + \epsilon$' % "真实值")
    else:
        ax.scatter(data[features], data[labels], color='b',
            label=u'%s: $y = x + \epsilon$' % "真实值".decode("utf-8"))
    # 根据截距的正负，打印不同的标签
    if model.intercept_ > 0:
        # 画线图，用红色线条表示模型结果
        # 在Python3中，str不需要decode
        if sys.version_info[0] == 3:
            ax.plot(data[features], model.predict(data[features]), color='r',
                label=u'%s: $y = %.3fx$ + %.3f'\
                % ("预测值", model.coef_, model.intercept_))
            # coef , intercept : 系数，截距
        else:
            ax.plot(data[features], model.predict(data[features]), color='r',
                label=u'%s: $y = %.3fx$ + %.3f'\
                % ("预测值".decode("utf-8"), model.coef_, model.intercept_))
    else:
        # 在Python3中，str不需要decode
        if sys.version_info[0] == 3:
            ax.plot(data[features], model.predict(data[features]), color='r',
                label=u'%s: $y = %.3fx$ + （%.3f)'\
                % ("预测值", model.coef_, model.intercept_))
        else:
            ax.plot(data[features], model.predict(data[features]), color='r',
                label=u'%s: $y = %.3fx$ -(%.3f)'\
                % ("预测值".decode("utf-8"), model.coef_, abs(model.intercept_)))
            # coef :预测系数，
    legend = plt.legend(shadow=True)
    legend.get_frame().set_facecolor('#6F93AE')
    # 显示均方差和决定系数
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.text(0.99, 0.01,
            u'%s%.3f\n%s%.3f'\
            % ("均方差：", error, "决定系数：", score),
            style='italic', verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, color='m', fontsize=13)
    else:
         ax.text(0.99, 0.01,
            u'%s%.3f\n%s%.3f'\
            % ("均方差：".decode("utf-8"), error, "决定系数：".decode("utf-8"), score),
            style='italic', verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, color='m', fontsize=13)
    # 展示上面所画的图片。图片将阻断程序的运行，直至所有的图片被关闭
    # 在Python shell里面，可以设置参数"block=False"，使阻断失效。
    plt.show()


if __name__=='__main__':
    path = './x_y_data.csv'
    data=read_csv_file(path=path,header=0)
    linerModel(data)