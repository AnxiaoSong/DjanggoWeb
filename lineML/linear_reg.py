"""
带有惩罚项： 一般使用范数来解决模型的约束性问题

"""
import  sys
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from lineML.read_csv import read_csv_file


def generateRandomVar():
     """
     随机产生随机数据
     :return:
     """
     np.random.seed(4872)
     """
    randint():随机产生整数
     """
     return np.random.randint(low=2,size=20)

def trainModel(X,Y):
    """
    训练模型
    :param X: 特征数据X
    :param Y: 测试数据Y
    :return:
    """
    model = sm.OLS(Y,X)
    #拟合模型
    res =model.fit()
    return res

def trainRegulizedModel(X,Y,alpha):
    """
    带有惩罚项的线性回归
    :param X:
    :param Y:
    :param alpha: 超参
    :return:
    """
    model = sm.OLS(Y,X)
    res = model.fit_regularized(alpha=alpha)
    return res

def visualizerModel(X,Y):
    """
    可化模型
    :param X:
    :param Y:
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    #正确显示中文符号字体
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(6,6),dpi=80)
    #图表中只画一幅图片
    ax = fig.add_subplot(1,1,1)
    # 随机生成100个10^-4 到10^-0,8之间的对数等分向量X

    alphas = np.logspace(-4,-0.8,100)
    coefs=[]
    for alpha in alphas:
        res = trainRegulizedModel(X,Y,alpha)
        coefs.append(res.params)
    coefs = np.array(coefs)
    if sys.version_info[0] == 3:
        ax.plot(alphas,coefs[:,1],"r:",label=u'%s'%"x的参数a")
        ax.plot(alphas,coefs[:,2],"g",label=u'%s'%"z的参数b")
        ax.plot(alphas,coefs[:,0],"b-.",label=u'%s'%'const的参数C')
    else:
        ax.plot(alphas, coefs[:, 1], "r:",
                label=u'%s' % "x的参数a".decode("utf-8"))
        ax.plot(alphas, coefs[:, 2], "g",
                label=u'%s' % "z的参数b".decode("utf-8"))
        ax.plot(alphas, coefs[:, 0], "b-.",
                label=u'%s' % "const的参数c".decode("utf-8"))
    legend = plt.legend(loc=4, shadow=True)
    legend.get_frame().set_facecolor("#6F93AE")
    ax.set_yticks(np.arange(-1, 1.3, 0.3))
    ax.set_xscale("log")
    ax.set_xlabel("$alpha$")
    plt.show()
def addReg(data):
    features =['x']
    labels= ['y']
    Y = data[labels]
    _X = data[features]
    _X['z'] = generateRandomVar()

    X= sm.add_constant(_X)
    res = trainRegulizedModel(X,Y,0.1)
    print("加入惩罚项（权重为0.1）的估计结果：\n%s" % trainRegulizedModel(X, Y, 0.1).params)
    visualizerModel(X,Y)
if __name__ == "__main__":
    path = "./x_y_data.csv"
    data = read_csv_file(path,header=0)
    addReg(data)