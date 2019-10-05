from statsmodels import api as sm
from lineML.read_csv import read_csv_file
import sys
import matplotlib.pyplot as plt
from  statsmodels.sandbox.regression.predstd import wls_prediction_std


def linearModelWithState(data):
    """
    线性回归分析步骤
    :param data:  建模数据
    :return:
    """
    features = ["x"]
    labels = ["y"]
    Y = data[labels]
    X = sm.add_constant(data[features])
    print("X:")
    print(X)
    print("Y:")
    print(Y)
    re = trainModel(X, Y)
    # 模型状态分析：
    modelSummary(re)
    #搭建的模型

    re_new = trainModel(data[features],Y)
    print(re_new.summary())
    visualizeModel(re_new,data, features,labels)

def trainModel(X, Y):
    """
    训练模型
    :param X: 训练数据集
    :param Y:  标签数据（真实值）
    :return:
    model: 返回模型
    """
    #创建线性回归
    model = sm.OLS(Y, X)
    # OLS普通小二乘法
    # WLS 加权小二乘法
    # 训练数据拟合
    re = model.fit()
    return re
def modelSummary(re):
    """
    分析线性的回归方程的统计性质
    :param model:
    :return:
    """
    print("整体分析：")
    print(re.summary())
    # 假定系数a=0评估
    print("检验假设x的系数等于0")
    print(re.f_test("x=0"))
    # 检测 截距 b=0 是否显著
    print("检验假设x的系数等于1 和 const的系数等于0")
    print(re.f_test("const=0"))
    # 检测 a=1 ,b =0 同时成立的显著性
    print(re.f_test(["x=1","const=0"]))

def visualizeModel(re, data, features, labels):
    """
    模型可视化
    """
    #计算预测结果的标准差，预测下界，预测上界

    prstd,preLow,preUp = wls_prediction_std(re, alpha=0.05)
    #为在matlabplot 中显示中,设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    #创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(111)

    if sys.version_info[0] == 3:
        ax.set_title(u'%s' %"线性回归统计分析实例")
    else:
        ax.set_title(u'%s' %"线性回归统计分析实例".decode("utf-8"))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    if sys.version_info[0] == 3:
        ax.scatter(data[features],data[labels],color='b',label=u'%s: $y = x+ \epsilon $'%"真实值")
    else:
        ax.scatter(data[features], data[labels], color='b', label=u'%s: $y = x+ \epsilon $' % "真实值".decode("utf-8"))

    if sys.version_info[0] == 3 :
        ax.plot(data[features],preUp,"r--",label=u'%s' %"95%置信区间")
        ax.plot(data[features],re.predict(data[features]),color="red", label=u'%s: $y = %.3fx$' %("预测值", re.params[features]))
    else:
        ax.plot(data[features], preUp, "r--", label=u'%s' % "95%置信区间".decode("utf-8"))
        ax.plot(data[features], re.predict(data[features]), color="red",
                label=u'%s: $y = %.3fx$' % ("预测值".decode("utf-8"), re.params[features]))
    ax.plot(data[features], preLow,'r--')
    legend = plt.legend(shadow=True)
    legend.get_frame().set_facecolor("#6f93AE")
    plt.show()

if __name__ == "__main__":
    path='./x_y_data.csv'
    data = read_csv_file(path,header=0)
    print(data)
    linearModelWithState(data)