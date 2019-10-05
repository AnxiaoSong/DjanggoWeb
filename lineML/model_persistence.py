"""
model的保存和读取
"""
import pandas as pd
import pickle
from sklearn import linear_model

from sklearn2pmml import PMMLPipeline

from  sklearn2pmml import sklearn2pmml


def savaAsPMMl(data, modelPath):
    """
    利用sklearn2pnml将模型存储为PMML
    :param data:
    :param modelPath:
    :return:
    """
    model = PMMLPipeline([
        ("regressor", linear_model.LinearRegression())])
    model.fit(data[["x"]], data["y"])
    sklearn2pmml(model,modelPath, with_repr=True)

def trainAndSaveModel(data,modelPath):
    """
    使用pickle保存训练模型
    :param data:
    :param modelPath:
    :return:
    """
    model =linear_model.LinearRegression()
    model.fit(data[['x']],data[['y']])
    pickle.dump(model,open(modelPath,"wb"))
    return model


def loadModel(modelPath):
    """
    使用pikcle 读取已用模型
    :param modelPath:
    :return:
    """
    model = pickle.load(open(modelPath,"rb"))

    return model
if __name__ == '__main__':
    path = "./x_y_data.csv"
    data = pd.read_csv(path,header=0)
    modelPath  = "linerModel"
    originalModel= trainAndSaveModel(data,modelPath)
    model = loadModel(modelPath)
    print("保存的模型对1的预测值：%s" % originalModel.predict([[1]]))
    print("读取的模型对1的预测值：%s" % model.predict([[1]]))
    savaAsPMMl(data,modelPath="linerModel.pmml")
