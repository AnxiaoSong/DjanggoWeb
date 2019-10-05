# 测试集和训练划分
from sklearn.model_selection  import train_test_split

#统计学模型
import  statsmodels.api as sm
# 读出文件
from logicML.read_file import readDataFromCSV

def logitRegression(data):
    """
    逻辑回归的使用
    :param data:
    :return:
    """
    trainData, testData = train_test_split(data,test_size=0.2)
    re = trainModel(trainData)
    modelSummary(re)

def trainModel(trainData):
    """
    训练时模型
    :param trainData:
    :return:
    """
    formula = "label_code ~ age + education_num + capital_gain + capital_loss + hours_per_week"
    model = sm.Logit.from_formula(formula,data=trainData)
    re = model.fit()
    return re


def modelSummary(re):
    """
    逻辑回归模型德 分析
    :param re:
    :return:
    """
    print("整体整体评估：")
    print(re.summary())

if __name__ == "__main__":
    path = "./adult.csv"
    data = readDataFromCSV(path)
    logitRegression(data)