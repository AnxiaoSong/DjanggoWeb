import pandas as pd


def read_csv_file(path, header=None):
    """
    读取CSV文件
    :param path:
    :param header:
    :return:
    """

    return pd.read_csv(path, delimiter=",", header=header)

