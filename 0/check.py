import pandas as pd
import numpy as np
from scipy.stats import pearsonr
def calculate_ic(y_true, y_pred):
    """
    计算IC（信息系数，Information Coefficient）

    数学公式：
    IC = corr(预测收益率, 真实收益率)
    使用皮尔森相关系数

    比赛评分标准：
    排名按IC值从高到低，IC越高越好
    """
    # 计算皮尔森相关系数
    ic_value, p_value = pearsonr(y_true, y_pred)
    return ic_value
def quick_fix():
    """快速修复方案"""
    # 1. 只使用E股票的数据
    df_e = pd.read_csv('data/1/E.csv')

    # 2. 只构建3个核心特征
    features = pd.DataFrame()
    features['price_change_1'] = df_e['LastPrice'].pct_change(1)
    features['price_change_10'] = df_e['LastPrice'].pct_change(10)
    features['bid_ask_spread'] = (df_e['AskPrice1'] - df_e['BidPrice1']) / df_e['LastPrice']
    features['target'] = df_e['Return5min']

    # 在quick_fix函数中添加：
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    # 4. 训练简单线性模型
    from sklearn.linear_model import LinearRegression
    X = features[['price_change_1', 'price_change_10', 'bid_ask_spread']]
    y = features['target']

    model = LinearRegression()
    model.fit(X[:-1000], y[:-1000])  # 留1000个样本测试
    y_pred = model.predict(X[-1000:])

    ic = calculate_ic(y[-1000:], y_pred)
    print(f"简单线性模型IC: {ic:.4f}")

    return ic
quick_fix()