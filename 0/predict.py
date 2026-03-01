
# ============ 第一部分：导入库 ============
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')  # 忽略警告信息，让输出更干净

# 机器学习相关库
from sklearn.model_selection import TimeSeriesSplit  # 时间序列交叉验证
from sklearn.preprocessing import StandardScaler  # 数据标准化
from sklearn.ensemble import RandomForestRegressor  # 特征选择用的随机森林
import lightgbm as lgb  # 主要预测模型

# 评估指标相关
from scipy.stats import pearsonr, spearmanr  # 计算IC（相关系数）

import matplotlib.pyplot as plt

# ============ 第八部分：预测新数据函数 ============
def predict_new_data(self, E_row, sector_rows):
    [A_row,B_row,C_row,D_row] = sector_rows
    #print("\n" + "=" * 60)
    #print("开始预测新数据")
    #print("=" * 60)

    # 合并五只股票数据（使用训练时的相同函数）
    #print("合并五只股票测试数据，写成一行...")
    test_merged_data = merge_all_stocks_one_line(A_row,B_row,C_row,D_row,E_row)#只有一行五个股票的数据
    self.whole_dataframe=pd.concat((self.whole_dataframe,test_merged_data)).reset_index(drop=True)

    self.whole_dataframe.to_csv("whole_data.csv", index=False)

    #只保留最近15min的数据（即1800行）
    if len(self.whole_dataframe) > 1800:
        self.whole_dataframe=self.whole_dataframe.iloc[1:]

    # 为新数据创建特征（使用与训练相同的特征工程）
    #print("为新数据创建特征...")
    test_features = create_all_features_enhanced(self.whole_dataframe)

    # 确保特征一致
    missing_features = set(self.selected_features) - set(test_features.columns)
    extra_features = set(test_features.columns) - set(self.selected_features)

    #print(f"测试数据特征数: {len(test_features.columns)}")
    #print(f"模型需要特征数: {len(self.selected_features)}")

    if missing_features:
        #print(f"警告：缺失{len(missing_features)}个特征")
        # 缺失的特征用0填充
        for feature in missing_features:
            test_features[feature] = 0

    if extra_features:
        #print(f"移除{len(extra_features)}个多余特征")
        test_features = test_features.drop(columns=list(extra_features))

    # 按正确的顺序选择特征
    x_test = test_features[self.selected_features]

    # 标准化
    x_test_scaled = self.x_scaler.transform(x_test)

    # 预测
    predictions = self.model.predict(x_test_scaled)

    # 创建结果DataFrame
    # result_df = pd.DataFrame({
    #     'Time': test_merged_data['Time'],
    #     'Predicted_Return5min': predictions
    # })

    # 保存结果

    #print("预测完成")

    return predictions[0]

def load_stock_data_one_line(row,stock_name):
    """
    .csv->dataframe
    改列名（+ABCDE,#target）
    """
    #print(f"正在加载{stock_name}股数据...")

    # 读取CSV文件
    df = row.to_frame().T
    #print(df)

    # 检查列名
    #print(f"  原始列名: {df.columns.tolist()[:5]}...")

    df.drop(columns=['Return5min'], inplace=True, errors='ignore')
    # 为所有列添加前缀（除了Time）
    rename_dict = {}
    for col in df.columns:
        if col == 'Time':
            continue  # 时间列保持不变
        elif col == 'Return5min':
            if stock_name == 'E':
                rename_dict[col] = 'target'  # E股票的目标变量
            else:
                rename_dict[col] = f'{stock_name}_Return5min'  # 其他股票的5分钟收益率
        else:
            rename_dict[col] = f'{stock_name}_{col}'

    df = df.rename(columns=rename_dict)

    # 显示处理后的列名
    #print(f"  处理后的列名示例: {[col for col in df.columns if 'Return5min' in col or col == 'target']}")

    return df


def merge_all_stocks_one_line(A_row,B_row,C_row,D_row,E_row):
    """
    合并五只股票的数据 - 修正版
    """
    #print("开始合并五只股票数据...")

    all_dfs = []


    df = load_stock_data_one_line(A_row,'A')
    all_dfs.append(df)
    df = load_stock_data_one_line(B_row, 'B')
    all_dfs.append(df)
    df = load_stock_data_one_line(C_row,'C')
    all_dfs.append(df)
    df = load_stock_data_one_line(D_row,'D')
    all_dfs.append(df)
    df = load_stock_data_one_line(E_row,'E')
    all_dfs.append(df)
    #print(all_dfs)
    # 使用reduce逐步合并 concat?
    from functools import reduce

    def merge_func(df1, df2):
        return pd.merge(df1, df2, on='Time', how='inner')

    merged_df = reduce(merge_func, all_dfs)

    #print(f"合并完成，总数据形状：{merged_df.shape}")
    #print(f"列名数量：{len(merged_df.columns)}")

    # 显示包含Return5min的列，确认没有重复
    return_cols = [col for col in merged_df.columns if 'Return5min' in col or 'target' in col]
    #print(f"收益率相关列：{return_cols}")

    return merged_df


# ============ 第三部分：特征工程函数 ============

def enhanced_stock_features(df, stock_prefix):
    """增强版股票特征"""
    features = {}

    # 基础价格列
    last_price_col = f'{stock_prefix}LastPrice'

    if last_price_col in df.columns:
        # === 价格趋势特征 ===
        # 1. 价格动量（短期、中期）
        for period in [5, 10, 20, 30, 60, 120]:  # 增加更多时间尺度
            features[f'{stock_prefix}price_momentum_{period}'] = (df[last_price_col].iloc[-1] - df[last_price_col].iloc[-1-period]) / df[last_price_col].iloc[-1-period]
            #那这里最前面的几个值就NAN了是嘛（

        # 2. 价格波动范围
        for window in [60, 120, 300, 600]:  # 30秒到5分钟
            roll_max = df[last_price_col].tail(window).max()
            roll_min = df[last_price_col].tail(window).min()
            features[f'{stock_prefix}price_range_{window}'] = (roll_max - roll_min) / (roll_min + 1e-8)#何意味避免0嘛

        # 3. 移动平均线特征
        ma_windows = {
            '30s': 60,  # 30秒 (比5分钟预测窗口短)
            '1min': 120,  # 1分钟 (预测窗口的1/5)
            '2min': 240,  # 2分钟 (预测窗口的2/5)
            '3min': 360,  # 3分钟 (预测窗口的3/5)
            '5min': 600,  # 5分钟 (与预测窗口相同)
            '10min': 1200,  # 10分钟 (预测窗口的2倍)
            '15min': 1800,  # 15分钟 (预测窗口的3倍)
        }

        # 1. 计算所有移动平均线
        ma_dict = {}
        for name, window in ma_windows.items():
            ma_key = f'{stock_prefix}ma_{name}'
            # 使用适当的min_periods避免开头太多NaN(window中大于等于min_periods个数不为NAN，rolling就不为NAN)
            min_periods = max(1, int(window * 0.1))
            ma_dict[name] = df[last_price_col].tail(window, min_periods=min_periods).mean()
            features[ma_key] = ma_dict[name]

        # 2. 价格相对于移动平均线的位置（核心特征）
        # 偏离度 = (价格 - MA) / MA
        for name in ['30s', '1min', '5min', '10min']:
            if name in ma_dict:#意义何在
                ma_value = ma_dict[name]
                price = df[last_price_col].iloc[-1]
                features[f'{stock_prefix}price_vs_ma_{name}_pct'] = (price - ma_value) / (ma_value + 1e-8)

                # 价格是否在MA之上（布尔特征）
                features[f'{stock_prefix}above_ma_{name}'] = (price > ma_value).astype(int)

        # 3. 移动平均线的趋势特征（MA的斜率）
        # 计算各MA在一段时间内的变化率
        for name in ['1min', '5min', '10min']:
            ma_value = ma_dict[name]
            # 短期变化：最近1分钟的变化
            ma_value_before_1min = df[f'{stock_prefix}ma_{name}'].iloc[-120]
            change_1min = ma_value - ma_value_before_1min
            features[f'{stock_prefix}ma_{name}_change_1min'] = change_1min / (ma_value_before_1min + 1e-8)

            # 只保留需要的历史数据（span=300最多需要300个数据点）
            max_span = 300
            ma_value_truncated = df[f'{stock_prefix}ma_{name}'].tail(max_span-1)  # 不足300行则返回全部
            ma_value_truncated.append(ma_value)
            # 计算EMA后只取最后一个值（比全量计算快很多）
            ema_short = ma_value_truncated.ewm(span=60, adjust=False).mean().iloc[-1]
            ema_long = ma_value_truncated.ewm(span=300, adjust=False).mean().iloc[-1]

            # 趋势强度计算
            trend_strength = (ema_short - ema_long) / (ema_long + 1e-8)
            features[f'{stock_prefix}ma_{name}_trend_strength'] = trend_strength

        # 4. 金叉死叉特征（技术分析核心）
        if all(name in ma_dict for name in ['30s', '1min', '5min']):
            ma_30s = ma_dict['30s']
            ma_1min = ma_dict['1min']
            ma_5min = ma_dict['5min']
            ma_30s_last=df[f'{stock_prefix}ma_{"30s"}'].iloc[-1]#overall_dataframe里面的上一个均线
            ma_5min_last=df[f'{stock_prefix}ma_{"5min"}'].iloc[-1]
            ma_1min_last=df[f'{stock_prefix}ma_{"1min"}'].iloc[-1]



            # 4.1 基本金叉死叉信号
            # 30秒线上穿1分钟线
            cross_up_30s_1min = (ma_30s > ma_1min) & (ma_30s_last <= ma_1min_last)
            cross_down_30s_1min = (ma_30s < ma_1min) & (ma_30s_last >= ma_1min_last)

            # 1分钟线上穿5分钟线（传统金叉）
            cross_up_1min_5min = (ma_1min > ma_5min) & (ma_1min_last <= ma_5min_last)
            cross_down_1min_5min = (ma_1min < ma_5min) & (ma_1min_last >= ma_5min_last)

            features[f'{stock_prefix}cross_up_30s_1min'] = cross_up_30s_1min.astype(int)
            features[f'{stock_prefix}cross_down_30s_1min'] = cross_down_30s_1min.astype(int)
            features[f'{stock_prefix}cross_up_1min_5min'] = cross_up_1min_5min.astype(int)
            features[f'{stock_prefix}cross_down_1min_5min'] = cross_down_1min_5min.astype(int)

            # 4.2 金叉/死叉后的时间（事件驱动特征）
            # 距离上次金叉的时间（tick数）
            for cross_name, cross_signal, name_in_dataframe in [('up_1min_5min', cross_up_1min_5min, f'{stock_prefix}cross_up_1min_5min'),
                                             ('down_1min_5min', cross_down_1min_5min, f'{stock_prefix}cross_down_1min_5min')]:
                # 标记金叉发生的位置
                cross_idx = df[name_in_dataframe][df[name_in_dataframe] == 1].last_valid_index()
                if cross_idx is not None:
                    # 计算距离上次金叉的时间
                    time_since_cross = len(df) - cross_idx
                    features[f'{stock_prefix}ticks_since_{cross_name}'] = time_since_cross
                else:
                    features[f'{stock_prefix}ticks_since_{cross_name}'] = 9999  # 很大值表示很久没有

            # 4.3 均线排列状态（多头/空头排列）
            # 多头排列：短期 > 中期 > 长期
            bull_alignment = (ma_30s > ma_1min) & (ma_1min > ma_5min)
            # 空头排列：短期 < 中期 < 长期
            bear_alignment = (ma_30s < ma_1min) & (ma_1min < ma_5min)

            features[f'{stock_prefix}bull_alignment'] = bull_alignment.astype(int)
            features[f'{stock_prefix}bear_alignment'] = bear_alignment.astype(int)

            # 排列强度：使用Z-score标准化
            ma_diff_30s_1min = (ma_30s - ma_1min) / (ma_1min + 1e-8)
            ma_diff_1min_5min = (ma_1min - ma_5min) / (ma_5min + 1e-8)
            alignment_strength = ma_diff_30s_1min + ma_diff_1min_5min
            features[f'{stock_prefix}alignment_strength'] = alignment_strength

        # 5. 移动平均线带宽特征（MA间的距离）
        if all(name in ma_dict for name in ['30s', '5min']):
            ma_30s = ma_dict['30s']
            ma_5min = ma_dict['5min']

            # 带宽 = (短期MA - 长期MA) / 长期MA
            ma_bandwidth = (ma_30s - ma_5min) / (ma_5min + 1e-8)
            features[f'{stock_prefix}ma_bandwidth'] = ma_bandwidth

            # 带宽的变化率
            bandwidth_change = ma_bandwidth - df[f'{stock_prefix}ma_bandwidth'].iloc[-120]  # 1分钟变化
            features[f'{stock_prefix}ma_bandwidth_change'] = bandwidth_change

        # 6. 价格与MA的背离特征
        if all(name in ma_dict for name in ['1min', '5min']):
            price = df[last_price_col]
            ma_1min = ma_dict['1min']
            ma_5min = ma_dict['5min']

            # 计算价格和MA的动量
            price_momentum = price.iloc(-1) - price.iloc(-61)  # 30秒动量
            ma_1min_momentum = ma_1min - df[f'{stock_prefix}ma_1min'].iloc(-60)
            ma_5min_momentum = ma_5min - df[f'{stock_prefix}ma_5min'].iloc(-60)

            # 背离：价格上涨但MA下跌，或反之
            divergence_1min = ((price_momentum > 0) & (ma_1min_momentum < 0)) | \
                              ((price_momentum < 0) & (ma_1min_momentum > 0))
            divergence_5min = ((price_momentum > 0) & (ma_5min_momentum < 0)) | \
                              ((price_momentum < 0) & (ma_5min_momentum > 0))

            features[f'{stock_prefix}divergence_1min'] = divergence_1min.astype(int)
            features[f'{stock_prefix}divergence_5min'] = divergence_5min.astype(int)

        # 4. 收益率波动率（使用1期收益率）
        ret_1 = df[last_price_col].pct_change()
        features[f'{stock_prefix}ret_volatility_10'] = ret_1.rolling(10).std()
        features[f'{stock_prefix}ret_volatility_20'] = ret_1.rolling(20).std()
        features[f'{stock_prefix}ret_volatility_30'] = ret_1.rolling(30).std()

    # === 成交量特征 ===
    # 成交量加权价格
    if (f'{stock_prefix}LastPrice' in df.columns and
            f'{stock_prefix}Volume' in df.columns):

        price = df[f'{stock_prefix}LastPrice']
        volume = df[f'{stock_prefix}Volume']

        for window in [120, 600, 1200]:  # 1分钟、5分钟、10分钟VWAP
            vwap = (price * volume).rolling(window).sum() / volume.rolling(window).sum()
            features[f'{stock_prefix}vwap_{window}'] = vwap
            features[f'{stock_prefix}price_vwap_diff_{window}'] = price - vwap
            features[f'{stock_prefix}price_vwap_ratio_{window}'] = price / (vwap + 1e-8) - 1

    # === 委托深度特征 ===
    bid_volume_cols = [f'{stock_prefix}BidVolume{i}' for i in range(1, 6)]
    ask_volume_cols = [f'{stock_prefix}AskVolume{i}' for i in range(1, 6)]

    if all(col in df.columns for col in bid_volume_cols + ask_volume_cols):
        # 5档深度不平衡
        total_bid_depth = df[bid_volume_cols].sum(axis=1)
        total_ask_depth = df[ask_volume_cols].sum(axis=1)
        features[f'{stock_prefix}depth_imbalance'] = (
                                                             total_bid_depth - total_ask_depth
                                                     ) / (total_bid_depth + total_ask_depth + 1e-6)

        # 深度变化率
        features[f'{stock_prefix}depth_change'] = total_bid_depth.pct_change(120) - total_ask_depth.pct_change(120)

    # === 订单流不平衡（高级版）===
    if (f'{stock_prefix}OrderBuyVolume' in df.columns and
            f'{stock_prefix}OrderSellVolume' in df.columns):
        # 订单流不平衡
        order_imbalance = (
                                  df[f'{stock_prefix}OrderBuyVolume'] -
                                  df[f'{stock_prefix}OrderSellVolume']
                          ) / (df[f'{stock_prefix}OrderBuyVolume'] + df[f'{stock_prefix}OrderSellVolume'] + 1e-6)

        features[f'{stock_prefix}order_imbalance'] = order_imbalance
        features[f'{stock_prefix}order_imbalance_ma'] = order_imbalance.rolling(120).mean()

    return features





# 添加MA特征后处理的辅助函数
def post_process_ma_features(features_df, stock_prefix='E'):
    """对移动平均线特征进行后处理"""

    # 1. 处理MA特征的缺失值
    ma_columns = [col for col in features_df.columns if 'ma_' in col]

    for col in ma_columns:
        # 前向填充
        features_df[col] = features_df[col].ffill()

        # 对于开头依然为NaN的，用第一个有效值填充
        if features_df[col].isna().any():
            first_valid = features_df[col].first_valid_index()
            if first_valid is not None:
                features_df[col] = features_df[col].fillna(features_df.loc[first_valid, col])

    # 2. 平滑MA衍生特征（减少噪声）
    smooth_columns = [col for col in features_df.columns if
                      any(x in col for x in ['_trend_', '_strength', '_alignment'])]

    for col in smooth_columns:
        # 使用EMA平滑
        features_df[f'{col}_smooth'] = features_df[col].ewm(span=30, adjust=False).mean()

    # 3. 创建MA特征组合（交互特征）
    if f'{stock_prefix}price_vs_ma_30s_pct' in features_df.columns and \
            f'{stock_prefix}price_vs_ma_5min_pct' in features_df.columns:
        # 短期偏离与长期偏离的差异
        features_df[f'{stock_prefix}ma_deviation_diff'] = (
                features_df[f'{stock_prefix}price_vs_ma_30s_pct'] -
                features_df[f'{stock_prefix}price_vs_ma_5min_pct']
        )

        # 偏离的一致性（符号是否相同）
        features_df[f'{stock_prefix}ma_deviation_consistent'] = (
                np.sign(features_df[f'{stock_prefix}price_vs_ma_30s_pct']) ==
                np.sign(features_df[f'{stock_prefix}price_vs_ma_5min_pct'])
        ).astype(int)

    return features_df


def enhanced_sector_features(stock_features_dict):
    """增强版板块特征 - 更强版本"""
    sector_features = {}

    # === 1. 直接使用其他股票的收益率作为特征 ===
    for stock in ['A', 'B', 'C', 'D']:
        if f'{stock}_Return5min' in stock_features_dict:
            # 直接使用收益率
            sector_features[f'{stock}_return'] = stock_features_dict[f'{stock}_Return5min']

            # 标准化后的收益率
            return_data = stock_features_dict[f'{stock}_Return5min']
            rolling_mean = return_data.rolling(30).mean()
            rolling_std = return_data.rolling(30).std()
            sector_features[f'{stock}_return_zscore'] = (return_data - rolling_mean) / (rolling_std + 1e-6)

    # === 2. 板块加权收益率 ===
    returns_list = []
    weights = []  # 可以根据相关性赋予权重

    for stock in ['A', 'B', 'C', 'D']:
        if f'{stock}_Return5min' in stock_features_dict:
            returns_list.append(stock_features_dict[f'{stock}_Return5min'])
            # 简单等权重
            weights.append(1.0)

    if returns_list:
        returns_df = pd.concat(returns_list, axis=1)
        # 等权重板块收益率
        sector_features['sector_weighted_return'] = returns_df.mean(axis=1)

        # 板块收益率离散度
        sector_features['sector_return_dispersion'] = returns_df.std(axis=1)

    # === 3. 板块动量领先指标（更直接的领先效应）===
    for lead_stock in ['A', 'B', 'C', 'D']:
        lead_return_col = f'{lead_stock}_Return5min'
        if lead_return_col in stock_features_dict:
            lead_return = stock_features_dict[lead_return_col]

            # 直接使用领先股票的收益率
            for lag in [1, 2, 5, 10]:
                sector_features[f'{lead_stock}_return_lag_{lag}'] = lead_return.shift(lag)

    # === 4. E股与其他股票收益率的差值 ===
    if 'E_Return5min' in stock_features_dict:
        e_return = stock_features_dict['E_Return5min']

        for stock in ['A', 'B', 'C', 'D']:
            if f'{stock}_Return5min' in stock_features_dict:
                stock_return = stock_features_dict[f'{stock}_Return5min']
                # 收益率差值
                sector_features[f'E_minus_{stock}_return'] = e_return - stock_return

                # 收益率比值
                sector_features[f'E_div_{stock}_return'] = e_return / (stock_return.abs() + 1e-6)

    # === 5. 板块技术指标综合 ===
    # 收集所有股票的ma_10min_pct特征
    ma_features = []
    for stock in ['A', 'B', 'C', 'D', 'E']:
        ma_feat = f'{stock}_price_vs_ma_10min_pct'
        if ma_feat in stock_features_dict:
            ma_features.append(stock_features_dict[ma_feat])

    if ma_features:
        ma_df = pd.concat(ma_features, axis=1)
        # 板块平均偏离
        sector_features['sector_ma_deviation_avg'] = ma_df.mean(axis=1)
        # 板块偏离离散度
        sector_features['sector_ma_deviation_std'] = ma_df.std(axis=1)
        # E股在板块中的排名
        e_ma = ma_df.iloc[:, -1] if 'E' in ma_df.columns else ma_features[-1]
        sector_features['E_ma_deviation_rank'] = (ma_df.T > e_ma).sum() / len(ma_features)

    return sector_features


def enhanced_time_features(time_series):
    """增强版时间特征"""
    time_features = {}

    # 将整数时间转换为datetime
    time_str = time_series.astype(str).str.zfill(9)
    hours = time_str.str[:2].astype(int)
    minutes = time_str.str[2:4].astype(int)

    # 只保留开盘/收盘标记
    time_features['is_opening_30min'] = ((hours == 9) & (minutes >= 30) & (minutes < 45))
    time_features['is_closing_30min'] = ((hours == 14) & (minutes >= 45))

    return time_features


def add_e_specific_features(stock_features_dict):
    """为E股添加特定特征（预测目标）"""
    e_features = {}

    # E股的各种收益率
    if 'E_price_momentum_5' in stock_features_dict:
        # 1. E股动量强度
        e_momentum_5 = stock_features_dict['E_price_momentum_5']
        e_std_20 = e_momentum_5.rolling(20).std()
        e_features['E_momentum_strength'] = e_momentum_5 / (e_std_20 + 1e-6)

        # 2. E股动量变化率
        e_features['E_momentum_change'] = e_momentum_5.pct_change()

        # 3. E股动量符号
        e_features['E_momentum_sign'] = np.sign(e_momentum_5)

        # 4. E股连续上涨/下跌次数
        sign_series = np.sign(e_momentum_5)
        consecutive = sign_series.groupby((sign_series != sign_series.shift()).cumsum()).cumcount() + 1
        e_features['E_consecutive_direction'] = consecutive * sign_series

    # E股与其他股票的互动
    for stock in ['A', 'B', 'C', 'D']:
        if f'{stock}_price_momentum_5' in stock_features_dict:
            stock_momentum = stock_features_dict[f'{stock}_price_momentum_5']
            e_momentum = stock_features_dict['E_price_momentum_5']

            # 5. E股相对于其他股票的动量差
            e_features[f'E_vs_{stock}_momentum_diff'] = e_momentum - stock_momentum

            # 6. E股与其他股票动量的比率
            e_features[f'E_{stock}_momentum_ratio'] = e_momentum / (stock_momentum.abs() + 1e-6)

    return e_features


def create_all_features_enhanced(df):
    """增强版特征创建features+target"""
    #print("开始创建增强版特征...")

    stock_features_dict = {}

    # 为每只股票创建增强特征
    for stock in ['A', 'B', 'C', 'D', 'E']:
        #print(f"  创建{stock}股增强特征...")
        # 增强特征
        features = enhanced_stock_features(df, f'{stock}_')
        stock_features_dict.update(features)

        # 对MA特征进行后处理（主要对E股票）
        if stock == 'E':
            ma_features = {k: v for k, v in features.items() if 'ma_' in k}
            if ma_features:
                ma_df = pd.DataFrame(ma_features)
                processed_ma = post_process_ma_features(ma_df, f'{stock}_')
                stock_features_dict.update(processed_ma)

    # 添加E股特定特征
    #print("  添加E股特定特征...")
    e_specific_features = add_e_specific_features(stock_features_dict)
    stock_features_dict.update(e_specific_features)

    # 创建板块特征
    #print("  创建增强版板块特征...")
    sector_features = enhanced_sector_features(stock_features_dict)

    # 创建极简时间特征
    #print("  创建极简时间特征...")
    time_features = enhanced_time_features(df['Time'])

    # 合并所有特征
    all_features = pd.DataFrame(stock_features_dict)
    all_features = all_features.join(pd.DataFrame(sector_features))
    all_features = all_features.join(pd.DataFrame(time_features))

    # 添加目标变量
    target_col_name = 'target' if 'target' in df.columns else 'E_Return5min'
    if target_col_name in df.columns:
        all_features['target'] = df[target_col_name]
    else:
        #print("错误: 没有找到目标列!")
        all_features['target'] = 0

    # 缺失值处理
    all_features = all_features.ffill().bfill().fillna(0)

    #print(f"增强特征创建完成，最终{len(all_features)}行，{len(all_features.columns)}个特征")

    #print(all_features.iloc[[-1]])
    return all_features.iloc[[-1]]


def feature_post_processing(features_df):
    """特征后处理"""
    print("\n特征后处理...")

    # 1. 移除相关性极高的特征
    corr_matrix = features_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 找到相关性大于0.95的特征对
    to_drop = []
    for column in upper_tri.columns:
        if column in features_df.columns:
            corr_series = upper_tri[column]
            high_corr = corr_series[corr_series > 0.98].index.tolist()
            to_drop.extend(high_corr)

    to_drop = list(set(to_drop))
    if to_drop:
        #print(f"  移除{len(to_drop)}个高相关性特征")
        features_df = features_df.drop(columns=to_drop)

    # 2. 移除方差过小的特征
    variances = features_df.var()
    low_var_features = variances[variances < 1e-8].index.tolist()
    if low_var_features:
        #print(f"  移除{len(low_var_features)}个低方差特征")
        features_df = features_df.drop(columns=low_var_features)

    # 4. 板块特征专门处理（新增部分）
    #print("  4. 处理板块特征...")

    # 4.1 识别板块特征
    sector_cols = [col for col in features_df.columns if any(x in col for x in [
        '_lead_', '_corr_', 'sector_', 'E_sector', 'A_', 'B_', 'C_', 'D_'
    ]) and col not in ['target']]

    #print(f"    找到{len(sector_cols)}个板块相关特征")

    if sector_cols:

        # 4.3 创建板块特征组合
        return_cols = [col for col in sector_cols if '_return' in col]
        if len(return_cols) >= 2:
            features_df['sector_return_composite'] = features_df[return_cols].mean(axis=1)
            #print(f"    创建了板块收益率综合特征，基于{len(return_cols)}个特征")

    # 6. 最终检查和处理
    #print("  6. 最终检查...")

    # 移除与target相关性为NaN的特征
    if 'target' in features_df.columns:
        target_corr = features_df.corrwith(features_df['target']).abs()
        nan_corr_features = target_corr[target_corr.isna()].index.tolist()
        nan_corr_features = [f for f in nan_corr_features if f != 'target']

        if nan_corr_features:
            #print(f"    移除{len(nan_corr_features)}个与目标相关性为NaN的特征")
            features_df = features_df.drop(columns=nan_corr_features)

    # 确保没有inf或NaN值
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(0)

    #print(f"  后处理完成，最终特征数: {len(features_df.columns)}")
    return features_df

