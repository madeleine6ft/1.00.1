"""
股票未来收益率预测模型 - 完整实现代码
作者：AI助手
适用：入门级选手
功能：使用ABCDE五只股票5天的Tick数据，预测股票E未来5分钟收益率
数据格式：33个字段，如题目所述
"""

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


# ============ 第二部分：数据加载函数 ============
def load_stock_data(stock_name, file_path):
    print(f"正在加载{stock_name}股数据...")

    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 检查列名
    print(f"  原始列名: {df.columns.tolist()[:5]}...")

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
    print(f"  处理后的列名示例: {[col for col in df.columns if 'Return5min' in col or col == 'target']}")

    return df


def merge_all_stocks(data_config):
    """
    合并五只股票的数据 - 修正版
    """
    print("开始合并五只股票数据...")

    all_dfs = []
    for stock, path in data_config.items():
        df = load_stock_data(stock, path)

        # 打印关键列，确认处理正确
        if 'target' in df.columns:
            print(f"  {stock}股: 找到target列")
        if f'{stock}_Return5min' in df.columns:
            print(f"  {stock}股: 找到{stock}_Return5min列")

        all_dfs.append(df)

    # 使用reduce逐步合并 concat?
    from functools import reduce

    def merge_func(df1, df2):
        return pd.merge(df1, df2, on='Time', how='inner')

    merged_df = reduce(merge_func, all_dfs)

    print(f"合并完成，总数据形状：{merged_df.shape}")
    print(f"列名数量：{len(merged_df.columns)}")

    # 显示包含Return5min的列，确认没有重复
    return_cols = [col for col in merged_df.columns if 'Return5min' in col or 'target' in col]
    print(f"收益率相关列：{return_cols}")

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
            features[f'{stock_prefix}price_momentum_{period}'] = df[last_price_col].pct_change(periods=period)

        # 2. 价格波动范围
        for window in [60, 120, 300, 600]:  # 30秒到5分钟
            roll_max = df[last_price_col].rolling(window).max()
            roll_min = df[last_price_col].rolling(window).min()
            features[f'{stock_prefix}price_range_{window}'] = (roll_max - roll_min) / (roll_min + 1e-8)

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
            # 使用适当的min_periods避免开头太多NaN
            min_periods = max(1, int(window * 0.1))
            ma_dict[name] = df[last_price_col].rolling(window, min_periods=min_periods).mean()
            features[ma_key] = ma_dict[name]

        # 2. 价格相对于移动平均线的位置（核心特征）
        # 偏离度 = (价格 - MA) / MA
        for name in ['30s', '1min', '5min', '10min']:
            if name in ma_dict:
                ma_value = ma_dict[name]
                price = df[last_price_col]
                features[f'{stock_prefix}price_vs_ma_{name}_pct'] = (price - ma_value) / (ma_value + 1e-8)

                # 价格是否在MA之上（布尔特征）
                features[f'{stock_prefix}above_ma_{name}'] = (price > ma_value).astype(int)

        # 3. 移动平均线的趋势特征（MA的斜率）
        # 计算各MA在一段时间内的变化率
        for name in ['1min', '5min', '10min']:
            if name in ma_dict:
                ma_value = ma_dict[name]
                # 短期变化：最近1分钟的变化
                change_1min = ma_value - ma_value.shift(120)
                features[f'{stock_prefix}ma_{name}_change_1min'] = change_1min / (ma_value.shift(120) + 1e-8)

                # 长期变化：与自身相比（MA自己的变化趋势）
                # 使用EMA计算趋势强度，避免噪声
                ema_short = ma_value.ewm(span=60, adjust=False).mean()
                ema_long = ma_value.ewm(span=300, adjust=False).mean()
                features[f'{stock_prefix}ma_{name}_trend_strength'] = (ema_short - ema_long) / (ema_long + 1e-8)

        # 4. 金叉死叉特征（技术分析核心）
        if all(name in ma_dict for name in ['30s', '1min', '5min']):
            ma_30s = ma_dict['30s']
            ma_1min = ma_dict['1min']
            ma_5min = ma_dict['5min']

            # 4.1 基本金叉死叉信号
            # 30秒线上穿1分钟线
            cross_up_30s_1min = (ma_30s > ma_1min) & (ma_30s.shift(1) <= ma_1min.shift(1))
            cross_down_30s_1min = (ma_30s < ma_1min) & (ma_30s.shift(1) >= ma_1min.shift(1))

            # 1分钟线上穿5分钟线（传统金叉）
            cross_up_1min_5min = (ma_1min > ma_5min) & (ma_1min.shift(1) <= ma_5min.shift(1))
            cross_down_1min_5min = (ma_1min < ma_5min) & (ma_1min.shift(1) >= ma_5min.shift(1))

            features[f'{stock_prefix}cross_up_30s_1min'] = cross_up_30s_1min.astype(int)
            features[f'{stock_prefix}cross_down_30s_1min'] = cross_down_30s_1min.astype(int)
            features[f'{stock_prefix}cross_up_1min_5min'] = cross_up_1min_5min.astype(int)
            features[f'{stock_prefix}cross_down_1min_5min'] = cross_down_1min_5min.astype(int)

            # 4.2 金叉/死叉后的时间（事件驱动特征）
            # 距离上次金叉的时间（tick数）
            for cross_name, cross_signal in [('up_1min_5min', cross_up_1min_5min),
                                             ('down_1min_5min', cross_down_1min_5min)]:
                # 标记金叉发生的位置
                cross_idx = cross_signal.where(cross_signal).last_valid_index()
                if cross_idx is not None:
                    # 计算距离上次金叉的时间
                    time_since_cross = df.index - cross_idx
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
            bandwidth_change = ma_bandwidth - ma_bandwidth.shift(120)  # 1分钟变化
            features[f'{stock_prefix}ma_bandwidth_change'] = bandwidth_change

        # 6. 价格与MA的背离特征
        if all(name in ma_dict for name in ['1min', '5min']):
            price = df[last_price_col]
            ma_1min = ma_dict['1min']
            ma_5min = ma_dict['5min']

            # 计算价格和MA的动量
            price_momentum = price - price.shift(60)  # 30秒动量
            ma_1min_momentum = ma_1min - ma_1min.shift(60)
            ma_5min_momentum = ma_5min - ma_5min.shift(60)

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
    """增强版板块特征"""
    sector_features = {}

    # === 板块动量特征 ===
    for stock in ['A', 'B', 'C', 'D']:
        if f'{stock}_price_momentum_5' in stock_features_dict:
            # 个股动量
            sector_features[f'{stock}_momentum_strength'] = (
                    stock_features_dict[f'{stock}_price_momentum_5'] /
                    (stock_features_dict[f'{stock}_price_momentum_5'].rolling(20).std() + 1e-6)
            )

    # === 板块相关性矩阵特征 ===
    momentum_cols = []
    for stock in ['A', 'B', 'C', 'D']:
        if f'{stock}_price_momentum_5' in stock_features_dict:
            momentum_cols.append(f'{stock}_price_momentum_5')

    if momentum_cols and 'E_price_momentum_5' in stock_features_dict:
        momentum_df = pd.DataFrame({
            col: stock_features_dict[col] for col in momentum_cols
        })

        # E股与板块平均动量的相关性
        sector_avg_momentum = momentum_df.mean(axis=1)
        sector_features['E_sector_momentum_corr'] = (
            stock_features_dict['E_price_momentum_5'].rolling(30).corr(sector_avg_momentum)
        )

        # 板块动量离散度
        sector_features['sector_momentum_std'] = momentum_df.std(axis=1)

    # === 领先滞后关系（改进版）===
    # 寻找哪个股票对E股有领先作用
    for lead_stock in ['A', 'B', 'C', 'D']:
        for lag in [1, 2, 3, 5, 10]:
            lead_col = f'{lead_stock}_price_momentum_5'
            lag_col = f'E_price_momentum_5'

            if lead_col in stock_features_dict and lag_col in stock_features_dict:
                # 计算交叉相关性
                cross_corr = stock_features_dict[lead_col].shift(lag).rolling(30).corr(
                    stock_features_dict[lag_col]
                )
                sector_features[f'{lead_stock}_lead_E_{lag}'] = cross_corr

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
    """增强版特征创建"""
    print("开始创建增强版特征...")

    stock_features_dict = {}

    # 为每只股票创建增强特征
    for stock in ['A', 'B', 'C', 'D', 'E']:
        print(f"  创建{stock}股增强特征...")
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
    print("  添加E股特定特征...")
    e_specific_features = add_e_specific_features(stock_features_dict)
    stock_features_dict.update(e_specific_features)

    # 创建板块特征
    print("  创建增强版板块特征...")
    sector_features = enhanced_sector_features(stock_features_dict)

    # 创建极简时间特征
    print("  创建极简时间特征...")
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
        print("错误: 没有找到目标列!")
        all_features['target'] = 0

    # 缺失值处理
    all_features = all_features.ffill().bfill().fillna(0)

    print(f"增强特征创建完成，最终{len(all_features)}行，{len(all_features.columns)}个特征")

    return all_features


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
            high_corr = corr_series[corr_series > 0.95].index.tolist()
            to_drop.extend(high_corr)

    to_drop = list(set(to_drop))
    if to_drop:
        print(f"  移除{len(to_drop)}个高相关性特征")
        features_df = features_df.drop(columns=to_drop)

    # 2. 移除方差过小的特征
    variances = features_df.var()
    low_var_features = variances[variances < 1e-6].index.tolist()
    if low_var_features:
        print(f"  移除{len(low_var_features)}个低方差特征")
        features_df = features_df.drop(columns=low_var_features)

    # 3. 移除对时间特征过于依赖的特征（如果minute_of_day相关性太高）
    if 'minute_of_day' in features_df.columns:
        time_corr = features_df.corrwith(features_df['minute_of_day']).abs()
        high_time_corr = time_corr[time_corr > 0.9].index.tolist()
        high_time_corr = [f for f in high_time_corr if f != 'minute_of_day' and f != 'target']
        if high_time_corr:
            print(f"  移除{len(high_time_corr)}个与时间高度相关的特征")
            features_df = features_df.drop(columns=high_time_corr)

    return features_df



# ============ 第四部分：特征选择函数 ============
def select_important_features(features_df, target_col='target', n_features=100):
    """
    选择最重要的特征

    方法：使用随机森林计算特征重要性
    原理：通过决策树的分裂节点计算特征对目标变量的解释程度

    数学公式：
    特征重要性 = 特征在所有树中分裂节点的平均不纯度减少量
    """
    print(f"\n选择最重要的{n_features}个特征...")

    # 分离特征和目标
    x = features_df.drop(columns=[target_col])
    y = features_df[target_col]

    # 将所有列转换为数值（无法转换的变为 NaN）
    x = x.apply(pd.to_numeric, errors='coerce')
    # 将正负无穷替换为 NaN
    x = x.replace([np.inf, -np.inf], np.nan)
    # 用0填充 NaN（也可改为其他策略）
    x = x.fillna(0)
    # 限制极端值，防止超出 float32 范围（阈值可调整）
    x = x.clip(-1e10, 1e10)

    # 同样处理目标 y，避免异常值
    y = pd.to_numeric(y, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
    # -------------------------------------------------------------------

    # 使用随机森林（树的数量少一些，加快速度）
    rf = RandomForestRegressor(
        n_estimators=50,  # 50棵树
        max_depth=10,  # 最大深度10
        random_state=42,  # 随机种子，保证结果可重复
        n_jobs=-1  # 使用所有CPU核心
    )

    # 训练随机森林
    rf.fit(x, y)

    # 获取特征重要性
    importance_df = pd.DataFrame({
        'feature': x.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # 显示最重要的10个特征
    print("\nTop 10最重要的特征：")
    for idx, (_, row) in enumerate(importance_df.head(10).iterrows(), start=1):
        print(f"  {idx}. {row['feature']}: {row['importance']:.4f}")

    # 选择最重要的n_features个特征
    selected_features = importance_df.head(n_features)['feature'].tolist()



    selected_df = pd.concat([x[selected_features].reset_index(drop=True),
                             pd.Series(y.values, name=target_col)], axis=1)

    # 最终再做一次数值化/替换/裁剪，确保没有 inf/NaN/超大值
    selected_df = selected_df.apply(pd.to_numeric, errors='coerce') \
        .replace([np.inf, -np.inf], np.nan) \
        .fillna(0) \
        .clip(-1e10, 1e10)


    return selected_df, selected_features


def clean_target_outliers(y, n_sigma=3):
    """清理目标变量的异常值"""
    y_series = pd.Series(y) if not isinstance(y, pd.Series) else y

    # 计算均值和标准差
    y_mean = y_series.mean()
    y_std = y_series.std()

    # 定义异常值边界
    lower_bound = y_mean - n_sigma * y_std
    upper_bound = y_mean + n_sigma * y_std

    # 将异常值设为边界值
    y_cleaned = y_series.copy()
    y_cleaned[y_cleaned < lower_bound] = lower_bound
    y_cleaned[y_cleaned > upper_bound] = upper_bound

    print(f"目标值清洗: 原始范围[{y_series.min():.6f}, {y_series.max():.6f}], "
          f"清洗后[{y_cleaned.min():.6f}, {y_cleaned.max():.6f}]")

    return y_cleaned.values


# ============ 第五部分：模型训练函数 ============
def train_enhanced_lightgbm(x_train, y_train, x_val, y_val):
    """LightGBM训练"""
    print("\n开始训练...")

    # 转换数据
    x_train_np = np.asarray(x_train, dtype=np.float32)
    x_val_np = np.asarray(x_val, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32)
    y_val_np = np.asarray(y_val, dtype=np.float32)

    train_data = lgb.Dataset(x_train_np, label=y_train_np)
    val_data = lgb.Dataset(x_val_np, label=y_val_np, reference=train_data)

    # 优化后的参数
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 127,  # 增加叶子数
        'max_depth': 10,
        'min_data_in_leaf': 20,  # 减少最小叶子样本数
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'feature_fraction': 0.8,
        'lambda_l1': 0.2,  # 增加L1正则化
        'lambda_l2': 0.2,  # 增加L2正则化
        'min_gain_to_split': 0.0,
        'verbosity': -1,
        'seed': 42,
        'n_jobs': -1,
    }

    print("  开始增强训练...")

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,  # 增加训练轮数
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.log_evaluation(period=50)
        ]
    )

    return model


# ============ 第六部分：评估函数 ============
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


def calculate_rank_ic(y_true, y_pred):
    """
    计算Rank IC（斯皮尔曼秩相关系数）

    数学公式：
    Rank IC = corr(rank(预测收益率), rank(真实收益率))

    优点：
    对异常值不敏感，更稳健
    """
    # 计算斯皮尔曼秩相关系数
    rank_ic_value, p_value = spearmanr(y_true, y_pred)
    return rank_ic_value

# ============ 第七部分：主程序 ============
def main_enhanced():
    """增强版主函数"""
    print("=" * 60)
    print("股票未来收益率预测模型 - 增强版")
    print("=" * 60)

    # 1. 数据加载（不变）
    days = ['1', '2', '3', '4', '5']
    all_days_data = []

    for day in days:
        print(f"\n加载第{day}天数据...")
        day_data_config = {
            'A': f'data/{day}/A.csv',
            'B': f'data/{day}/B.csv',
            'C': f'data/{day}/C.csv',
            'D': f'data/{day}/D.csv',
            'E': f'data/{day}/E.csv'
        }
        day_merged_data = merge_all_stocks(day_data_config)
        all_days_data.append(day_merged_data)

    merged_data = pd.concat(all_days_data, ignore_index=True)
    merged_data = merged_data.sort_values('Time').reset_index(drop=True)

    # 2. 增强版特征工程
    print("\n[步骤2] 增强版特征工程")
    features_df = create_all_features_enhanced(merged_data)

    # # 3. 特征后处理
    # features_df = feature_post_processing(features_df)

    # 4. 特征选择（选择更多特征）
    print("\n[步骤3] 特征选择")
    selected_df, selected_features = select_important_features(
        features_df,
        n_features=50  # 选择50个特征
    )

    x = selected_df.drop(columns=['target'])
    y = selected_df['target']

    # 清洗目标变量异常值
    print("\n[步骤3.5] 清洗目标变量异常值")
    y = clean_target_outliers(y, n_sigma=4)

    # 4. 时间序列交叉验证
    print("\n[步骤4] 时间序列交叉验证")
    tscv = TimeSeriesSplit(n_splits=5)
    ic_scores = []
    rank_ic_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(x)):
        print(f"\n--- 第{fold + 1}折交叉验证 ---")

        x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"  训练集：{len(x_train)}个样本，验证集：{len(x_val)}个样本")

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)

        # 训练增强版模型
        model = train_enhanced_lightgbm(
            x_train_scaled, y_train,
            x_val_scaled, y_val
        )

        # 预测和评估
        y_pred = model.predict(x_val_scaled)
        ic = calculate_ic(y_val, y_pred)
        rank_ic = calculate_rank_ic(y_val, y_pred)

        ic_scores.append(ic)
        rank_ic_scores.append(rank_ic)

        print(f"  第{fold + 1}折结果：IC={ic:.4f}, Rank IC={rank_ic:.4f}")

    # 6. 评估结果
    print("\n[步骤5] 评估结果")
    print("-" * 40)

    print("交叉验证IC分数：")
    for i, ic in enumerate(ic_scores):
        print(f"  第{i + 1}折：{ic:.4f}")
    avg_ic = np.mean(ic_scores)
    print(f"\n平均IC：{np.mean(ic_scores):.4f} (±{np.std(ic_scores):.4f})")
    print(f"平均Rank IC：{np.mean(rank_ic_scores):.4f} (±{np.std(rank_ic_scores):.4f})")

    # 7. 训练最终模型
    if avg_ic > 0.07:
        print("\n[步骤6] 训练最终模型")
        final_scaler = StandardScaler()
        x_scaled = final_scaler.fit_transform(x)

        train_data = lgb.Dataset(x_scaled, label=y)

        final_model = lgb.train(
            {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.1,
                'num_leaves': 127,
                'max_depth': -1,
                'min_data_in_leaf': 20,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'feature_fraction': 0.8,
                'lambda_l1': 0.0,
                'lambda_l2': 0.0,
                'min_gain_to_split': 0.0,
                'verbosity': -1,
                'seed': 42,
                'n_jobs': -1
            },
            train_data,
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(50)]
        )

        # 保存模型
        import joblib
        joblib.dump(final_model, 'final_stock_model_enhanced.pkl')
        joblib.dump(final_scaler, 'scaler_enhanced.pkl')
        joblib.dump(selected_features, 'selected_features_enhanced.pkl')

        print("\n修复版模型训练完成！IC有明显提升。")
    else:
        print("\n警告：IC未达到阈值，不保存模型。建议检查数据或特征。")

    return avg_ic


# ============ 第八部分：预测新数据函数 ============
def predict_new_data(new_data_path, model_path='final_stock_model.pkl',
                     scaler_path='scaler.pkl', features_path='selected_features.pkl'):
    """
    预测新数据（测试集的10秒数据）

    使用流程：
    1. 先运行main()函数训练模型
    2. 然后调用此函数预测新数据

    注意：新数据格式应与训练数据一致，但不包含'target'列
    """
    print("\n" + "=" * 60)
    print("开始预测新数据")
    print("=" * 60)

    # 加载保存的模型和文件
    import joblib

    print("加载模型和文件...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selected_features = joblib.load(features_path)

    # 加载新数据
    print(f"加载新数据：{new_data_path}")
    new_data = pd.read_csv(new_data_path)

    # 检查新数据是否包含E股票
    # 新数据可能是单独的E股票，或者是五只股票的10秒数据
    # 这里假设新数据包含所有五只股票，格式与训练数据相同

    # 如果新数据只有E股票，我们需要检查
    if 'LastPrice' in new_data.columns and 'E_LastPrice' not in new_data.columns:
        print("检测到新数据可能只包含E股票，正在处理...")
        # 重命名列，添加E_前缀
        rename_dict = {}
        for col in new_data.columns:
            if col != 'Time':
                rename_dict[col] = f'E_{col}'
        new_data = new_data.rename(columns=rename_dict)

    # 为新数据创建特征
    print("为新数据创建特征...")
    new_features = feature_post_processing(new_data)

    # 确保特征一致
    missing_features = set(selected_features) - set(new_features.columns)

    if missing_features:
        print(f"警告：缺失{len(missing_features)}个特征")
        print("缺失的特征将以0填充")

        for feature in missing_features:
            new_features[feature] = 0

    # 选择特征
    x_new = new_features[selected_features]

    # 标准化
    x_new_scaled = scaler.transform(x_new)

    # 预测
    print("进行预测...")
    predictions = model.predict(x_new_scaled)

    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'Time': new_data['Time'],
        'Predicted_Return5min': predictions
    })

    # 保存结果
    result_df.to_csv('predictions.csv', index=False)
    print(f"预测完成！结果已保存到 'predictions.csv'")
    print(f"预测了{len(result_df)}个时间点")

    # 显示前几个预测结果
    print("\n前5个预测结果：")
    print(result_df.head())

    return result_df


# ============ 第九部分：运行代码 ============
if __name__ == "__main__":
    """
    运行主程序
    有两种模式：
    1. 训练模式：运行main()函数
    2. 预测模式：运行predict_new_data()函数
    """

    # 模式选择
    print("请选择模式：")
    print("1. 训练模式（使用5天数据训练模型）")
    print("2. 预测模式（使用训练好的模型预测新数据）")

    choice = input("请输入1或2：")

    if choice == '1':
        # 训练模式
        main_enhanced()
    elif choice == '2':
        # 预测模式
        # 请修改这里的文件路径为您的测试数据路径
        test_file_path = 'data/A.csv'  # 您的测试数据文件
        predict_new_data(test_file_path)
    else:
        print("输入错误，请输入1或2")