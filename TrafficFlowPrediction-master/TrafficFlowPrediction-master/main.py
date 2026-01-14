"""
Traffic Flow Prediction with Neural Networks - LSTM with Lag Correction
"""
import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from keras.models import load_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def shift_predictions(y_pred):
    """
    将滞后一个单位的预测值向后平移一个单位，对齐到正确的时间

    # Arguments
        y_pred: 原始预测值数组（滞后一个单位）

    # Returns
        平移后的预测值数组（与真实值时间对齐）
    """
    # 创建一个和原始数组相同大小的新数组
    shifted = np.zeros_like(y_pred)

    # 向后平移一个位置：y_pred[t]（预测的是t时刻的值，但实际上是t-1时刻的预测）-> shifted[t]（调整到t时刻）
    if len(y_pred) > 1:
        # 从第一个位置开始，使用后一个预测值
        shifted[:-1] = y_pred[1:]
        # 最后一个位置用趋势外推法填充
        if len(y_pred) > 2:
            # 使用最后两个点的趋势
            trend = y_pred[-1] - y_pred[-2]
            shifted[-1] = y_pred[-1] + trend
        else:
            shifted[-1] = y_pred[-1]
    else:
        # 如果只有一个值，直接返回
        shifted = y_pred.copy()

    return shifted


def plot_results(y_true, y_pred_original, y_pred_shifted):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, true data.
        y_pred_original: List/ndarray, original predicted data.
        y_pred_shifted: List/ndarray, shifted predicted data.
    """
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=len(y_true), freq='5min')

    fig = plt.figure(figsize=(12, 8))

    # 子图1: 原始预测和真实值
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x, y_true, label='True Data', linewidth=2, color='blue')
    ax1.plot(x, y_pred_original, label='LSTM (Original)', linestyle='--', color='orange')

    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Time of Day')
    ax1.set_ylabel('Flow')
    ax1.set_title('Original LSTM Predictions (Lagged by one time unit)')

    # 子图2: 平移后的预测和真实值
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x, y_true, label='True Data', linewidth=2, color='blue')
    ax2.plot(x, y_pred_shifted, label='LSTM (Lag Corrected)', linestyle='-', color='green')

    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Time of Day')
    ax2.set_ylabel('Flow')
    ax2.set_title('LSTM Predictions After Lag Correction')

    # 设置时间格式
    date_format = mpl.dates.DateFormatter("%H:%M")
    ax1.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.tight_layout()

    # 保存图像
    plt.savefig('lstm_lag_correction.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_input_data(X_test_original, y_test_original, scaler, lag, filename='lstm_input_data.csv'):
    """
    保存用于预测的输入数据

    # Arguments
        X_test_original: 原始的X_test数据（标准化后的）
        y_test_original: 原始的y_test数据（标准化后的）
        scaler: 标准化器对象
        lag: 时间窗口大小
        filename: 保存的文件名
    """
    # 反标准化输入特征
    # 将三维数据转换为二维进行反标准化
    X_test_2d = X_test_original.reshape(X_test_original.shape[0], X_test_original.shape[1])
    X_test_inverse = scaler.inverse_transform(X_test_2d)

    # 反标准化目标值
    y_test_inverse = scaler.inverse_transform(y_test_original.reshape(-1, 1)).flatten()

    # 创建DataFrame来保存输入数据
    # 为每个时间窗口创建列名
    column_names = []
    for i in range(lag, 0, -1):
        column_names.append(f'lag_{i}')

    # 创建输入数据DataFrame
    input_df = pd.DataFrame(X_test_inverse, columns=column_names)

    # 添加时间戳
    start_time = pd.Timestamp('2016-03-04 00:00:00')
    time_index = []
    for i in range(len(input_df)):
        # 每个预测对应的时间点是窗口结束后的下一个时间点
        time_index.append(start_time + pd.Timedelta(minutes=5*(i+lag)))

    input_df.insert(0, 'prediction_time', time_index)

    # 添加真实值
    # 注意：y_test对应的是每个输入样本预测的目标值
    input_df['true_value'] = y_test_inverse

    # 保存到CSV
    input_df.to_csv(filename, index=False)
    print(f"Input data saved to '{filename}'")

    return input_df


def main():
    from keras.src.losses import mean_squared_error as mse
    # 只加载LSTM模型
    lstm = load_model('model/lstm.h5', custom_objects={'mse': mse})

    lag = 12
    file1 = 'data/train.csv'
    file2 = 'data/test.csv'
    _, _, X_test, y_test, scaler = process_data(file1, file2, lag)

    # 保存原始测试数据用于后续分析
    X_test_original = X_test.copy()
    y_test_original = y_test.copy()

    # 反标准化真实值用于评估
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    # 准备测试数据
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # 保存输入数据到CSV
    input_data_df = save_input_data(
        X_test_original,
        y_test_original,
        scaler,
        lag,
        filename='lstm_input_data.csv'
    )

    # 进行预测
    predicted = lstm.predict(X_test)
    predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]

    # 只取前288个点用于显示和评估
    n_points = 288
    original_pred = predicted[:n_points]
    y_test_subset = y_test[:n_points]

    # 应用平移
    shifted_pred = shift_predictions(original_pred)

    # 打印原始预测结果
    print("=" * 60)
    print("LSTM - Original Predictions (First 288 points):")
    print("=" * 60)
    eva_regress(y_test_subset, original_pred)

    # 打印平移后预测结果
    print("\n" + "=" * 60)
    print("LSTM - Shifted Predictions (After Lag Correction):")
    print("=" * 60)
    eva_regress(y_test_subset, shifted_pred)

    # 计算改进效果
    original_mape = MAPE(y_test_subset, original_pred)
    shifted_mape = MAPE(y_test_subset, shifted_pred)
    improvement = ((original_mape - shifted_mape) / original_mape) * 100
    print(f"\nMAPE Improvement after lag correction: {improvement:.2f}%")

    # 验证滞后假设
    if len(y_test_subset) > 2:
        # 检查原始预测是否与下一个时间点的真实值更匹配
        min_len = min(len(original_pred) - 1, len(y_test_subset) - 1)
        if min_len > 1:
            # 原始预测[t] 与 真实值[t+1] 的相关性
            original_pred_truncated = original_pred[:min_len]
            y_test_next = y_test_subset[1:1+min_len]
            correlation_lagged = np.corrcoef(original_pred_truncated, y_test_next)[0, 1]

            # 原始预测[t] 与 真实值[t] 的相关性
            correlation_same_time = np.corrcoef(original_pred[:min_len], y_test_subset[:min_len])[0, 1]

            print(f"\nLag Analysis:")
            print(f"Correlation between original_pred[t] and y_true[t+1]: {correlation_lagged:.4f}")
            print(f"Correlation between original_pred[t] and y_true[t]: {correlation_same_time:.4f}")

            if correlation_lagged > correlation_same_time:
                print("✓ LAG CONFIRMED: Predictions are indeed lagged by one time unit")
                lag_ratio = correlation_lagged / correlation_same_time if correlation_same_time != 0 else float('inf')
                print(f"  Lag correlation is {lag_ratio:.2f} times higher than same-time correlation")
            else:
                print("✗ LAG NOT CONFIRMED: Predictions may not be lagged")

    # 绘制结果对比
    plot_results(y_test_subset, original_pred, shifted_pred)

    # 保存调整后的预测结果到文件
    results_df = pd.DataFrame({
        'Time': pd.date_range('2016-3-4 00:00', periods=len(y_test_subset), freq='5min'),
        'True_Value': y_test_subset,
        'Original_Prediction': original_pred,
        'Shifted_Prediction': shifted_pred
    })
    results_df.to_csv('lstm_lag_correction_results.csv', index=False)
    print("\nResults saved to 'lstm_lag_correction_results.csv'")

    # 创建一个包含所有信息的综合CSV文件
    # 获取与预测对应的输入数据
    input_for_predictions = input_data_df.iloc[:n_points].copy()
    input_for_predictions['Original_Prediction'] = original_pred
    input_for_predictions['Shifted_Prediction'] = shifted_pred

    # 重命名列名使其更清晰
    column_mapping = {'prediction_time': 'Prediction_Time', 'true_value': 'True_Value'}
    input_for_predictions.rename(columns=column_mapping, inplace=True)

    # 保存综合数据
    comprehensive_filename = 'lstm_comprehensive_data.csv'
    input_for_predictions.to_csv(comprehensive_filename, index=False)
    print(f"Comprehensive data saved to '{comprehensive_filename}'")

    # 打印数据信息
    print(f"\nData Information:")
    print(f"Number of predictions: {len(original_pred)}")
    print(f"Time window (lag): {lag}")
    print(f"Input data shape: {X_test_original.shape}")
    print(f"Generated files:")
    print(f"  1. lstm_input_data.csv - 包含所有用于预测的输入数据")
    print(f"  2. lstm_lag_correction_results.csv - 包含预测结果对比")
    print(f"  3. lstm_comprehensive_data.csv - 包含输入数据和预测结果的综合文件")
    print(f"  4. lstm_lag_correction.png - 预测对比图表")


if __name__ == '__main__':
    main()
