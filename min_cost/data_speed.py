import pandas as pd
import numpy as np

def calculate_speed_from_flow(input_file, output_file):
    # 1. 读取数据
    df = pd.read_csv(input_file)

    # 2. 数据预处理
    # 假设 Shifted_Prediction 是 5 分钟间隔的流量计数
    # 转换为小时流量 (veh/h)
    hourly_flow = df['Shifted_Prediction'] * 12

    # 3. 设定参数 (基于加州州际高速公路标准)
    # 自由流速度 v0 = 110 km/h (约 68 mph)
    v0 = 80
    # 通行能力 Capacity = 2000 veh/h (单车道标准)
    capacity = 1000.0
    # BPR 模型参数
    alpha = 0.15
    beta = 4.0

    # 4. 计算速度 (BPR 函数)
    # v = v0 / (1 + alpha * (q / C)^beta)
    # 增加 np.maximum 确保流量非负
    term = 1 + alpha * np.power((np.maximum(hourly_flow, 0) / capacity), beta)
    df['speed'] = v0 / term

    # 5. 保存结果
    df.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存至 {output_file}")
    print(f"速度单位: km/h")

# 执行转换
calculate_speed_from_flow('lstm_lag_correction_results.csv', 'lstm_lag_correction_results_with_speed.csv')
