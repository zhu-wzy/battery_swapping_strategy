import pandas as pd
import numpy as np

def calculate_load_std(input_file, output_file):
    # 1. 读取数据
    df = pd.read_csv(input_file)

    # 2. 筛选充电站
    mask = df['Type'] == 'f'
    stations = df[mask].copy()

    # 3. 获取核心参数
    # 当前所有站点的排队时间列表
    current_waits = stations['wait_time_min'].values
    # 各站点的充电桩数量
    num_chargers = stations['num_chargers'].values

    # 服务率 (Service Rate) mu = 2.0 辆/小时
    mu = 2.0

    # 计算每个站点"增加一辆车"带来的边际时间成本 (分钟)
    # Delta T = 1 / (N * mu) 小时 = 60 / (N * mu) 分钟
    # 注意：防止除以0 (虽然数据中桩数应该>0)
    marginal_cost = 60.0 / (np.maximum(num_chargers, 1) * mu)

    # 4. 循环计算 "What-if" 标准差
    load_std_results = []

    for i in range(len(stations)):
        # 复制一份当前的排队时间数组
        temp_waits = current_waits.copy()

        # 假设：如果车去了第 i 个站，该站排队时间增加
        temp_waits[i] += marginal_cost[i]

        # 计算新的系统标准差
        new_std = np.std(temp_waits)
        load_std_results.append(new_std)

    # 5. 保存结果
    df.loc[mask, 'load_std'] = load_std_results
    df['load_std'] = df['load_std'].fillna(0) # 非站点填充0

    df.to_csv(output_file, index=False)
    print(f"计算完成，结果已保存至: {output_file}")

# 运行
calculate_load_std('c101_21_queuing_cost.csv', 'c101_21_load_std.csv')
