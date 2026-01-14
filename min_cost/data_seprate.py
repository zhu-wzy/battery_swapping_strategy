import pandas as pd
import numpy as np

# 1. 读取数据
file_path = 'lstm_lag_correction_results_with_speed.csv'
df = pd.read_csv(file_path)

# 2. 准备参数
n_roads = 20  # 目标路段数
total_rows = len(df)

# 计算每条路应该有多少行数据 (288 // 20 = 14 行)
rows_per_road = total_rows // n_roads

# 计算需要截取的总长度 (20 * 14 = 280)，多余的尾部数据将被丢弃
cutoff = n_roads * rows_per_road

# 3. 提取并重塑数据
# 获取 speed 列的前 cutoff 个数据
speed_values = df['speed'].values[:cutoff]

# 重塑数组：
# reshape(n_roads, rows_per_road) -> 变成 (20, 14) 的矩阵
# .T (转置) -> 变成 (14, 20)，即 14行 x 20列，每一列代表一条路
reshaped_data = speed_values.reshape(n_roads, rows_per_road).T

# 4. 创建新的 DataFrame
column_names = [f'Road_{i+1}' for i in range(n_roads)]
df_roads = pd.DataFrame(reshaped_data, columns=column_names)

# 5. 保存或查看结果
print(f"原始数据行数: {total_rows}")
print(f"每条路数据行数: {rows_per_road}")
print(f"生成的表格形状: {df_roads.shape} (行, 列)")

# 保存为新文件
df_roads.to_csv('reshaped_road_speeds.csv', index=False)
print("已保存为 reshaped_road_speeds.csv")
print(df_roads.head())
