import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_and_plot_new_nodes(input_file, output_file):
    # 1. 读取数据
    try:
        df = pd.read_csv(input_file)
    except:
        df = pd.read_excel(input_file)

    # 2. 获取参考类型 (第一行 D0 的类型)
    station_type = df.iloc[0]['Type'] # 通常为 'd'

    # 3. 生成 19 个随机点
    num_new = 49
    new_rows = []
    for i in range(num_new):
        row = {
            'StringID': f'New{i+1}',
            'Type': station_type, # 类型设为 'd'
            'x': np.random.randint(0, 100),
            'y': np.random.randint(0, 100)
        }
        # 填充其他列为 0
        for col in df.columns:
            if col not in row:
                row[col] = 0
        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)

    # 4. 插入数据 (在 index 0 和 1 之间)
    part1 = df.iloc[:1]
    part2 = df.iloc[1:]
    final_df = pd.concat([part1, new_df, part2]).reset_index(drop=True)

    # 5. 保存 CSV
    final_df.to_csv(output_file, index=False)
    print(f"新数据已保存至: {output_file}")

    # 6. 绘图
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制 D0
    d0 = final_df.iloc[0]
    plt.scatter(d0['x'], d0['y'], c='black', marker='s', s=150, label='原始车站 (D0)', zorder=10)

    # 绘制新生成的点 (New...)
    new_points = final_df[final_df['StringID'].str.startswith('New')]
    plt.scatter(new_points['x'], new_points['y'], c='red', marker='o', s=80, label=f'新生成用户点 (Type={station_type})', alpha=0.7)

    # 绘制原有充电站 (S...)
    stations = final_df[final_df['StringID'].str.startswith('S')]
    plt.scatter(stations['x'], stations['y'], c='green', marker='^', s=60, label='充电站 (S)', alpha=0.6)

    plt.title(f'节点分布图 (新增19个 Type={station_type} 的节点)')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig('nodes_distribution_d_type.png')
    plt.show()

# 运行
input_csv = 'c101_21.xlsx'
output_csv = 'c101_21_added_19_station_type.csv'
generate_and_plot_new_nodes(input_csv, output_csv)
