import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib.pyplot as plt

# --- 加入这两行代码解决中文乱码 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# -------------------------------

def plot_connections(file_path):
    data_rows = []
    header = None

    # 1. 读取数据
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('StringID') and header is None:
            header = line.split()
            continue
        if '/' in line: continue

        parts = line.split()
        if header and len(parts) == len(header):
            if re.match(r'^[DSC]\d+', parts[0]):
                data_rows.append(parts)

    if header and data_rows:
        df = pd.DataFrame(data_rows, columns=header)
        df['x'] = pd.to_numeric(df['x'])
        df['y'] = pd.to_numeric(df['y'])

        # 2. 筛选 Depot 和 Stations
        depot = df[df['Type'] == 'd'].iloc[0]  # 假设只有一个仓库
        stations = df[df['Type'] == 'f']

        # 3. 绘图初始化
        plt.figure(figsize=(12, 10))

        # 绘制仓库
        plt.scatter(depot['x'], depot['y'], c='red', marker='s', s=150, zorder=10, label='Depot (车站)', edgecolors='black')

        # 绘制充电站
        plt.scatter(stations['x'], stations['y'], c='green', marker='^', s=80, zorder=10, label='Charging Station (充电站)')

        # 4. 连线并计算距离
        for _, station in stations.iterrows():
            # 计算距离
            dist = np.sqrt((station['x'] - depot['x'])**2 + (station['y'] - depot['y'])**2)

            # 绘制虚线
            plt.plot([depot['x'], station['x']], [depot['y'], station['y']],
                     c='gray', linestyle='--', alpha=0.5, zorder=1)

            # 在中点标注距离
            mid_x = (depot['x'] + station['x']) / 2
            mid_y = (depot['y'] + station['y']) / 2

            # 为了防止文字重叠，添加一个白色背景框
            plt.text(mid_x, mid_y, f'{dist:.1f}', fontsize=9, color='blue',
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

        plt.title('Distance between Depot and Charging Stations')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.show()

if __name__ == "__main__":
    plot_connections('c101_21.txt')
