import pandas as pd
import matplotlib.pyplot as plt
import re

def plot_stations_only(file_path):
    data_rows = []
    header = None

    # 1. 读取并解析数据
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 识别表头
        if line.startswith('StringID') and header is None:
            header = line.split()
            continue

        # 跳过底部的参数行
        if '/' in line:
            continue

        # 识别数据行
        parts = line.split()
        if header and len(parts) == len(header):
            if re.match(r'^[DSC]\d+', parts[0]):
                data_rows.append(parts)

    if header and data_rows:
        df = pd.DataFrame(data_rows, columns=header)
        df['x'] = pd.to_numeric(df['x'])
        df['y'] = pd.to_numeric(df['y'])
    else:
        print("未找到有效数据")
        return

    # 2. 数据筛选：排除客户 (Type == 'c')
    # 只保留 Depot ('d') 和 Station ('f')
    df_filtered = df[df['Type'] != 'c']

    depot = df_filtered[df_filtered['Type'] == 'd']
    stations = df_filtered[df_filtered['Type'] == 'f']

    # 3. 绘制散点图
    plt.figure(figsize=(10, 8))

    # 绘制充电站 (绿色三角形)
    plt.scatter(stations['x'], stations['y'],
                c='green', marker='^', label='Station', alpha=0.8, s=60)

    # 绘制仓库 (红色方块)
    plt.scatter(depot['x'], depot['y'],
                c='red', marker='s', label='Depot', s=100, edgecolors='black')

    # 添加图例、标题和标签
    plt.title('Node Locations (Customers Hidden)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()

if __name__ == "__main__":
    plot_stations_only('c101_21.txt')
