import pandas as pd
import numpy as np
import os

def add_distance_column(input_file, output_file):
    # 1. 读取 Excel 文件中的 Nodes 表
    # 确保文件和脚本在同一目录，或者提供绝对路径
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        return

    try:
        df = pd.read_excel(input_file, sheet_name='Nodes')

        # 如果还有 Config 表，也顺便读取，方便最后一起保存
        try:
            df_config = pd.read_excel(input_file, sheet_name='Config')
        except:
            df_config = pd.DataFrame() # 如果没有 Config 表则创建一个空的

    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    # 2. 获取车站 (Depot) 的坐标
    # 假设 Type 为 'd' 的是车站
    depot_row = df[df['Type'] == 'd']
    if depot_row.empty:
        print("错误：数据中未找到车站 (Type='d')")
        return

    # 取第一行作为车站 (通常只有一个)
    depot_x = depot_row.iloc[0]['x']
    depot_y = depot_row.iloc[0]['y']

    # 3. 计算距离
    # 公式：sqrt((x1-x2)^2 + (y1-y2)^2)
    # 我们只对充电站 (Type='f') 计算距离，或者对所有点计算都可以。
    # 这里根据你的要求，计算出距离并填入新列 'distance'

    def calculate_dist(row):
        # 如果是充电站 (f) 或 车站 (d)，计算距离
        # 如果你也想计算客户 (c) 的距离，去掉下面的 if 判断即可
        if row['Type'] == 'f':
            return np.sqrt((row['x'] - depot_x)**2 + (row['y'] - depot_y)**2)
        elif row['Type'] == 'd':
            return 0.0
        else:
            return None # 客户点暂不计算，或者你可以改为计算所有点

    # 应用函数创建新列
    df['distance'] = df.apply(calculate_dist, axis=1)

    # 4. 保存到新的 Excel 文件
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Nodes', index=False)
            if not df_config.empty:
                df_config.to_excel(writer, sheet_name='Config', index=False)

        print(f"✅ 处理完成！")
        print(f"包含距离的新文件已保存为: {os.path.abspath(output_file)}")

        # 打印一下预览
        print("\n--- 充电站距离预览 ---")
        print(df[df['Type'] == 'f'][['StringID', 'Type', 'x', 'y', 'distance']].head())

    except Exception as e:
        print(f"保存 Excel 失败: {e}")

if __name__ == "__main__":
    # 输入文件名 (你之前的 Excel)
    input_excel = 'c101_21.xlsx'
    # 输出文件名
    output_excel = 'c101_21_with_distance.xlsx'

    add_distance_column(input_excel, output_excel)
