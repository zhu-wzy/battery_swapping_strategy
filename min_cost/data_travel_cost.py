import pandas as pd
import numpy as np

def calculate_travel_time(row, speeds_df):
    """
    分段积分计算行驶时间
    :param row: Nodes 表中的一行
    :param speeds_df: 包含 Road_1 ~ Road_20 车速的 DataFrame
    :return: 行驶时间 (分钟)
    """
    # 1. 过滤：只计算充电站 (Type 'f')
    if row['Type'] != 'f':
        return None  # 非充电站不计算，或返回 0

    # 2. 匹配道路编号：S1 -> Road_1, S2 -> Road_2
    station_id = row['StringID']
    try:
        # 提取 'S' 后面的数字
        road_num = int(station_id[1:])
        col_name = f'Road_{road_num}'
    except:
        return None # 无法解析 ID

    # 检查该道路是否有速度数据
    if col_name not in speeds_df.columns:
        return None

    # 3. 核心算法：分段积分
    target_distance = row['distance'] # 目标总距离 (km)
    if target_distance <= 0:
        return 0.0

    # 获取该条路的速度分布 (单位: km/h)
    speed_profile = speeds_df[col_name].values

    accumulated_dist = 0.0 # 已行驶距离
    elapsed_time = 0.0     # 已消耗时间 (分钟)
    interval_min = 5.0     # 时间间隔 5 分钟
    interval_hour = 5.0 / 60.0 # 时间间隔 (小时)

    reached = False

    # 遍历每一个时间段
    for v in speed_profile:
        # 计算当前 5 分钟能跑多远
        dist_step = v * interval_hour

        # 如果 当前累计 + 这段距离 >= 目标距离，说明在这 5 分钟内到达了
        if accumulated_dist + dist_step >= target_distance:
            remaining_dist = target_distance - accumulated_dist
            if v > 0:
                # 计算最后这一小段需要多少分钟
                time_fraction_min = (remaining_dist / v) * 60.0
                elapsed_time += time_fraction_min
            reached = True
            break
        else:
            # 如果还没到，累加距离和时间，继续下一个 5 分钟
            accumulated_dist += dist_step
            elapsed_time += interval_min

    # 4. 异常处理：如果跑完了所有时间段还没到
    if not reached:
        # 假设以最后时刻的速度继续行驶
        last_speed = speed_profile[-1]
        remaining_dist = target_distance - accumulated_dist
        if last_speed > 0:
            time_extra_min = (remaining_dist / last_speed) * 60.0
            elapsed_time += time_extra_min
        else:
            return float('inf') # 速度为0，永远到不了

    return elapsed_time

def main():
    # --- 文件路径配置 ---
    nodes_file = 'c101_21_with_distance.xlsx'
    speeds_file = 'reshaped_road_speeds.csv'
    output_file = 'c101_21_final.xlsx'

    print("正在读取数据...")
    # 读取节点数据和配置数据
    nodes_df = pd.read_excel(nodes_file, sheet_name='Nodes')
    try:
        config_df = pd.read_excel(nodes_file, sheet_name='Config')
    except:
        config_df = pd.DataFrame() # 如果没有Config表则忽略

    # 读取速度数据
    speeds_df = pd.read_csv(speeds_file)

    print("正在计算行驶时间...")
    # 应用函数
    nodes_df['travel_cost'] = nodes_df.apply(lambda row: calculate_travel_time(row, speeds_df), axis=1)

    print("正在保存文件...")
    # 使用 ExcelWriter 保存，确保保留多个 Sheet
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        nodes_df.to_excel(writer, sheet_name='Nodes', index=False)
        if not config_df.empty:
            config_df.to_excel(writer, sheet_name='Config', index=False)

    print(f"✅ 处理完成！结果已保存至: {output_file}")
    # 打印部分结果预览
    print("\n结果预览 (前5个充电站):")
    print(nodes_df[nodes_df['Type'] == 'f'][['StringID', 'distance', 'travel_cost']].head())

if __name__ == "__main__":
    main()
