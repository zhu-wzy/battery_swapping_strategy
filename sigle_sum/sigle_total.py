import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 核心算法函数
# ==========================================

def mmn_analytical(lamda, mu, N):
    """
    M/M/N 排队论公式计算
    """
    rho = lamda / mu
    if rho >= N:
        return 999.0 # 超负荷

    sum_k = sum([(rho**k)/math.factorial(k) for k in range(N)])
    term_N = (rho**N) / (math.factorial(N) * (1 - rho/N))
    P0 = 1 / (sum_k + term_N)

    P_wait = (rho**N) / (math.factorial(N) * (1 - rho/N)) * P0
    Lq = P_wait * (rho / (N - rho))
    Wq = Lq / lamda
    return Wq # 单位：小时

def calculate_travel_time(row, speeds_df):
    """
    分段积分法计算行驶时间 (分钟)
    """
    if row['Type'] != 'f': return 0.0

    # 解析 ID (S1 -> Road_1)
    try:
        road_num = int(row['StringID'][1:])
        col_name = f'Road_{road_num}'
    except:
        return 0.0

    if col_name not in speeds_df.columns: return 0.0

    target_distance = row['distance']
    if target_distance <= 0: return 0.0

    speed_profile = speeds_df[col_name].values
    accumulated_dist = 0.0
    elapsed_time = 0.0
    interval_hour = 5.0 / 60.0 # 5分钟间隔

    for v in speed_profile:
        dist_step = v * interval_hour
        if accumulated_dist + dist_step >= target_distance:
            remaining = target_distance - accumulated_dist
            if v > 0:
                elapsed_time += (remaining / v) * 60.0
            return elapsed_time
        accumulated_dist += dist_step
        elapsed_time += 5.0

    # 如果跑完所有数据还没到，按最后速度估算
    last_speed = speed_profile[-1]
    if last_speed > 0:
        elapsed_time += ((target_distance - accumulated_dist) / last_speed) * 60.0
    return elapsed_time

# ==========================================
# 2. 主处理流程
# ==========================================

def run_optimization(nodes_file, speeds_file, output_csv, output_plot, w1=0.5, w2=0.5):
    print(f"正在读取文件...\n  节点: {nodes_file}\n  车速: {speeds_file}")

    # --- 读取数据 ---
    try:
        if nodes_file.endswith('.csv'):
            df = pd.read_csv(nodes_file)
        else:
            df = pd.read_excel(nodes_file)
        speeds_df = pd.read_csv(speeds_file)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 筛选充电站
    mask = df['Type'] == 'f'
    stations = df[mask].copy()

    # -------------------------------------------------
    # 步骤 1: 计算行驶时间 (Travel Cost)
    # -------------------------------------------------
    print("步骤 1/4: 计算行驶时间...")
    df.loc[mask, 'travel_cost'] = df[mask].apply(lambda row: calculate_travel_time(row, speeds_df), axis=1)

    # -------------------------------------------------
    # 步骤 2: 生成排队数据 (基于距离的确定性生成)
    # -------------------------------------------------
    print("步骤 2/4: 生成排队数据...")
    # 按距离排序，用于分配参数
    stations_sorted = df[mask].sort_values('distance')

    # 参数组：(桩数 N, 基础到达率 lambda)
    # 分4组，距离近的繁忙(N大, lambda大)，距离远的空闲
    groups = [
        {'N': 8, 'lam': 15.0},
        {'N': 6, 'lam': 10.0},
        {'N': 4, 'lam': 5.0},
        {'N': 3, 'lam': 2.0}
    ]
    mu = 2.0 # 服务率

    group_size = math.ceil(len(stations_sorted) / 4)

    for i, (idx, row) in enumerate(stations_sorted.iterrows()):
        g_idx = min(i // group_size, 3)
        params = groups[g_idx]

        # 组内微调：越近 lambda 稍微越大
        lam_adj = (group_size - (i % group_size)) * 0.1
        lam = params['lam'] + lam_adj
        N = params['N']

        # 计算排队时间
        wq_hours = mmn_analytical(lam, mu, N)

        df.loc[idx, 'num_chargers'] = int(N)
        df.loc[idx, 'arrival_rate'] = round(lam, 2)
        df.loc[idx, 'wait_time_min'] = wq_hours * 60.0

    # -------------------------------------------------
    # 步骤 3: 计算负载均衡指标 (Load Std)
    # -------------------------------------------------
    print("步骤 3/4: 计算负载均衡指标...")
    # 重新获取更新后的 station 数据
    stations = df[mask].copy()
    current_waits = stations['wait_time_min'].fillna(0).values
    num_chargers = stations['num_chargers'].fillna(1).values

    # 计算每站增加一辆车的边际时间成本 (分钟) = 60 / (N*mu)
    marginal_costs = 60.0 / (np.maximum(num_chargers, 1) * mu)

    load_stds = []
    for i in range(len(stations)):
        temp_waits = current_waits.copy()
        temp_waits[i] += marginal_costs[i] # 模拟选择该站
        load_stds.append(np.std(temp_waits))

    df.loc[mask, 'load_std'] = load_stds

    # -------------------------------------------------
    # 步骤 4: 多目标优化与评分 (Total Cost)
    # -------------------------------------------------
    print("步骤 4/4: 多目标优化计算...")
    stations = df[mask].copy()

    # 计算总时间
    stations['total_time'] = stations['travel_cost'] + stations['wait_time_min']

    # 归一化
    t_min, t_max = stations['total_time'].min(), stations['total_time'].max()
    s_min, s_max = stations['load_std'].min(), stations['load_std'].max()

    def norm(s, mn, mx):
        return (s - mn) / (mx - mn) if mx > mn else 0

    stations['norm_time'] = norm(stations['total_time'], t_min, t_max)
    stations['norm_std']  = norm(stations['load_std'], s_min, s_max)

    # 计算 Total Cost
    stations['total_cost'] = w1 * stations['norm_time'] + w2 * stations['norm_std']

    # 回填
    df.loc[mask, 'total_cost'] = stations['total_cost']

    # 填充空值
    fill_cols = ['travel_cost', 'num_chargers', 'arrival_rate', 'wait_time_min', 'load_std', 'total_cost']
    for c in fill_cols:
        df[c] = df[c].fillna(0)

    # 保存结果
    df.to_csv(output_csv, index=False)
    print(f"✅ 结果已保存: {output_csv}")

    # -------------------------------------------------
    # 绘图
    # -------------------------------------------------
    best_station = stations.loc[stations['total_cost'].idxmin()]
    depot = df[df['Type'] == 'd']

    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei'] # Win
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制普通站
    plt.scatter(stations['x'], stations['y'], c='gray', alpha=0.6, s=60, label='其他充电站')

    # 绘制仓库
    if not depot.empty:
        plt.scatter(depot['x'], depot['y'], c='black', marker='s', s=100, label='当前位置')

    # 绘制最优站
    plt.scatter(best_station['x'], best_station['y'], c='red', s=250, marker='*', zorder=10,
                label=f"最优选择: {best_station['StringID']}")

    # 标注
    txt = f"{best_station['StringID']}\nCost:{best_station['total_cost']:.4f}\nTime:{best_station['total_time']:.1f}m"
    plt.annotate(txt, (best_station['x'], best_station['y']), xytext=(15, 15),
                 textcoords='offset points', bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.title(f"多目标最优充电站推荐 (w1={w1}, w2={w2})")
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(output_plot, dpi=300)
    print(f"✅ 图表已保存: {output_plot}")
    print("="*40)
    print(f"🏆 最优站点: {best_station['StringID']}")
    print(f"   综合评分: {best_station['total_cost']:.4f}")
    print(f"   总耗时:   {best_station['total_time']:.2f} 分钟")
    print(f"   负载均衡: {best_station['load_std']:.4f}")
    print("="*40)

# ==========================================
# 3. 运行接口
# ==========================================
if __name__ == "__main__":
    # 请确保这两个文件在当前目录下
    input_nodes = 'c101_21_with_distance.xlsx'  # 第一个文件 (含 distance)
    input_speeds = 'reshaped_road_speeds.csv'   # 第二个文件 (含车速)

    output_data = 'c101_21_final_result.csv'     # 输出表格
    output_img = 'optimal_station_selection.png' # 输出图片

    # 检查文件是否存在
    if not os.path.exists(input_speeds):
        print(f"❌ 找不到文件: {input_speeds}")
    else:
        run_optimization(input_nodes, input_speeds, output_data, output_img)
