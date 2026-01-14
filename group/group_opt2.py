import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 核心物理模型函数
# ==========================================

def mmn_analytical(lamda, mu, N):
    """ 计算 M/M/N 排队模型的等待时间 (Wq) """
    rho = lamda / mu
    if rho >= N: return 999.0
    sum_k = sum([(rho**k)/math.factorial(k) for k in range(int(N))])
    term_N = (rho**N) / (math.factorial(int(N)) * (1 - rho/N))
    P0 = 1 / (sum_k + term_N)
    P_wait = (rho**N) / (math.factorial(int(N)) * (1 - rho/N)) * P0
    Lq = P_wait * (rho / (N - rho))
    Wq = Lq / lamda
    return Wq

def calculate_travel_time_segment(dist, speed_profile):
    """ 分段积分法计算行驶时间 """
    if dist <= 0: return 0.0
    accumulated_dist = 0.0
    elapsed_time = 0.0
    interval_hour = 5.0 / 60.0
    for v in speed_profile:
        dist_step = v * interval_hour
        if accumulated_dist + dist_step >= dist:
            remaining = dist - accumulated_dist
            if v > 0: elapsed_time += (remaining / v) * 60.0
            return elapsed_time
        accumulated_dist += dist_step
        elapsed_time += 5.0
    last_speed = speed_profile[-1]
    if last_speed > 0: elapsed_time += ((dist - accumulated_dist) / last_speed) * 60.0
    return elapsed_time

# ==========================================
# 2. 遗传算法求解器 (综合优化版)
# ==========================================

def run_ga_advanced(nodes_file, speeds_file, output_csv='ga_final_result.csv'):
    print(f"🔹 [系统启动] 读取数据中...")
    try:
        nodes_df = pd.read_csv(nodes_file)
    except:
        nodes_df = pd.read_excel(nodes_file)
    speeds_df = pd.read_csv(speeds_file)

    users = nodes_df[nodes_df['Type'] == 'd'].reset_index(drop=True)
    stations = nodes_df[nodes_df['Type'] == 'f'].reset_index(drop=True)

    num_users = len(users)
    num_stations = len(stations)

    # --- 1. 预计算静态参数 ---
    # 计算基础排队时间 (Base Wait Time) 和 边际排队成本 (Marginal Cost)
    depot_pos = nodes_df[nodes_df['StringID'] == 'D0']
    dx, dy = (depot_pos.iloc[0]['x'], depot_pos.iloc[0]['y']) if not depot_pos.empty else (40, 50)

    stations['dist_to_depot'] = np.sqrt((stations['x'] - dx)**2 + (stations['y'] - dy)**2)
    stations_sorted = stations.sort_values('dist_to_depot')

    groups = [{'N': 8, 'lam': 15.0}, {'N': 6, 'lam': 10.0},
              {'N': 4, 'lam': 5.0}, {'N': 3, 'lam': 2.0}]
    mu = 2.0

    # 存储站点属性
    station_base_wait = np.zeros(num_stations)
    station_marginal_cost = np.zeros(num_stations) # 每多一人增加的时间

    group_size = math.ceil(num_stations / 4)

    for i, (idx, row) in enumerate(stations_sorted.iterrows()):
        g_idx = min(i // group_size, 3)
        params = groups[g_idx]
        lam = params['lam'] + (group_size - (i % group_size)) * 0.1
        N = params['N']

        # 计算基础等待时间
        wq_hours = mmn_analytical(lam, mu, N)
        station_base_wait[idx] = wq_hours * 60.0

        # 计算边际成本: 假设多一人排队，时间增加 60/(N*mu) 分钟
        # 这是一个简化估算，用于动态惩罚
        station_marginal_cost[idx] = 60.0 / (N * mu)

    # --- 2. 预计算行驶时间矩阵 (Travel Time Matrix) ---
    # 这一步是静态的，可以预先算好以节省时间
    print("🔹 [预处理] 计算行驶时间矩阵...")
    travel_time_matrix = np.zeros((num_users, num_stations))

    for u_idx, user in users.iterrows():
        for s_idx, station in stations.iterrows():
            dist = np.sqrt((user['x'] - station['x'])**2 + (user['y'] - station['y'])**2)
            speed_idx = s_idx % speeds_df.shape[1]
            speed_profile = speeds_df.iloc[:, speed_idx].values
            tt = calculate_travel_time_segment(dist, speed_profile)
            travel_time_matrix[u_idx, s_idx] = tt

    # --- 3. 遗传算法配置 ---
    POP_SIZE = 200
    GENERATIONS = 800
    MUTATION_RATE = 0.2

    # !!! 权重设置 !!!
    # 总目标 = Time_Cost + (W * Std_Dev)
    # 建议 W 取 10~20，因为 Time 通常几百，Std 通常几十
    WEIGHT_STD = 15.0

    def create_individual():
        return [random.randint(0, num_stations - 1) for _ in range(num_users)]

    # === 核心修改：动态适应度函数 ===
    def calculate_fitness(ind):
        # 1. 统计每个站点的分配人数
        counts = np.zeros(num_stations)
        for s in ind:
            counts[s] += 1

        # 2. 计算动态排队时间
        # 某个站的排队时间 = 基础时间 + (该站人数 * 边际时间)
        # 注意：这里我们简化处理，假设分配到该站的所有人都要承受这个拥挤度
        dynamic_wait_times = station_base_wait + (counts * station_marginal_cost)

        # 3. 计算总时间成本
        total_time = 0
        for u_i, s_j in enumerate(ind):
            # 用户u去站点s: 行驶时间(固定) + 站点s当前的动态排队时间
            total_time += travel_time_matrix[u_i, s_j] + dynamic_wait_times[s_j]

        # 4. 计算负载均衡度 (Std)
        # 计算所有站点“拥挤后等待时间”的标准差
        load_std = np.std(dynamic_wait_times)

        # 5. 综合得分 (越小越好)
        score = total_time + (WEIGHT_STD * load_std)
        return score, total_time, load_std

    def crossover(p1, p2):
        size = len(p1)
        if size < 2: return p1[:]
        cx1 = random.randint(0, size - 1)
        cx2 = random.randint(0, size - 1)
        if cx1 > cx2: cx1, cx2 = cx2, cx1
        child = p1[:]
        child[cx1:cx2+1] = p2[cx1:cx2+1]
        return child

    def mutate(ind):
        for i in range(len(ind)):
            if random.random() < MUTATION_RATE:
                ind[i] = random.randint(0, num_stations - 1)
        return ind

    print(f"🔹 [优化开始] 启动遗传算法 ({GENERATIONS} 代)...")
    population = [create_individual() for _ in range(POP_SIZE)]

    history_score = []
    history_time = []
    history_std = []

    global_best_ind = None
    global_best_score = float('inf')
    global_best_metrics = (0, 0) # time, std

    for gen in range(GENERATIONS):
        # 计算所有个体的适应度
        # returns list of (ind, (score, time, std))
        results = []
        for ind in population:
            s, t, std = calculate_fitness(ind)
            results.append((ind, s, t, std))

        # 排序
        results.sort(key=lambda x: x[1])

        best_of_gen = results[0]
        history_score.append(best_of_gen[1])
        history_time.append(best_of_gen[2])
        history_std.append(best_of_gen[3])

        if best_of_gen[1] < global_best_score:
            global_best_score = best_of_gen[1]
            global_best_ind = best_of_gen[0][:]
            global_best_metrics = (best_of_gen[2], best_of_gen[3])

        # 选择
        selected = [x[0] for x in results[:POP_SIZE//2]]

        next_pop = [global_best_ind[:]]
        while len(next_pop) < POP_SIZE:
            p1 = random.choice(selected)
            p2 = random.choice(selected)
            child = crossover(p1, p2)
            child = mutate(child)
            next_pop.append(child)

        population = next_pop

    print(f"✅ 优化完成!")
    print(f"   🏆 综合得分: {global_best_score:.2f}")
    print(f"   ⏱️ 真实总时间: {global_best_metrics[0]:.2f} min (已包含动态拥挤成本)")
    print(f"   ⚖️ 负载均衡度: {global_best_metrics[1]:.4f}")

    # --- 绘图 (Show Only) ---
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 双轴绘图
    fig, ax1 = plt.subplots(figsize=(10,6))

    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('综合得分 (Score)', color='red')
    ax1.plot(history_score, color='red', linewidth=2, label='Score')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('负载均衡度 (Std)', color='blue')
    ax2.plot(history_std, color='blue', linestyle='--', linewidth=1.5, label='Load Std')
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.title('遗传算法优化过程: 综合成本 & 均衡度下降')
    fig.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.scatter(stations['x'], stations['y'], c='green', marker='^', s=100, label='充电站', zorder=5)
    plt.scatter(users['x'], users['y'], c='blue', marker='o', s=60, label='用户', zorder=5)

    for u_idx, s_idx in enumerate(global_best_ind):
        u_pt = users.iloc[u_idx]
        s_pt = stations.iloc[s_idx]
        plt.plot([u_pt['x'], s_pt['x']], [u_pt['y'], s_pt['y']], 'k--', alpha=0.2)

    plt.title(f'最优匹配方案\n总时间:{global_best_metrics[0]:.1f} | 均衡度:{global_best_metrics[1]:.2f}')
    plt.legend()
    plt.show()

    # --- 保存结果 ---
    # 需要重新计算一次最终的单项成本
    station_counts = np.zeros(num_stations)
    for s in global_best_ind: station_counts[s] += 1
    final_wait_times = station_base_wait + (station_counts * station_marginal_cost)

    results = []
    for u, s in enumerate(global_best_ind):
        tt = travel_time_matrix[u, s]
        wt = final_wait_times[s]
        results.append({
            'User_ID': users.iloc[u]['StringID'],
            'Assigned_Station': stations.iloc[s]['StringID'],
            'Travel_Time': round(tt, 2),
            'Dynamic_Wait_Time': round(wt, 2),
            'Total_Cost': round(tt + wt, 2)
        })
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"💾 最终结果表已保存: {output_csv}")

if __name__ == "__main__":
    node_file = 'c101_21_added_19_station_type.csv'
    speed_file = 'reshaped_road_speeds.csv'

    if os.path.exists(node_file) and os.path.exists(speed_file):
        run_ga_advanced(node_file, speed_file)
    else:
        print("❌ 错误：找不到输入文件。")
