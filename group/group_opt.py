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
    """
    计算 M/M/N 排队模型的等待时间 (Wq)
    :param lamda: 到达率 (辆/小时)
    :param mu: 服务率 (辆/小时)
    :param N: 充电桩数量
    :return: 平均等待时间 (小时)
    """
    rho = lamda / mu
    if rho >= N:
        return 999.0 # 系统过载惩罚值

    sum_k = sum([(rho**k)/math.factorial(k) for k in range(int(N))])
    term_N = (rho**N) / (math.factorial(int(N)) * (1 - rho/N))
    P0 = 1 / (sum_k + term_N)

    P_wait = (rho**N) / (math.factorial(int(N)) * (1 - rho/N)) * P0
    Lq = P_wait * (rho / (N - rho))
    Wq = Lq / lamda
    return Wq

def calculate_travel_time_segment(dist, speed_profile):
    """
    分段积分法计算行驶时间
    :param dist: 距离 (km)
    :param speed_profile: 速度列表 (km/h)，假设每5分钟变一次
    :return: 行驶时间 (分钟)
    """
    if dist <= 0: return 0.0

    accumulated_dist = 0.0
    elapsed_time = 0.0
    interval_hour = 5.0 / 60.0 # 5分钟时间步长

    for v in speed_profile:
        dist_step = v * interval_hour
        if accumulated_dist + dist_step >= dist:
            remaining = dist - accumulated_dist
            if v > 0:
                elapsed_time += (remaining / v) * 60.0
            return elapsed_time
        accumulated_dist += dist_step
        elapsed_time += 5.0

    # 如果跑完所有时间段还没到，按最后时刻速度估算
    last_speed = speed_profile[-1]
    if last_speed > 0:
        elapsed_time += ((dist - accumulated_dist) / last_speed) * 60.0
    return elapsed_time

# ==========================================
# 2. 遗传算法求解器 (通用版)
# ==========================================

def run_ga_matching(nodes_file, speeds_file, output_csv='ga_final_result.csv'):
    print(f"🔹 正在读取数据...")

    # 1. 读取数据
    try:
        nodes_df = pd.read_csv(nodes_file)
    except:
        nodes_df = pd.read_excel(nodes_file)
    speeds_df = pd.read_csv(speeds_file)

    # 分离用户和充电站
    users = nodes_df[nodes_df['Type'] == 'd'].reset_index(drop=True)
    stations = nodes_df[nodes_df['Type'] == 'f'].reset_index(drop=True)

    num_users = len(users)
    num_stations = len(stations)
    print(f"🔹 检测到: 用户 {num_users} 人, 充电站 {num_stations} 个")

    # 2. 生成充电站排队参数 (基于距离逻辑)
    # 获取参考点 D0 (或第一个用户)
    depot_pos = nodes_df[nodes_df['StringID'] == 'D0']
    if not depot_pos.empty:
        dx, dy = depot_pos.iloc[0]['x'], depot_pos.iloc[0]['y']
    else:
        dx, dy = users.iloc[0]['x'], users.iloc[0]['y']

    stations['dist_to_depot'] = np.sqrt((stations['x'] - dx)**2 + (stations['y'] - dy)**2)
    stations_sorted = stations.sort_values('dist_to_depot')

    # 定义参数组 (桩数N, 到达率lambda)
    groups = [{'N': 8, 'lam': 15.0}, {'N': 6, 'lam': 10.0},
              {'N': 4, 'lam': 5.0}, {'N': 3, 'lam': 2.0}]
    mu = 2.0

    station_wait_times = np.zeros(num_stations)
    group_size = math.ceil(num_stations / 4)

    for i, (idx, row) in enumerate(stations_sorted.iterrows()):
        g_idx = min(i // group_size, 3)
        params = groups[g_idx]
        lam = params['lam'] + (group_size - (i % group_size)) * 0.1
        N = params['N']
        wq_hours = mmn_analytical(lam, mu, N)
        # 将结果存回对应的 station index (注意是 iloc 对应的顺序)
        # 因为我们之后是按 stations 的行序来索引的，所以需要映射回 row index
        # 这里 stations_sorted 的 index 是原始 stations df 的 index
        station_wait_times[idx] = wq_hours * 60.0

    # 3. 构建成本矩阵 Cost[User][Station]
    print("🔹 正在计算成本矩阵...")
    cost_matrix = np.zeros((num_users, num_stations))

    for u_idx, user in users.iterrows():
        for s_idx, station in stations.iterrows():
            dist = np.sqrt((user['x'] - station['x'])**2 + (user['y'] - station['y'])**2)

            # 匹配车速数据 (S1->Road_1, 超过20则循环)
            speed_idx = s_idx % speeds_df.shape[1]
            speed_profile = speeds_df.iloc[:, speed_idx].values

            tt = calculate_travel_time_segment(dist, speed_profile)
            wt = station_wait_times[s_idx]

            cost_matrix[u_idx, s_idx] = tt + wt

    # 4. 遗传算法配置
    POP_SIZE = 200        # 种群大小
    GENERATIONS = 500     # 迭代次数
    MUTATION_RATE = 0.2   # 变异率

    # 编码：整数编码，长度=num_users，值范围=[0, num_stations-1]
    def create_individual():
        return [random.randint(0, num_stations - 1) for _ in range(num_users)]

    def calculate_fitness(ind):
        total_cost = 0
        for u_i, s_j in enumerate(ind):
            total_cost += cost_matrix[u_i, s_j]
        return total_cost

    def crossover(p1, p2):
        # 两点交叉
        size = len(p1)
        if size < 2: return p1[:]
        cx1 = random.randint(0, size - 1)
        cx2 = random.randint(0, size - 1)
        if cx1 > cx2: cx1, cx2 = cx2, cx1

        child = p1[:]
        child[cx1:cx2+1] = p2[cx1:cx2+1]
        return child

    def mutate(ind):
        # 随机重置变异
        for i in range(len(ind)):
            if random.random() < MUTATION_RATE:
                ind[i] = random.randint(0, num_stations - 1)
        return ind

    # 5. 运行遗传算法
    print(f"🔹 启动遗传算法 ({GENERATIONS} 代)...")
    population = [create_individual() for _ in range(POP_SIZE)]
    best_loss_history = []
    global_best_ind = None
    global_best_cost = float('inf')

    for gen in range(GENERATIONS):
        scores = [(ind, calculate_fitness(ind)) for ind in population]
        scores.sort(key=lambda x: x[1])

        current_best = scores[0][1]
        best_loss_history.append(current_best)

        if current_best < global_best_cost:
            global_best_cost = current_best
            global_best_ind = scores[0][0][:]

        # 锦标赛选择
        selected = []
        for _ in range(POP_SIZE // 2):
            candidates = random.sample(scores, 3)
            winner = min(candidates, key=lambda x: x[1])[0]
            selected.append(winner)

        next_pop = [global_best_ind[:]] # 精英保留

        while len(next_pop) < POP_SIZE:
            p1 = random.choice(selected)
            p2 = random.choice(selected)
            child = crossover(p1, p2)
            child = mutate(child)
            next_pop.append(child)

        population = next_pop

    print(f"✅ 优化完成! 最低总成本: {global_best_cost:.2f} 分钟")

    # 6. 结果保存与绘图
    # 损失曲线
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei'] # Win
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(best_loss_history, color='red', linewidth=2)
    plt.title('遗传算法总成本下降曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('总成本 (min)')
    plt.grid(True, linestyle='--')
    plt.savefig('ga_loss_curve.png')
    print("📊 损失图已保存: ga_loss_curve.png")

    # 匹配连线图
    plt.figure(figsize=(12, 10))
    plt.scatter(stations['x'], stations['y'], c='green', marker='^', s=100, label='充电站', zorder=5)
    plt.scatter(users['x'], users['y'], c='blue', marker='o', s=60, label='用户', zorder=5)

    for u_idx, s_idx in enumerate(global_best_ind):
        u_pt = users.iloc[u_idx]
        s_pt = stations.iloc[s_idx]
        # 仅绘制前 100 条线防止过乱，或者全部绘制但透明度高
        plt.plot([u_pt['x'], s_pt['x']], [u_pt['y'], s_pt['y']], 'k--', alpha=0.2)

    plt.title(f'最优匹配方案 (用户={num_users}, 站点={num_stations})')
    plt.legend()
    plt.savefig('ga_matching_plot.png')
    print("🗺️ 匹配图已保存: ga_matching_plot.png")

    # 保存表格
    results = []
    for u, s in enumerate(global_best_ind):
        results.append({
            'User_ID': users.iloc[u]['StringID'],
            'Assigned_Station': stations.iloc[s]['StringID'],
            'Cost_Min': round(cost_matrix[u, s], 2)
        })
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"💾 结果已保存: {output_csv}")

if __name__ == "__main__":
    # 在这里修改你的文件名
    input_nodes = 'c101_21_added_19_station_type.csv'
    input_speeds = 'reshaped_road_speeds.csv'

    if os.path.exists(input_nodes) and os.path.exists(input_speeds):
        run_ga_matching(input_nodes, input_speeds)
    else:
        print("❌ 错误：未找到输入文件，请检查文件名。")
