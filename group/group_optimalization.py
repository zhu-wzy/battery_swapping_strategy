import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心计算函数 (行驶时间 + 排队时间)
# ==========================================

def mmn_analytical(lamda, mu, N):
    """
    计算 M/M/N 排队模型的等待时间 (Wq)
    """
    rho = lamda / mu
    if rho >= N:
        return 999.0 # 系统过载惩罚值

    sum_k = sum([(rho**k)/math.factorial(k) for k in range(N)])
    term_N = (rho**N) / (math.factorial(N) * (1 - rho/N))
    P0 = 1 / (sum_k + term_N)

    P_wait = (rho**N) / (math.factorial(N) * (1 - rho/N)) * P0
    Lq = P_wait * (rho / (N - rho))
    Wq = Lq / lamda
    return Wq # 单位：小时

def calculate_travel_time_segment(dist, speed_profile):
    """
    分段积分法计算行驶时间 (分钟)
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

    # 如果跑完所有时间段还没到，按最后时刻速度估算剩余时间
    last_speed = speed_profile[-1]
    if last_speed > 0:
        elapsed_time += ((dist - accumulated_dist) / last_speed) * 60.0
    return elapsed_time

# ==========================================
# 2. 主逻辑：遗传算法匹配求解器
# ==========================================

def run_ga_matching(nodes_file, speeds_file):
    print(f"正在读取数据: {nodes_file} ...")

    # 读取数据
    try:
        nodes_df = pd.read_csv(nodes_file)
    except:
        nodes_df = pd.read_excel(nodes_file)
    speeds_df = pd.read_csv(speeds_file)

    # 1. 分离用户和充电站
    # 根据之前步骤，用户类型为 'd' (D0 + New...), 充电站类型为 'f'
    users = nodes_df[nodes_df['Type'] == 'd'].reset_index(drop=True)
    stations = nodes_df[nodes_df['Type'] == 'f'].reset_index(drop=True)

    num_users = len(users)
    num_stations = len(stations)
    print(f"检测到 {num_users} 个用户和 {num_stations} 个充电站")

    # 确保数量一致以便进行 1对1 匹配
    if num_users != num_stations:
        print("注意：用户和站点数量不一致，截断至较小值以进行匹配。")
        limit = min(num_users, num_stations)
        users = users.iloc[:limit]
        stations = stations.iloc[:limit]
        num_entities = limit
    else:
        num_entities = num_users

    # 2. 预计算每个充电站的等待时间 (Wait Time)
    # 逻辑：距离 Depot (D0) 越近的站越拥堵
    depot_pos = nodes_df[nodes_df['StringID'] == 'D0']
    if not depot_pos.empty:
        dx, dy = depot_pos.iloc[0]['x'], depot_pos.iloc[0]['y']
    else:
        dx, dy = 40, 50 # 默认值

    stations['dist_to_depot'] = np.sqrt((stations['x'] - dx)**2 + (stations['y'] - dy)**2)
    stations_sorted = stations.sort_values('dist_to_depot')

    # 定义4组参数 (桩数N, 到达率lambda)
    groups = [
        {'N': 8, 'lam': 15.0},
        {'N': 6, 'lam': 10.0},
        {'N': 4, 'lam': 5.0},
        {'N': 3, 'lam': 2.0}
    ]
    mu = 2.0 # 服务率

    station_wait_times = np.zeros(num_entities) # 存储每个站点的等待时间 (min)
    group_size = math.ceil(num_entities / 4)

    for i, (idx, row) in enumerate(stations_sorted.iterrows()):
        g_idx = min(i // group_size, 3)
        params = groups[g_idx]
        lam = params['lam'] + (group_size - (i % group_size)) * 0.1 # 微调 lambda
        N = params['N']

        wq_hours = mmn_analytical(lam, mu, N)
        station_wait_times[idx] = wq_hours * 60.0

    # 3. 构建成本矩阵 (Cost Matrix)
    # Cost[i][j] = User i 去 Station j 的总耗时
    print("正在构建成本矩阵...")
    cost_matrix = np.zeros((num_entities, num_entities))

    for u_idx, user in users.iterrows():
        for s_idx, station in stations.iterrows():
            # 计算距离
            dist = np.sqrt((user['x'] - station['x'])**2 + (user['y'] - station['y'])**2)

            # 获取对应的车速配置 (假设 Station S1 -> Road_1)
            try:
                st_str = str(station['StringID'])
                if st_str.startswith('S'):
                    r_num = int(st_str[1:])
                else:
                    r_num = s_idx + 1
                col_name = f'Road_{r_num}'

                if col_name in speeds_df.columns:
                    speed_profile = speeds_df[col_name].values
                else:
                    speed_profile = speeds_df.iloc[:, s_idx % 20].values
            except:
                speed_profile = speeds_df.iloc[:, 0].values

            # 计算行驶时间
            tt = calculate_travel_time_segment(dist, speed_profile)
            # 获取等待时间
            wt = station_wait_times[s_idx]

            cost_matrix[u_idx, s_idx] = tt + wt

    # 4. 遗传算法 (GA) 求解二分图匹配
    print("启动遗传算法搜索最佳匹配...")

    POP_SIZE = 100
    GENERATIONS = 150
    MUTATION_RATE = 0.15

    # 个体编码：长度为N的列表，第 i 位的值 j 代表 User i -> Station j
    def create_individual():
        ind = list(range(num_entities))
        random.shuffle(ind)
        return ind

    def calculate_fitness(ind):
        total_cost = 0
        for u, s in enumerate(ind):
            total_cost += cost_matrix[u, s]
        return total_cost

    # 种群初始化
    population = [create_individual() for _ in range(POP_SIZE)]
    best_loss_history = []
    global_best_ind = None
    global_best_cost = float('inf')

    for gen in range(GENERATIONS):
        # 评估
        costs = [(ind, calculate_fitness(ind)) for ind in population]
        costs.sort(key=lambda x: x[1]) # 按成本升序排列

        best_gen_cost = costs[0][1]
        best_loss_history.append(best_gen_cost)

        if best_gen_cost < global_best_cost:
            global_best_cost = best_gen_cost
            global_best_ind = costs[0][0][:]

        # 选择 (保留前 50%)
        selected = [x[0] for x in costs[:POP_SIZE//2]]

        next_pop = []
        next_pop.append(global_best_ind[:]) # 精英保留

        # 交叉变异生成新后代
        while len(next_pop) < POP_SIZE:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)

            # 顺序交叉 (Order Crossover)
            start, end = sorted(random.sample(range(num_entities), 2))
            child = [-1] * num_entities
            child[start:end] = parent1[start:end]

            fill_idx = 0
            for gene in parent2:
                if gene not in child:
                    while child[fill_idx] != -1:
                        fill_idx += 1
                    child[fill_idx] = gene

            # 交换变异
            if random.random() < MUTATION_RATE:
                i1, i2 = random.sample(range(num_entities), 2)
                child[i1], child[i2] = child[i2], child[i1]

            next_pop.append(child)

        population = next_pop

    print(f"✅ 优化完成! 最低总成本: {global_best_cost:.2f} 分钟")

    # 5. 绘图与保存
    # ------------------
    # 图 1: 损失下降曲线
    # ------------------
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(best_loss_history, color='blue', linewidth=2)
    plt.title('遗传算法优化过程：总成本(Loss)下降曲线')
    plt.xlabel('迭代代数 (Generation)')
    plt.ylabel('所有用户总成本 (Total Cost)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('ga_loss_curve.png')
    print("图表已保存: ga_loss_curve.png")

    # ------------------
    # 图 2: 匹配结果图
    # ------------------
    plt.figure(figsize=(10, 8))
    # 绘制点
    plt.scatter(stations['x'], stations['y'], c='green', marker='^', s=100, label='充电站', zorder=5)
    plt.scatter(users['x'], users['y'], c='blue', marker='o', s=80, label='用户', zorder=5)

    # 绘制连线
    for u_idx, s_idx in enumerate(global_best_ind):
        u_pt = users.iloc[u_idx]
        s_pt = stations.iloc[s_idx]
        plt.plot([u_pt['x'], s_pt['x']], [u_pt['y'], s_pt['y']], 'k--', alpha=0.3)

    plt.title(f'最佳匹配方案 (总成本: {global_best_cost:.1f} min)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig('ga_matching_plot.png')
    print("图表已保存: ga_matching_plot.png")

    # 保存 CSV
    results = []
    for u, s in enumerate(global_best_ind):
        results.append({
            'User': users.iloc[u]['StringID'],
            'Assigned_Station': stations.iloc[s]['StringID'],
            'Cost': cost_matrix[u, s]
        })
    pd.DataFrame(results).to_csv('ga_final_result.csv', index=False)
    print("数据已保存: ga_final_result.csv")

if __name__ == "__main__":
    # 请确保文件在同一目录下
    node_file = 'c101_21_added_19_station_type.csv'
    speed_file = 'reshaped_road_speeds.csv'

    run_ga_matching(node_file, speed_file)
