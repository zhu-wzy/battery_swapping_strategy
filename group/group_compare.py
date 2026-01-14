import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 基础物理模型 (排队论 + 行驶时间)
# ==========================================

def mmn_analytical(lamda, mu, N):
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

def load_and_prep_data(nodes_file, speeds_file):
    # 读取数据
    try:
        nodes_df = pd.read_csv(nodes_file)
    except:
        nodes_df = pd.read_excel(nodes_file)
    speeds_df = pd.read_csv(speeds_file)

    users = nodes_df[nodes_df['Type'] == 'd'].reset_index(drop=True)
    stations = nodes_df[nodes_df['Type'] == 'f'].reset_index(drop=True)
    num_users = len(users)
    num_stations = len(stations)

    # 1. 预计算站点参数
    depot_pos = nodes_df[nodes_df['StringID'] == 'D0']
    dx, dy = (depot_pos.iloc[0]['x'], depot_pos.iloc[0]['y']) if not depot_pos.empty else (40, 50)
    stations['dist_to_depot'] = np.sqrt((stations['x'] - dx)**2 + (stations['y'] - dy)**2)
    stations_sorted = stations.sort_values('dist_to_depot')

    groups = [{'N': 8, 'lam': 15.0}, {'N': 6, 'lam': 10.0},
              {'N': 4, 'lam': 5.0}, {'N': 3, 'lam': 2.0}]
    mu = 2.0

    station_base_wait = np.zeros(num_stations)
    station_marginal_cost = np.zeros(num_stations)
    group_size = math.ceil(num_stations / 4)

    for i, (idx, row) in enumerate(stations_sorted.iterrows()):
        g_idx = min(i // group_size, 3)
        params = groups[g_idx]
        lam = params['lam'] + (group_size - (i % group_size)) * 0.1
        N = params['N']
        wq_hours = mmn_analytical(lam, mu, N)
        station_base_wait[idx] = wq_hours * 60.0
        station_marginal_cost[idx] = 60.0 / (N * mu)

    # 2. 预计算行驶时间矩阵
    travel_time_matrix = np.zeros((num_users, num_stations))
    for u_idx, user in users.iterrows():
        for s_idx, station in stations.iterrows():
            dist = np.sqrt((user['x'] - station['x'])**2 + (user['y'] - station['y'])**2)
            speed_idx = s_idx % speeds_df.shape[1]
            speed_profile = speeds_df.iloc[:, speed_idx].values
            tt = calculate_travel_time_segment(dist, speed_profile)
            travel_time_matrix[u_idx, s_idx] = tt

    return users, stations, travel_time_matrix, station_base_wait, station_marginal_cost

# ==========================================
# 3. 评估函数 (双目标)
# ==========================================

def evaluate_solution(ind, travel_time_matrix, station_base_wait, station_marginal_cost):
    num_stations = station_base_wait.shape[0]
    counts = np.zeros(num_stations)
    for s in ind:
        counts[s] += 1

    # 动态排队时间 = 基础 + (拥挤 * 边际)
    dynamic_wait_times = station_base_wait + (counts * station_marginal_cost)

    # 目标 1: 总时间成本 (越小越好)
    total_time = 0
    for u_i, s_j in enumerate(ind):
        total_time += travel_time_matrix[u_i, s_j] + dynamic_wait_times[s_j]

    # 目标 2: 负载均衡 Std (越小越好)
    load_std = np.std(dynamic_wait_times)

    return total_time, load_std

# ==========================================
# 4. 原始加权 GA (Baseline) - 用于对比
# ==========================================

def run_weighted_ga(num_users, num_stations, travel_time_matrix, station_base_wait, station_marginal_cost):
    POP_SIZE = 100
    GENERATIONS = 500
    WEIGHT_STD = 15.0 # 使用之前设定的固定权重

    def create_ind():
        return [random.randint(0, num_stations - 1) for _ in range(num_users)]

    def get_fitness(ind):
        t, s = evaluate_solution(ind, travel_time_matrix, station_base_wait, station_marginal_cost)
        # 加权和目标函数
        return t + WEIGHT_STD * s, t, s

    pop = [create_ind() for _ in range(POP_SIZE)]
    best_res = (None, float('inf'), 0, 0) # ind, score, time, std

    for _ in range(GENERATIONS):
        scored_pop = []
        for ind in pop:
            score, t, s = get_fitness(ind)
            scored_pop.append((ind, score, t, s))

        scored_pop.sort(key=lambda x: x[1])
        if scored_pop[0][1] < best_res[1]:
            best_res = scored_pop[0]

        # 简单的进化操作
        selected = [x[0] for x in scored_pop[:POP_SIZE//2]]
        new_pop = []
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.sample(selected, 2)
            pt = random.randint(0, num_users-1)
            child = p1[:pt] + p2[pt:]
            if random.random() < 0.2:
                idx = random.randint(0, num_users-1)
                child[idx] = random.randint(0, num_stations-1)
            new_pop.append(child)
        pop = new_pop

    return best_res[2], best_res[3] # 返回 Time, Std

# ==========================================
# 5. NSGA-II 算法实现 (核心)
# ==========================================

# 支配关系判断
def dominates(obj1, obj2):
    # 最小化问题：obj1 支配 obj2 当且仅当 obj1 <= obj2 且至少有一个 <
    better_or_equal = (obj1[0] <= obj2[0]) and (obj1[1] <= obj2[1])
    strictly_better = (obj1[0] < obj2[0]) or (obj1[1] < obj2[1])
    return better_or_equal and strictly_better

# 快速非支配排序
def fast_non_dominated_sort(population_objs):
    fronts = [[]]
    domination_count = [0] * len(population_objs)
    dominated_solutions = [[] for _ in range(len(population_objs))]

    for p in range(len(population_objs)):
        for q in range(len(population_objs)):
            if dominates(population_objs[p], population_objs[q]):
                dominated_solutions[p].append(q)
            elif dominates(population_objs[q], population_objs[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]

# 拥挤距离计算 (Crowding Distance)
def crowding_distance_assignment(pop_indices, population_objs):
    l = len(pop_indices)
    distances = {i: 0 for i in pop_indices}
    if l == 0: return distances

    # 对每个目标分别计算
    for m in range(2):
        sorted_indices = sorted(pop_indices, key=lambda i: population_objs[i][m])

        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        obj_range = population_objs[sorted_indices[-1]][m] - population_objs[sorted_indices[0]][m]
        if obj_range == 0: obj_range = 1.0

        for k in range(1, l-1):
            distances[sorted_indices[k]] += (population_objs[sorted_indices[k+1]][m] - population_objs[sorted_indices[k-1]][m]) / obj_range

    return distances

def run_nsga_ii(num_users, num_stations, travel_time_matrix, station_base_wait, station_marginal_cost):
    POP_SIZE = 100
    GENERATIONS = 1000
    MUTATION_RATE = 0.2

    def create_ind():
        return [random.randint(0, num_stations - 1) for _ in range(num_users)]

    population = [create_ind() for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        # 评估当前种群
        pop_objs = [evaluate_solution(ind, travel_time_matrix, station_base_wait, station_marginal_cost) for ind in population]

        # 生成子代
        offspring = []
        while len(offspring) < POP_SIZE:
            p1, p2 = random.sample(population, 2)
            pt = random.randint(0, num_users-1)
            child = p1[:pt] + p2[pt:] # 单点交叉
            if random.random() < MUTATION_RATE: # 变异
                idx = random.randint(0, num_users-1)
                child[idx] = random.randint(0, num_stations-1)
            offspring.append(child)

        # 合并种群 (父代 + 子代)
        combined_pop = population + offspring
        combined_objs = pop_objs + [evaluate_solution(ind, travel_time_matrix, station_base_wait, station_marginal_cost) for ind in offspring]

        # 非支配排序分层
        fronts = fast_non_dominated_sort(combined_objs)

        # 选择下一代
        new_pop = []

        for front in fronts:
            if len(new_pop) + len(front) <= POP_SIZE:
                for idx in front:
                    new_pop.append(combined_pop[idx])
            else:
                # 最后一层根据拥挤距离筛选
                distances = crowding_distance_assignment(front, combined_objs)
                sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)

                num_needed = POP_SIZE - len(new_pop)
                for i in range(num_needed):
                    idx = sorted_front[i]
                    new_pop.append(combined_pop[idx])
                break

        population = new_pop

    # 提取最终的 Pareto 前沿解
    final_objs = [evaluate_solution(ind, travel_time_matrix, station_base_wait, station_marginal_cost) for ind in population]
    fronts = fast_non_dominated_sort(final_objs)
    pareto_front = [final_objs[i] for i in fronts[0]]

    # 排序以便绘图好看
    pareto_front.sort(key=lambda x: x[0])

    return pareto_front

# ==========================================
# 6. 主程序
# ==========================================

def main_comparison(nodes_file, speeds_file):
    # 1. 准备数据
    users, stations, tt_matrix, st_base, st_marg = load_and_prep_data(nodes_file, speeds_file)
    num_users = len(users)
    num_stations = len(stations)

    # 2. 运行 Baseline (Weighted GA)
    print(f"正在运行原始加权 GA (Baseline)...")
    base_time, base_std = run_weighted_ga(num_users, num_stations, tt_matrix, st_base, st_marg)
    print(f"原始结果: Time={base_time:.2f}, Std={base_std:.2f}")

    # 3. 运行 NSGA-II
    print(f"正在运行改进算法 NSGA-II...")
    pareto_front = run_nsga_ii(num_users, num_stations, tt_matrix, st_base, st_marg)
    print(f"NSGA-II 找到 {len(pareto_front)} 个帕累托最优解。")

    # 4. 绘图对比
    pareto_times = [sol[0] for sol in pareto_front]
    pareto_stds = [sol[1] for sol in pareto_front]

    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.scatter(pareto_times, pareto_stds, c='red', label='NSGA-II 帕累托前沿', marker='o', alpha=0.7)
    plt.scatter([base_time], [base_std], c='blue', label='原始加权 GA (Baseline)', marker='*', s=250, zorder=10)

    plt.title('改进算法(NSGA-II) 与 原算法(Weighted GA) 对比')
    plt.xlabel('时间总成本 (Total Time Cost / min)')
    plt.ylabel('负载均衡度 (Load Std)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存与显示
    plt.savefig('nsga2_vs_ga_comparison.png')
    plt.show()

    # 保存数据
    df_pareto = pd.DataFrame(pareto_front, columns=['Total_Time', 'Load_Std'])
    df_pareto.to_csv('nsga2_pareto_solutions.csv', index=False)
    print("帕累托解集已保存至: nsga2_pareto_solutions.csv")

if __name__ == "__main__":
    node_file = 'c101_21_added_19_station_type.csv'
    speed_file = 'reshaped_road_speeds.csv'

    if os.path.exists(node_file) and os.path.exists(speed_file):
        main_comparison(node_file, speed_file)
    else:
        print("❌ 错误：找不到输入文件。")
