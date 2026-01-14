import math
import numpy as np
import time  # 用于统计仿真耗时

# ===================== 1. 解析解（严格M/M/N公式） =====================
def mmn_analytical(lamda, mu, N):
    """
    严格计算M/M/N排队模型所有核心指标（时间单位：小时）
    公式参考：M/M/N排队论经典公式（Erlang C公式）
    :param lamda: 顾客到达率（辆/小时）
    :param mu: 单个服务台服务率（辆/小时，1/平均服务时间）
    :param N: 服务台数量（充电桩数）
    :return: 核心指标字典
    """
    rho = lamda / mu  # 总服务强度（所有服务台的平均利用率）
    if rho >= N:
        raise ValueError(f"服务强度ρ={rho:.2f} ≥ 充电桩数N={N}，系统不稳定（队列无限长）！")

    # 步骤1：计算空闲概率P0（系统中无顾客的概率）
    sum_k0_to_Nminus1 = sum([(rho**k)/math.factorial(k) for k in range(N)])
    term_N = (rho**N) / (math.factorial(N) * (1 - rho/N))
    P0 = 1 / (sum_k0_to_Nminus1 + term_N)

    # 步骤2：计算关键指标（Erlang C公式）
    P_wait = (rho**N) / (math.factorial(N) * (1 - rho/N)) * P0  # 顾客需要排队的概率
    Lq = P_wait * (rho / (N - rho))  # 平均排队长度（仅队列中的顾客数）
    Ls = Lq + rho  # 系统内平均顾客数（排队+正在服务）
    Wq = Lq / lamda  # 平均排队时间（小时，仅等待时间）
    Ws = Wq + 1/mu  # 系统内平均逗留时间（等待+服务，小时）
    utilization = rho / N  # 充电桩平均利用率

    return {
        "空闲概率P0": P0,
        "顾客排队概率": P_wait,
        "平均排队长度Lq": Lq,
        "系统内平均顾客数Ls": Ls,
        "平均排队时间Wq": Wq,
        "系统内平均逗留时间Ws": Ws,
        "充电桩平均利用率": utilization
    }

# ===================== 2. 仿真解（严谨版M/M/N） =====================
def mmn_simulation(lamda, mu, N, num_customers, seed=42):
    """
    严谨的M/M/N蒙特卡洛仿真（FCFS先到先服务）
    :param lamda: 到达率（辆/小时）
    :param mu: 服务率（辆/小时）
    :param N: 充电桩数
    :param num_customers: 模拟服务的顾客总数（对应你说的“训练轮数”）
    :param seed: 随机种子（保证结果可复现）
    :return: 仿真指标字典 + 耗时
    """
    np.random.seed(seed)  # 固定随机种子，结果可复现
    start_time = time.time()  # 记录开始时间

    # 初始化变量
    arrival_times = []  # 每个顾客的到达时间（小时）
    service_times = []  # 每个顾客的服务时间（小时）
    start_service_times = []  # 每个顾客开始服务的时间（小时）
    completion_times = []  # 每个顾客完成服务的时间（小时）

    # 生成所有顾客的到达时间（指数分布）
    inter_arrival_times = np.random.exponential(scale=1/lamda, size=num_customers)
    arrival_times.append(inter_arrival_times[0])  # 第一个顾客到达时间
    for i in range(1, num_customers):
        arrival_times.append(arrival_times[i-1] + inter_arrival_times[i])

    # 生成所有顾客的服务时间（指数分布）
    service_times = np.random.exponential(scale=1/mu, size=num_customers)

    # 初始化充电桩状态：存储每个充电桩的完成服务时间（初始为0，空闲）
    charger_completion = [0.0] * N

    # 逐个处理每个顾客（FCFS）
    for i in range(num_customers):
        # 打印进度（每10%打印一次）
        if (i+1) % max(1, num_customers//10) == 0:
            progress = (i+1)/num_customers*100
            print(f"仿真进度: {progress:.0f}% ({i+1}/{num_customers})")

        # 找到最早空闲的充电桩
        earliest_free_charger = min(charger_completion)
        charger_idx = charger_completion.index(earliest_free_charger)

        # 计算开始服务时间：顾客到达时间 OR 充电桩空闲时间（取晚的那个）
        start_service = max(arrival_times[i], earliest_free_charger)
        start_service_times.append(start_service)

        # 计算完成服务时间
        completion = start_service + service_times[i]
        completion_times.append(completion)

        # 更新充电桩状态
        charger_completion[charger_idx] = completion

    # 计算核心指标
    waiting_times = np.array(start_service_times) - np.array(arrival_times)  # 每个顾客的排队时间
    total_waiting_time = np.sum(waiting_times)
    avg_waiting_time = np.mean(waiting_times)  # 平均排队时间（Wq）
    avg_service_time = np.mean(service_times)  # 平均服务时间
    avg_system_time = np.mean(np.array(completion_times) - np.array(arrival_times))  # 平均逗留时间（Ws）
    avg_queue_length = total_waiting_time / completion_times[-1]  # 平均排队长度（Lq，Little公式）
    utilization = (np.sum(service_times) / N) / completion_times[-1]  # 充电桩平均利用率
    P_wait = np.sum(waiting_times > 0) / num_customers  # 顾客需要排队的概率

    # 计算耗时
    simulation_time = time.time() - start_time

    return {
        "平均排队时间Wq": avg_waiting_time,
        "平均服务时间": avg_service_time,
        "系统内平均逗留时间Ws": avg_system_time,
        "平均排队长度Lq": avg_queue_length,
        "顾客排队概率": P_wait,
        "充电桩平均利用率": utilization
    }, simulation_time

# ===================== 3. 主程序（对比解析解+仿真解） =====================
if __name__ == "__main__":
    # ---------- 核心参数（可自定义） ----------
    lambda_arrival = 30    # 车辆到达率：2辆/小时
    mu_service = 2        # 单个桩服务率：1辆/小时（平均1小时充1辆车）
    num_chargers = 20      # 充电桩数量
    num_customers = 10000  # 模拟服务的顾客数（改大这个数会变慢）

    # ---------- 运行解析解 ----------
    print("===== 解析解（M/M/N理论值） =====")
    analytical_results = mmn_analytical(lambda_arrival, mu_service, num_chargers)
    for key, value in analytical_results.items():
        print(f"{key}: {value:.4f}")

    # ---------- 运行仿真解 ----------
    print("\n===== 仿真解（蒙特卡洛模拟） =====")
    simulation_results, sim_time = mmn_simulation(
        lambda_arrival, mu_service, num_chargers, num_customers, seed=42
    )
    for key, value in simulation_results.items():
        print(f"{key}: {value:.4f}")
    print(f"\n仿真总耗时: {sim_time:.2f} 秒（服务{num_customers}个顾客）")

    # ---------- 对比关键指标 ----------
    print("\n===== 解析解 vs 仿真解 对比（平均排队时间Wq） =====")
    print(f"解析解Wq: {analytical_results['平均排队时间Wq']:.4f} 小时")
    print(f"仿真解Wq: {simulation_results['平均排队时间Wq']:.4f} 小时")
    print(f"误差: {abs(analytical_results['平均排队时间Wq'] - simulation_results['平均排队时间Wq']):.4f} 小时")
