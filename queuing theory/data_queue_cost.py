import pandas as pd
import math

# ===================== M/M/N 排队模型公式 =====================
def mmn_analytical(lamda, mu, N):
    rho = lamda / mu  # 服务强度
    if rho >= N:
        # 如果超负荷，返回一个标记值
        return {"Wq": 999.0}

    # 计算 P0 (空闲概率)
    sum_k = sum([(rho**k)/math.factorial(k) for k in range(N)])
    term_N = (rho**N) / (math.factorial(N) * (1 - rho/N))
    P0 = 1 / (sum_k + term_N)

    # 计算排队时间 Wq
    P_wait = (rho**N) / (math.factorial(N) * (1 - rho/N)) * P0
    Lq = P_wait * (rho / (N - rho))
    Wq = Lq / lamda

    return {"Wq": Wq}

def generate_deterministic_queues(input_file, output_file):
    # 1. 读取数据
    try:
        df_nodes = pd.read_csv(input_file)
    except:
        df_nodes = pd.read_excel(input_file.replace('.csv', '.xlsx'), sheet_name='Nodes')

    # 2. 筛选并排序
    # 我们只处理充电站 (Type 'f')，并按距离从小到大排序
    df_stations = df_nodes[df_nodes['Type'] == 'f'].copy()
    df_stations = df_stations.sort_values(by='distance')

    # 3. 定义确定性的参数组 (4组，每组5个站)
    # 格式: {'N': 桩数, 'base_lambda': 基础到达率}
    # 服务率 mu 固定为 2 (30分钟充一辆)
    groups = [
        {'N': 8, 'base_lambda': 15.0},  # 第一组：近距离，很堵
        {'N': 6, 'base_lambda': 10.0},  # 第二组：中距离，较堵
        {'N': 4, 'base_lambda': 5.0},   # 第三组：远距离，还行
        {'N': 3, 'base_lambda': 2.0}    # 第四组：超远，很空
    ]

    mu = 2.0
    num_stations = len(df_stations) # 应该是20
    group_size = 5

    # 4. 逐个赋值
    for i, (idx, row) in enumerate(df_stations.iterrows()):
        # 确定属于哪一组 (0, 1, 2, 3)
        group_idx = min(i // group_size, len(groups) - 1)
        params = groups[group_idx]

        N = params['N']

        # 组内微调：让排在前面的(更近的) lambda 稍微大一点
        # 例如第一组 i=0~4:
        # i=0 (最近) -> add 0.5
        # i=4 (最远) -> add 0.1
        pos_in_group = i % group_size
        lambda_adjustment = (group_size - pos_in_group) * 0.1
        lam = params['base_lambda'] + lambda_adjustment

        # 计算排队时间
        metrics = mmn_analytical(lam, mu, N)

        # 写入原表格
        df_nodes.loc[idx, 'num_chargers'] = int(N)
        df_nodes.loc[idx, 'arrival_rate'] = round(lam, 2)
        df_nodes.loc[idx, 'wait_time_min'] = round(metrics['Wq'] * 60, 2) # 转为分钟

    # 5. 填充非充电站的数据为0，避免空值
    df_nodes['num_chargers'] = df_nodes['num_chargers'].fillna(0)
    df_nodes['arrival_rate'] = df_nodes['arrival_rate'].fillna(0)
    df_nodes['wait_time_min'] = df_nodes['wait_time_min'].fillna(0)

    # 6. 保存
    df_nodes.to_csv(output_file, index=False)
    print(f"✅ 处理完成！确定性结果已保存至: {output_file}")

    # 打印预览
    print("\n--- 结果预览 (按距离排序) ---")
    print(df_nodes[df_nodes['Type']=='f'][['StringID', 'distance', 'num_chargers', 'wait_time_min']].sort_values('distance'))

if __name__ == "__main__":
    input_csv = 'c101_21_final.xlsx'
    output_csv = 'c101_21_queuing_cost.csv'
    generate_deterministic_queues(input_csv, output_csv)
