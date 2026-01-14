import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def select_optimal_station(file_path, w1=0.5, w2=0.5, output_file='c101_21_optimization_result.csv'):
    """
    多目标求解最优充电站，并将综合得分保存为 total_cost 列
    :param file_path: 数据文件路径
    :param w1: 时间最小化权重
    :param w2: 负载均衡权重
    :param output_file: 结果保存路径
    """
    # 1. 读取数据
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_excel(file_path)

    # 2. 筛选充电站
    stations = df[df['Type'] == 'f'].copy()
    depot = df[df['Type'] == 'd']

    # 3. 计算目标函数
    # 目标1：综合所需时间
    stations['total_time'] = stations['travel_cost'] + stations['wait_time_min']

    # 目标2：负载均衡 (load_std 已在文件中)

    # 4. 数据归一化 (Min-Max Scaling)
    t_min, t_max = stations['total_time'].min(), stations['total_time'].max()
    std_min, std_max = stations['load_std'].min(), stations['load_std'].max()

    # 防止除以零
    if t_max - t_min != 0:
        stations['norm_time'] = (stations['total_time'] - t_min) / (t_max - t_min)
    else:
        stations['norm_time'] = 0

    if std_max - std_min != 0:
        stations['norm_std'] = (stations['load_std'] - std_min) / (std_max - std_min)
    else:
        stations['norm_std'] = 0

    # 5. 计算综合评分 (Total Cost)
    # 修改点：将结果列命名为 total_cost
    stations['total_cost'] = w1 * stations['norm_time'] + w2 * stations['norm_std']

    # --- 关键步骤：将 total_cost 保存回原始 DataFrame ---
    df.loc[stations.index, 'total_cost'] = stations['total_cost']
    df['total_cost'] = df['total_cost'].fillna(0) # 非充电站填充 0

    # 保存文件
    df.to_csv(output_file, index=False)
    print(f"✅ 包含 total_cost 的结果已保存至: {output_file}")

    # 6. 选择最优站
    best_station = stations.loc[stations['total_cost'].idxmin()]

    # --- 打印结果 ---
    print("="*30)
    print(f"【多目标优化结果】 (权重: 时间={w1}, 均衡={w2})")
    print(f"🏆 推荐充电站: {best_station['StringID']}")
    print(f"⏱️ 预计总耗时: {best_station['total_time']:.2f} 分钟")
    print(f"   (行驶: {best_station['travel_cost']:.2f} + 排队: {best_station['wait_time_min']:.2f})")
    print(f"⚖️ 预估系统不均衡度(Std): {best_station['load_std']:.4f}")
    print(f"📊 综合评分 (Total Cost): {best_station['total_cost']:.4f}")
    print("="*30)

    # 7. 绘图展示
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.scatter(stations['x'], stations['y'], c='gray', alpha=0.6, s=60, label='其他充电站')

    if not depot.empty:
        plt.scatter(depot['x'], depot['y'], c='black', marker='s', s=100, label='车辆当前位置/仓库')

    plt.scatter(best_station['x'], best_station['y'], c='red', s=250, marker='*', zorder=10,
                label=f'最优选择: {best_station["StringID"]}')

    label_text = (f"{best_station['StringID']}\n"
                  f"总时:{best_station['total_time']:.1f}m\n"
                  f"Cost:{best_station['total_cost']:.3f}")

    plt.annotate(label_text,
                 (best_station['x'], best_station['y']),
                 xytext=(15, 15), textcoords='offset points',
                 bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                 fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.title(f'多目标充电站推荐图 (时间权重:{w1}, 均衡权重:{w2})')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # plt.savefig('optimal_result.png')
    plt.show()

# --- 运行函数 ---
select_optimal_station('c101_21_load_std.csv', w1=0.5, w2=0.5)
