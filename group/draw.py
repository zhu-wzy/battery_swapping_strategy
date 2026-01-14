import graphviz

def draw_nsga2_principle():
    dot = graphviz.Digraph('NSGA-II Principle', comment='NSGA-II Flowchart')
    dot.attr(rankdir='TB', size='10,10', dpi='300')

    # 定义节点样式
    dot.attr('node', shape='box', style='filled', fontname='Helvetica')

    # 1. 初始输入
    with dot.subgraph(name='cluster_input') as c:
        c.attr(style='dashed', label='Step 1: 种群混合', color='grey')
        c.node('Pt', '父代种群 Pt\n(大小 N)', fillcolor='lightblue')
        c.node('Qt', '子代种群 Qt\n(大小 N)', fillcolor='lightyellow')
        c.node('Rt', '混合种群 Rt = Pt + Qt\n(大小 2N)', fillcolor='lightgrey')

    # 2. 排序过程
    with dot.subgraph(name='cluster_sort') as c:
        c.attr(style='solid', label='Step 2: 快速非支配排序 (Fast Non-dominated Sorting)', color='black')
        c.node('Sort', '计算支配关系 & 分层', shape='diamond', fillcolor='orange')

        # 前沿层级
        c.node('F1', 'Front 1 (F1)\n最优层', fillcolor='lightgreen')
        c.node('F2', 'Front 2 (F2)\n次优层', fillcolor='lightgreen')
        c.node('F3', 'Front 3 (F3)\n...', fillcolor='lightgreen')
        c.node('Fi', 'Front i\n(临界层)', fillcolor='gold')

    # 3. 筛选过程
    with dot.subgraph(name='cluster_select') as c:
        c.attr(style='dashed', label='Step 3: 精英筛选与截断', color='grey')
        c.node('Crowding', '拥挤距离计算 (Crowding Distance)\n在临界层 Fi 内部排序', shape='hexagon', fillcolor='violet')
        c.node('NextGen', '新一代种群 Pt+1\n(大小 N)', fillcolor='lightblue')

    # 连接关系
    dot.edge('Pt', 'Rt')
    dot.edge('Qt', 'Rt')
    dot.edge('Rt', 'Sort')

    dot.edge('Sort', 'F1', label='Rank 1')
    dot.edge('Sort', 'F2', label='Rank 2')
    dot.edge('Sort', 'F3', label='Rank 3')
    dot.edge('Sort', 'Fi', label='Rank i')

    # 填充逻辑
    dot.edge('F1', 'NextGen', label='全部放入 (若不满 N)')
    dot.edge('F2', 'NextGen', label='全部放入')
    dot.edge('F3', 'NextGen', label='...')

    dot.edge('Fi', 'Crowding', label='容量不足以放入整层')
    dot.edge('Crowding', 'NextGen', label='按拥挤度由大到小\n填满剩余名额')

    return dot

# 生成图片
chart = draw_nsga2_principle()
chart.render('nsga2_principle_diagram', format='png', cleanup=True)
print("NSGA-II 原理图已生成。")
