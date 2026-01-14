import pandas as pd
import re
import os  # 引入操作系统模块处理路径

def txt_to_excel(txt_filename, excel_filename):
    """
    将EVRP标准格式txt文件转换为Excel文件
    """

    # --- 核心修改：获取当前脚本所在的绝对路径 ---
    # 获取当前执行脚本所在的文件夹路径
    if '__file__' in globals():
        current_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        # 如果是在 Jupyter Notebook 中运行，使用当前工作目录
        current_dir = os.getcwd()

    # 拼接出完整的文件路径
    txt_path = os.path.join(current_dir, txt_filename)
    excel_path = os.path.join(current_dir, excel_filename)

    print(f"正在读取文件: {txt_path}")

    # 检查输入文件是否存在
    if not os.path.exists(txt_path):
        print(f"❌ 错误：在以下路径未找到文件：\n{txt_path}")
        print("请确保 c101_21.txt 确实和代码在同一个文件夹内。")
        return

    # --- 以下逻辑保持不变 ---
    data_rows = []
    parameters = []
    header = None

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('StringID') and header is None:
            header = line.split()
            continue
        if '/' in line:
            try:
                parts = line.split('/')
                param_name = parts[0].strip()
                param_value = parts[1].strip()
                parameters.append({'Parameter': param_name, 'Value': float(param_value)})
            except:
                pass
            continue
        parts = line.split()
        if header and len(parts) == len(header):
            if re.match(r'^[DSC]\d+', parts[0]):
                data_rows.append(parts)

    if header and data_rows:
        df_nodes = pd.DataFrame(data_rows, columns=header)
        numeric_cols = header[2:]
        for col in numeric_cols:
            df_nodes[col] = pd.to_numeric(df_nodes[col])
    else:
        df_nodes = pd.DataFrame()

    df_params = pd.DataFrame(parameters)

    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_nodes.to_excel(writer, sheet_name='Nodes', index=False)
            if not df_params.empty:
                df_params.to_excel(writer, sheet_name='Config', index=False)

        # --- 打印最终保存的绝对路径 ---
        print("-" * 30)
        print("✅ 转换成功！")
        print(f"文件已保存到绝对路径:\n{excel_path}")
        print("-" * 30)

    except Exception as e:
        print(f"❌ 保存 Excel 时出错: {e}")
        # 如果文件被占用（比如Excel开着），这里会报错

if __name__ == "__main__":
    # 只要确保你的 txt 文件和这个脚本放在一起，这里只写文件名即可
    input_file = 'c101_21.txt'
    output_file = 'c101_21.xlsx'

    txt_to_excel(input_file, output_file)
