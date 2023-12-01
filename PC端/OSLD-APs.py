import random
import numpy as np
import pandas as pd
def calculate_stability(ap_data):
    # 计算每个 AP 的稳定度 STA
    mean_rss = np.mean(ap_data)
    variance = np.var(ap_data)
    stability = len(ap_data) / (variance + 1e-9)  # 加一个小常数以防止被零除
    return stability
def osld_aps_algorithm(ap_data, k, m, N_v, N_sh):
    # 步骤1:计算每个AP的稳定性
    ap_stabilities = {}
    for ap, rss_data in ap_data.items():
        stability = calculate_stability(rss_data)
        ap_stabilities[ap] = stability

    # 步骤2:选择稳定性最高的N_v个ap
    selected_aps = sorted(ap_stabilities.keys(), key=lambda ap: ap_stabilities[ap], reverse=True)[:N_v]
    print(selected_aps)
    #步骤3:初始化AP权重
    ap_weights = {ap: 1.0 for ap in selected_aps}

    # 步骤4:更新AP权重的主循环
    for i in range(m):
        # 随机选择一个样本点R
        R = random.choice(list(ap_data.values()))
        print(R)
        # 找出k个最近的正确邻居H
        k_nearest_correct_neighbors = sorted(selected_aps, key=lambda ap: np.abs(np.mean(ap_data[ap]) - np.mean(R)))[:k]
        print(k_nearest_correct_neighbors)
        # 找出k个最近的错误邻居M(C)
        k_nearest_wrong_neighbors = []
        for _ in range(k):
            # 随机选择一个不同类别的样本C
            random_C = random.choice([c for c in selected_aps if c not in k_nearest_correct_neighbors])
            k_nearest_wrong_neighbors.append(random_C)
        print(k_nearest_wrong_neighbors)
        # 更新每个AP的权重
        for ap in selected_aps:
            diff_H = np.abs(np.mean(ap_data[ap]) - np.mean(R))
            print(ap_data[ap])
            print(ap_data['AP6'])
            diff_M_C = np.mean(
                [np.abs(np.mean(ap_data[ap]) - np.mean(ap_data[C_ap])) for C_ap in k_nearest_wrong_neighbors])

            ap_weights[ap] = ap_weights[ap] - (diff_H / (i + 1) + diff_M_C / (i + 1))

    # 步骤5:选择权重最高的顶级N_sh个ap
    final_selected_aps = sorted(selected_aps, key=lambda ap: ap_weights[ap], reverse=True)[:N_sh]

    return final_selected_aps

if __name__ == "__main__":
    # 用你实际的RSS数据替换为:{: {AP1: [RSS values], AP2: [RSS values], ...}
    ap_data = pd.read_excel('24_2.xls', header=None)
    print('1')
    print(ap_data)
    ap_data_dict = {}
    for i in range(20):
        ap_name = f'AP{i + 1}'
        ap_data_dict[ap_name] = ap_data.iloc[i * 4: (i + 1) * 4, 0].tolist()
    print(ap_data_dict)
    k = 10 # 最近邻居数
    m = 1000 # 迭代次数
    N_v = 18  # 根据稳定性初始选择AP的数量
    N_sh = 11  # 根据权重最终选择AP的个数

    selected_aps = osld_aps_algorithm(ap_data_dict, k, m, N_v, N_sh)
    selected_ap_data = {ap: ap_data_dict[ap] for ap in selected_aps}
    print(selected_ap_data)
    selected_ap_df = pd.DataFrame(selected_ap_data)

    # 保存选定的AP的RSS强度为ap_option.csv，不保存列名
    selected_ap_df.to_csv('24_2_option.csv', index=False, header=False)