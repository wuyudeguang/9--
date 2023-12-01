import subprocess
import socket
import random
import numpy as np
import pandas as pd
from keras.models import load_model
import tensorflow as tf

def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))
model = load_model('lstm_Model_5_fold_1.h5', custom_objects={'rmse': rmse})
# 获取WiFi强度和SSID数据
def get_wifi_info():
    cmd = "sudo iwlist wlan0 scan | grep 'Signal\|ESSID'"
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode('utf-8').split('\n')
    wifi_info = []
    desired_ssids = ['BUPT-portal', 'BUPT-mobile', 'BUPT-iot', 'eduroam']
    for i in range(len(lines)-1):
        if 'Signal level=' in lines[i]:
            strength = lines[i].split('Signal level=')[1].split(' ')[0]
            ssid = lines[i+1].split('ESSID:')[1].replace('"', '')
            if ssid in desired_ssids:
                wifi_info.append((ssid, strength))
    return wifi_info
# 创建socket连接
def create_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = '10.129.196.196'
    port = 12345
    s.connect((host, port))
    return s

def osld_aps_algorithm(ap_data, k, m, N_sh):
 #步骤3:初始化AP权重
    selected_aps =ap_data
    ap_weights = {ap: 1.0 for ap in selected_aps}

    # 步骤4:更新AP权重的主循环
    for i in range(m):
        # 随机选择一个样本点R
        R = random.choice(list(ap_data.values()))
        #print(R)
        # 找出k个最近的正确邻居H
        k_nearest_correct_neighbors = sorted(selected_aps, key=lambda ap: np.abs(np.mean(ap_data[ap]) - np.mean(R)))[:k]
        #print(k_nearest_correct_neighbors)
        # 找出k个最近的错误邻居M(C)
        k_nearest_wrong_neighbors = []
        for _ in range(k):
            # 随机选择一个不同类别的样本C
            random_C = random.choice([c for c in selected_aps if c not in k_nearest_correct_neighbors])
            k_nearest_wrong_neighbors.append(random_C)
        #print(k_nearest_wrong_neighbors)
        # 更新每个AP的权重
        for ap in selected_aps:
            diff_H = np.abs(np.mean(ap_data[ap]) - np.mean(R))
            diff_M_C = np.mean(
                [np.abs(np.mean(ap_data[ap]) - np.mean(ap_data[C_ap])) for C_ap in k_nearest_wrong_neighbors])

            ap_weights[ap] = ap_weights[ap] - (diff_H / (i + 1) + diff_M_C / (i + 1))

    # 步骤5:选择权重最高的顶级N_sh个ap
    final_selected_aps = sorted(selected_aps, key=lambda ap: ap_weights[ap], reverse=True)[:N_sh]

    return final_selected_aps

# 主函数
def main():

 while True:


    k = 5 # 最近邻居数
    m = 400 # 迭代次数
    N_sh = 11  # 根据权重最终选择AP的个数
    wifi_data1 = []
    wifi_data2 = []
    # 获取WiFi强度和SSID数据
    while len(wifi_data1) < 11:

      wifi_info = get_wifi_info()
      ap_data = {'AP' + str(i+1): [int(rssi)] for i, (ssid, rssi) in enumerate(wifi_info)}
      #print(ap_data)

      selected_aps = osld_aps_algorithm(ap_data, k, m, N_sh)
      selected_ap_data = {ap: ap_data[ap] for ap in selected_aps}
      #print("Selected APs:", selected_aps)
      #print(selected_ap_data)

      wifi_data1 = [value[0] for key, value in selected_ap_data.items()]
    wifi_data1 = np.asarray(wifi_data1)
    wifi_data1 = (wifi_data1 + 100) / 100
    print('wifi_data1:',wifi_data1)
    while len(wifi_data2) < 11:
      wifi_info = get_wifi_info()
      ap_data = {'AP' + str(i+1): [int(rssi)] for i, (ssid, rssi) in enumerate(wifi_info)}
      selected_aps = osld_aps_algorithm(ap_data, k, m, N_sh)
      selected_ap_data = {ap: ap_data[ap] for ap in selected_aps}
      wifi_data2 = [value[0] for key, value in selected_ap_data.items()]
    wifi_data2= np.asarray(wifi_data2)
    wifi_data2 = (wifi_data2 + 100) / 100
    print('wifi_data2:',wifi_data2)
    wifi_data=np.append(wifi_data1,wifi_data2)
    wifi_data = wifi_data.reshape(1, 2, 11)
    #print(wifi_data)
    predicted_location = model.predict(wifi_data)
    # 处理预测结果（示例：输出预测的位置坐标）
    #print('Predicted Location:', predicted_location)
    # 关闭连接
    #s.close()fi_info = get_wifi_info()
    print('Predicted Location:',predicted_location)
    client_socket.sendall(str(location).encode('utf-8'))
    #s.close()

if __name__ == '__main__':

    # 创建一个TCP/IP套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 获取本地主机名和端口号
    host = '10.129.219.238'
    port = 12345  # 端口号可以任意选择，但在客户端和服务器端必须一致

    # 绑定套接字到地址和端口号
    server_socket.bind((host, port))

    # 开始监听传入连接
    server_socket.listen(1)

    print('等待客户端连接...')
    client_socket, client_address = server_socket.accept()
    print('连接来自:', client_address)
    main()
    client_socket.close()
    server_socket.close()