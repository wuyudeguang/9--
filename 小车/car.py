# coding:utf-8
# 加入摄像头模块，让小车实现自动循迹行驶
# 思路为：摄像头读取图像，进行二值化，将黑色的赛道凸显出来
# 选择下方的一行像素，黑色为0，白色为255
# 找到黑色值的中点
# 目标中点与标准中点（320）进行比较得出偏移量
# 根据偏移量来控制小车左右轮的转速
# 考虑了偏移过多失控->停止;偏移量在一定范围内->高速直行(这样会速度不稳定，已删)

import cv2
import numpy as np
import Adafruit_PCA9685
import time
import socket

# 设置接收端的IP地址和端口号
host = "10.129.196.196"
port = 12345
# 创建socket对象
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定IP地址和端口号
s.bind((host, port))

# 监听连接
s.listen(1)

# 等待连接
print("等待连接...")
conn, addr = s.accept()
print("已连接:", addr)
# 接收数据
data = conn.recv(1024).decode()
print("接收到的数据:", data)

# 关闭连接
conn.close()
############小车运动初始化#############

# 使用默认地址（0x40）初始化PCA9685。
pwm = Adafruit_PCA9685.PCA9685()

# 将频率设置为 60hz
pwm.set_pwm_freq(60)

print('Moving All servo motors one at a time, press Ctrl-C to quit...')


############摄像头初始化#############

# 打开摄像头，图像尺寸640*480（长*高），opencv存储值为480*640（行*列）
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#设置摄像头帧率
cap.set(3,320)
cap.set(4,240)

# center定义
center = 160


############绿色卡片检测初始化初始化#############

# 定义绿色颜色阈值
green_lower = np.array([35, 43, 46])
green_upper = np.array([85, 255, 255])

change = 0
previous_cnts = 0
now_cnts = 0
move=388
servo=367
green_number=4
change_number=2

#绿色卡片检测函数
def ChestGreen():
    # 将帧从BGR颜色空间转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 创建绿色掩模
    mask = cv2.inRange(hsv, green_lower, green_upper)
    # 执行腐蚀和膨胀操作，以去除噪声
    mask = cv2.erode(mask, None, iterations=2)
    # 高斯模糊处理,消除高斯噪声
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask

frame_count=0
#start_time=time.time()

############工作程序#############
while True:
    ret, frame = cap.read()

    if not ret:
        print("无法捕获图像")
        break

    if frame is not None:
        frame_count += 1

############循迹黑线#############
        # 转化为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 大津法二值化
        retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        # 膨胀，白区域变大
        dst = cv2.dilate(dst, None, iterations=2)
        # # 腐蚀，白区域变小
        # dst = cv2.erode(dst, None, iterations=6)
        # 单看第400行的像素值
        color = dst[200]
        # 找到黑色的像素点个数
        black_count = np.sum(color == 0)
        # 找到黑色的像素点索引
        black_index = np.where(color == 0)
        # 防止black_count=0的报错
        if black_count <= 1:
            black_count = 1
        # 找到黑色像素的中心点位置
        center = (black_index[0][black_count - 1] + black_index[0][0]) / 2
        # 计算出center与标准中心点的偏移量
        direction = 160 - center
        #print(direction)                 ##########打印偏移量##############

############识别绿色#############
        # 高斯模糊处理,消除高斯噪声
        frame2 = cv2.GaussianBlur(frame,(5,5),0)
        #指示图像中的绿色物体位置（mask绿 res白）
        mask = ChestGreen()
        res = cv2.bitwise_and(frame2,frame2,mask=mask)
        # 查找绿色物体的轮廓
        cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        #小车停止设置conts
        if len(cnts)>=green_number: ###这里改识别到的像素个数控制
            now_cnts = 1
        else:
            now_cnts = 0
        if now_cnts == 1 and previous_cnts == 0:
            change = change +1
            print(change)                  ##########打印第几次识别到绿色卡片#######
            #pwm.set_pwm(0,0,200)
        previous_cnts = now_cnts
        # 显示图像
        cv2.imshow("mask",mask)
        cv2.imshow("res",res)

        '''
        # 计算帧率
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:  # 每秒钟计算一次帧率
            frame_rate = frame_count / elapsed_time
            #print(f"帧率: {frame_rate:.2f}")
            frame_count = 0
            start_time = time.time()
        '''
############小车运动#############
        '''
        if abs(direction) > 500:
        '''
        if change==change_number:
            pwm.set_pwm(0,0,300)
        elif change<change_number:
            # 前进
            pwm.set_pwm(0,0,move)
            #print("成功")
            #pwm.set_pwm(1,0,servo)

            # 右转
            if direction < 0:
                if direction < -100:
                    direction = -100
                pwm.set_pwm(1,0,servo + int(direction/2))

            # 左转
            elif direction >= 0:
                # 限制在70以内
                if direction > 100:
                    direction = 100
                pwm.set_pwm(1,0,servo + int(direction/2))


        #摄像头展示窗口
        cv2.imshow('Black Line Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()