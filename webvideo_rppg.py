# -*- coding: gbk -*-
from flask import Flask, render_template, Response


import cv2
import time
import mediapipe as mp
import numpy as np
from PIL import Image
from pyzbar import pyzbar
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from collections import deque

app = Flask(__name__)

# 存储rPPG信号的变量
rppg_signal = []
times = []
frame_count = 0 
start_time = time.time() 
fps = 0
count = 0
face_detection_interval = 30  # 每隔   帧进行一次人脸检测  
roi = None   
# 定义队列，用于存储心跳值，设置最大长度为30
heartbeat_queue = deque(maxlen=15*10)
# 定义中值队列，用于存储中值心跳值，设置最大长度为10
median_heartbeat_queue = deque(maxlen=30)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
 
    cap = cv2.VideoCapture(0)
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 截取视频帧的左半边
            height, width, _ = frame.shape  
            new_width = width // 2
            new_frame = frame[:, :new_width, :]
            
            # 在图像上绘制帧率信息
            #cv2.putText(new_frame, f"FPS: {calculate_fps()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   
            # 设置ROI区域
            global count,face_detection_interval  
            count += 1
            #print(count)
            if count % face_detection_interval == 0 :
              global roi
              roi = get_center_roi(new_frame)
              
           
            if roi is not None:
            
                # 应用rPPG算法估计心
                global heartbeat
                heartbeat = apply_rppg_algorithm(roi)
                # 获取心跳值并添加到队列中
                heartbeat_queue.append(heartbeat)
                #获取中值 
                median_heartbeat = np.median(heartbeat_queue)  
                # 将中值心跳值存入中值队列中
                median_heartbeat_queue.append(median_heartbeat)
                # 计算中值队列的平均值
                if len(median_heartbeat_queue) > 0:
                    average_median_heartbeat = sum(median_heartbeat_queue) / len(median_heartbeat_queue)
                # 在图像上绘制心率信息
                # 将字体颜色设置为黑色 (0, 0, 0)
                font_color = (0, 0, 0)
                # 在新的帧上添加文字
                cv2.putText(new_frame, f"Heart Rate: {(int)(heartbeat)} BPM", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
                cv2.putText(new_frame, f"Heart Avg : {(int)(average_median_heartbeat)} BPM", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
                cv2.putText(new_frame, f"FPS: {(int)(fps)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)

             
                    
            _, encoded_image = cv2.imencode('.jpg', new_frame)  

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_image.tobytes() + b'\r\n')
            
        except Exception as e:
            print(e)
            # 释放资源
            cap.release()  
  
            break

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
def get_center_roi(image):
    # 创建Haar级联分类器对象
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Haar级联分类器检测人脸
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # 在图像上绘制矩形框标记人脸区域
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 返回第一个检测到的人脸区域作为ROI
        return image[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]
    else:
        # 如果未检测到人脸，返回空值
        return None

def apply_rppg_algorithm(frame):
    heart_rate_estimate_bpm = 0;
    
    global frame_count,start_time
    frame_count += 1

    # 使用time.time()获取当前时间，并计算每秒处理的帧数（fps）
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time > 1.0:  # 每秒钟更新一次fps
        global fps
        fps = frame_count / elapsed_time
        #print("FPS:", fps)
        frame_count = 0
        start_time = current_time
    
    # 将帧转换为灰度图
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    # 计算帧中的平均强度
    mean_intensity = np.mean(gray_frame)  

    #print(mean_intensity) 
    # 将平均强度添加到rPPG信号中
    global rppg_signal
    # 128个采样点
    MAX_SIGNAL_LENGTH = 128
    rppg_signal.append(mean_intensity)  
 
    # 检查列表长度是否超过最大值
    if len(rppg_signal) > MAX_SIGNAL_LENGTH:     
      # 删除最早加入的值（列表的第一个元素）
      rppg_signal.pop(0)

      # 定义带通滤波器的参数
      low_freq = 1.0  # 最低频率
      high_freq = 3.0  # 最高频率
      if fps == 0:
        fps = 15
      fs = fps  # 采样率
      # 计算带通滤波器的系数
      nyquist_freq = 0.5 * fs  # 采样频率的一半
      low_cutoff = low_freq / nyquist_freq
      high_cutoff = high_freq / nyquist_freq
      # 检查截止频率是否在0和1之间
      if 0 < low_cutoff < 1 and 0 < high_cutoff < 1:
          # 设计带通滤波器
          b, a = butter(2, [low_cutoff, high_cutoff], btype='band')
          # 应用带通滤波器
          filtered_signal = filtfilt(b, a, rppg_signal)
          #print(filtered_signal)
          # 对rPPG信号执行FFT
          fft_signal = np.fft.fft(filtered_signal)
          #print((fft_signal)) 
          freqs = np.fft.fftfreq(len(filtered_signal), 1.0 / fps)
          #print((freqs)) 
          # 60到130每分钟的心跳频率范围对应的频率范围
          min_heart_rate = 60  # 最小心率（bpm）
          max_heart_rate = 130  # 最大心率（bpm）
          min_freq = min_heart_rate / 60.0  # 最小频率（每秒钟）
          max_freq = max_heart_rate / 60.0  # 最大频率（每秒钟）
          
          # 找到频率范围对应的索引
          min_freq_index = np.argmax(freqs >= min_freq)
          max_freq_index = np.argmax(freqs >= max_freq)
          
          # 在频率范围内找到幅度最大的频率分量的索引
          max_amplitude_index = np.argmax(np.abs(fft_signal[min_freq_index:max_freq_index]))
          
          # 获取对应的频率值（即心率估计值）
          heart_rate_estimate = freqs[min_freq_index:max_freq_index][max_amplitude_index]
          
          # 将心率估计值从频率转换为每分钟（bpm）
          heart_rate_estimate_bpm = heart_rate_estimate * 60.0
      #else:
          # 处理截止频率超出范围的情况
          #print("Error: Digital filter critical frequencies must be 0 < Wn < 1")

    
      #print("------------"+f"{heart_rate_estimate_bpm}")       
    return heart_rate_estimate_bpm  # 转换为每分钟的心跳数
    #return 80    

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
