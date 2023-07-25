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

# �洢rPPG�źŵı���
rppg_signal = []
times = []
frame_count = 0 
start_time = time.time() 
fps = 0
count = 0
face_detection_interval = 30  # ÿ��   ֡����һ���������  
roi = None   
# ������У����ڴ洢����ֵ��������󳤶�Ϊ30
heartbeat_queue = deque(maxlen=15*10)
# ������ֵ���У����ڴ洢��ֵ����ֵ��������󳤶�Ϊ10
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
            
            # ��ȡ��Ƶ֡������
            height, width, _ = frame.shape  
            new_width = width // 2
            new_frame = frame[:, :new_width, :]
            
            # ��ͼ���ϻ���֡����Ϣ
            #cv2.putText(new_frame, f"FPS: {calculate_fps()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   
            # ����ROI����
            global count,face_detection_interval  
            count += 1
            #print(count)
            if count % face_detection_interval == 0 :
              global roi
              roi = get_center_roi(new_frame)
              
           
            if roi is not None:
            
                # Ӧ��rPPG�㷨������
                global heartbeat
                heartbeat = apply_rppg_algorithm(roi)
                # ��ȡ����ֵ����ӵ�������
                heartbeat_queue.append(heartbeat)
                #��ȡ��ֵ 
                median_heartbeat = np.median(heartbeat_queue)  
                # ����ֵ����ֵ������ֵ������
                median_heartbeat_queue.append(median_heartbeat)
                # ������ֵ���е�ƽ��ֵ
                if len(median_heartbeat_queue) > 0:
                    average_median_heartbeat = sum(median_heartbeat_queue) / len(median_heartbeat_queue)
                # ��ͼ���ϻ���������Ϣ
                # ��������ɫ����Ϊ��ɫ (0, 0, 0)
                font_color = (0, 0, 0)
                # ���µ�֡���������
                cv2.putText(new_frame, f"Heart Rate: {(int)(heartbeat)} BPM", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
                cv2.putText(new_frame, f"Heart Avg : {(int)(average_median_heartbeat)} BPM", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
                cv2.putText(new_frame, f"FPS: {(int)(fps)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)

             
                    
            _, encoded_image = cv2.imencode('.jpg', new_frame)  

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_image.tobytes() + b'\r\n')
            
        except Exception as e:
            print(e)
            # �ͷ���Դ
            cap.release()  
  
            break

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
def get_center_roi(image):
    # ����Haar��������������
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # ��ͼ��ת��Ϊ�Ҷ�ͼ
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ʹ��Haar�����������������
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # ��ͼ���ϻ��ƾ��ο�����������
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ���ص�һ����⵽������������ΪROI
        return image[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]
    else:
        # ���δ��⵽���������ؿ�ֵ
        return None

def apply_rppg_algorithm(frame):
    heart_rate_estimate_bpm = 0;
    
    global frame_count,start_time
    frame_count += 1

    # ʹ��time.time()��ȡ��ǰʱ�䣬������ÿ�봦���֡����fps��
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time > 1.0:  # ÿ���Ӹ���һ��fps
        global fps
        fps = frame_count / elapsed_time
        #print("FPS:", fps)
        frame_count = 0
        start_time = current_time
    
    # ��֡ת��Ϊ�Ҷ�ͼ
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    # ����֡�е�ƽ��ǿ��
    mean_intensity = np.mean(gray_frame)  

    #print(mean_intensity) 
    # ��ƽ��ǿ����ӵ�rPPG�ź���
    global rppg_signal
    # 128��������
    MAX_SIGNAL_LENGTH = 128
    rppg_signal.append(mean_intensity)  
 
    # ����б����Ƿ񳬹����ֵ
    if len(rppg_signal) > MAX_SIGNAL_LENGTH:     
      # ɾ����������ֵ���б�ĵ�һ��Ԫ�أ�
      rppg_signal.pop(0)

      # �����ͨ�˲����Ĳ���
      low_freq = 1.0  # ���Ƶ��
      high_freq = 3.0  # ���Ƶ��
      if fps == 0:
        fps = 15
      fs = fps  # ������
      # �����ͨ�˲�����ϵ��
      nyquist_freq = 0.5 * fs  # ����Ƶ�ʵ�һ��
      low_cutoff = low_freq / nyquist_freq
      high_cutoff = high_freq / nyquist_freq
      # ����ֹƵ���Ƿ���0��1֮��
      if 0 < low_cutoff < 1 and 0 < high_cutoff < 1:
          # ��ƴ�ͨ�˲���
          b, a = butter(2, [low_cutoff, high_cutoff], btype='band')
          # Ӧ�ô�ͨ�˲���
          filtered_signal = filtfilt(b, a, rppg_signal)
          #print(filtered_signal)
          # ��rPPG�ź�ִ��FFT
          fft_signal = np.fft.fft(filtered_signal)
          #print((fft_signal)) 
          freqs = np.fft.fftfreq(len(filtered_signal), 1.0 / fps)
          #print((freqs)) 
          # 60��130ÿ���ӵ�����Ƶ�ʷ�Χ��Ӧ��Ƶ�ʷ�Χ
          min_heart_rate = 60  # ��С���ʣ�bpm��
          max_heart_rate = 130  # ������ʣ�bpm��
          min_freq = min_heart_rate / 60.0  # ��СƵ�ʣ�ÿ���ӣ�
          max_freq = max_heart_rate / 60.0  # ���Ƶ�ʣ�ÿ���ӣ�
          
          # �ҵ�Ƶ�ʷ�Χ��Ӧ������
          min_freq_index = np.argmax(freqs >= min_freq)
          max_freq_index = np.argmax(freqs >= max_freq)
          
          # ��Ƶ�ʷ�Χ���ҵ���������Ƶ�ʷ���������
          max_amplitude_index = np.argmax(np.abs(fft_signal[min_freq_index:max_freq_index]))
          
          # ��ȡ��Ӧ��Ƶ��ֵ�������ʹ���ֵ��
          heart_rate_estimate = freqs[min_freq_index:max_freq_index][max_amplitude_index]
          
          # �����ʹ���ֵ��Ƶ��ת��Ϊÿ���ӣ�bpm��
          heart_rate_estimate_bpm = heart_rate_estimate * 60.0
      #else:
          # �����ֹƵ�ʳ�����Χ�����
          #print("Error: Digital filter critical frequencies must be 0 < Wn < 1")

    
      #print("------------"+f"{heart_rate_estimate_bpm}")       
    return heart_rate_estimate_bpm  # ת��Ϊÿ���ӵ�������
    #return 80    

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
