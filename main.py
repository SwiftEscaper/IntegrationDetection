import time
import numpy as np
from collections import defaultdict
import cv2
import requests
from keras.models import load_model
import tensorflow as tf
import threading

import log
import CrashDetection
import GetFrame

model = tf.keras.models.load_model('C:/vgg16_1.h5')

# 화재 예측 함수
@tf.function
def predict_function(input_tensor):
    return model(input_tensor)

def main():
    accident_flag = 0
    model_path = "yolov8n.pt"
    
    # 로그 파일 디렉토리 설정
    log_directory = log.setup_log()
    
    #######################################################################
    
    # CCTV API 설정 (위도 / 경도)
    lat, lng = 37.517423, 127.17903
    accident_lat, accident_lng = lat, lng
    
    cctv_url, processed_name = GetFrame.get_cctv_data(lat, lng)
    
    #######################################################################
    model, video = CrashDetection.init_model(model_path, cctv_url)

    counter, cars_dict, images_saved = 1, {}, []
    # 사라진 차량 기록 / 차량이 사라졌다고 간주되는 최대 프레임 수 (조정 필요)
    frames_since_last_seen = defaultdict(lambda: 0)
    max_frames_missing = 15
    prev_time, cur_time = time.time(), time.time()

    while True:
        # 시간 측정
        cur_time = time.time()
        times = cur_time - prev_time # 한 프레임 당 처리 시간
        prev_time = cur_time
        
        print(f'Processing frame: {counter}')
        
        #######################################################################
        
        frame, bounding_boxes, track_ids = GetFrame.process_frame(video, model)
        # cars_dict에 저장
        
        if frame is None:
            break
        
        images_saved.append(frame)  # 배열에 이미지 저장 -> 나중에 사고 영역 출력에 사용
        
        #######################################################################
        
        # 이미지 전처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(frame_rgb, (224, 224))
        img_array = np.asarray(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)  # TensorFlow 텐서로 변환
        prediction = predict_function(input_tensor)
            
        log.fire_value_log(prediction)  # 결과 저장
        
        # 일단 임의로 지정함
        if prediction[0][0] < prediction[0][1]:
            accident_flag = 1
        
        #######################################################################
        
        # 이전 기록이 있으면 기록 업데이트
        cars_dict = CrashDetection.update_car_data(cars_dict, bounding_boxes, track_ids, times)
        
        
        
        # frame에서 사라진 차량 기록
        CrashDetection.remove_missing_cars(cars_dict, track_ids, frames_since_last_seen, max_frames_missing)
                      
        # 추적 ID log
        if counter % 50 == 0:
            log.track_log(cars_dict, counter, track_ids, log_directory)  

        # 사고가 감지되지 않았을 때 100 프레임마다 계산
        if accident_flag not in (1, 2) and counter % 100 == 0:
            
            # overlapped: set type, 프레임 내에서 충돌하거나 겹친 차량들의 고유 ID를 저장
            # frame_overlapped: 차량들이 충돌하거나 겹쳤다고 판단된 프레임의 번호
            result, checks, cars_data, overlapped, frame_overlapped = CrashDetection.analyze_cars(
                cars_dict, counter, filter_flag=1, T_var=10, k_overlap=0.5, T_acc=2, 
                frame_overlapped_interval=5, trajectory_thresold=15
            )
            
            reusltThreshold = 1.99
            
            # len(overlapped): 오류 없는지 확인 -> 충돌 사고 판단 알고리즘 수정하기...
            if result > reusltThreshold and len(overlapped) > 0:
                accident_flag = 2
                
                accident_lat, accident_lng = CrashDetection.calculation_location(overlapped, cars_data, frame_overlapped,)

            log.final_log(counter, result, reusltThreshold,  
                         overlapped=overlapped, frame_overlapped=frame_overlapped, 
                         cars_data=cars_data, lat=accident_lat, lng=accident_lng, images_saved=images_saved)
        
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        counter = counter + 1
        
        #if True: # 서버 test를 위해 매frame마다 전송
        if accident_flag in (1, 2):  # 사고라면
            accident_data = {
                'tunnel_name': processed_name,
                'accident_type': accident_flag,
                'latitude': accident_lat,
                'longitude': accident_lng
            }
            
            '''
            try:
                response = requests.post("http://127.0.0.1:8000/accident/", json=accident_data)
                with open('final.txt', 'a', encoding='utf-8') as file:
                    file.write(str(response.json()))
                    file.write(' -> server success\n\n')
            except Exception as e:
                print(f"Error occurred while reporting accident: {e}")
            '''
                
            # break  # Exit loop after detecting accident
    

if __name__ == "__main__":
    main()
