from keras.models import load_model
import time
from collections import defaultdict
import cv2
import os
import numpy as np
import json

from keras.preprocessing import image

import CrashDetection
import GetFrame

import tensorflow as tf

model = tf.keras.models.load_model('C:/vgg16_1.h5')

# 화재 예측 함수
@tf.function
def predict_function(input_tensor):
    return model(input_tensor)

def main():
    accident_flag = 0
    
    # 로그 파일 디렉토리 설정
    log_directory = CrashDetection.setup_track_log()
    #######################################################################
    # CCTV API 설정
    lat = 37.517423  # 위도
    lng = 127.179039  # 경도
    accident_lat, accident_lng = lat, lng
    #######################################################################
    model_path = "yolov8n.pt"
    cctv_data = GetFrame.get_cctv_data(lat, lng)
    
    model, video = CrashDetection.init_model(model_path, cctv_data['cctvurl'])

    counter, cars_dict, images_saved = 1, {}, []
    
    # cars_dict: 차량은 고유 라벨로 식별
    '''
    cars_dict = {
        '1': [
            [[center1_frame1, center1_frame2, ...]],  # 중심점 리스트
            [[[vector1_frame1], [vector1_frame2], ...]],  # 위치 벡터 리스트
            [velocity1_frame1, velocity1_frame2, ...],  # 속도 리스트
            [acceleration1_frame1, acceleration1_frame2, ...],  # 가속도 리스트
            [x, y, w, h]  # 박스 정보
        ],
    }
    '''

    # 사라진 차량 기록
    frames_since_last_seen = defaultdict(lambda: 0)
    # 차량이 사라졌다고 간주되는 최대 프레임 수 (조정 필요)
    max_frames_missing = 30

    prev_time, cur_time = 0, 0

    if os.path.isfile('fire.txt'):
        os.remove('fire.txt')
        
    if os.path.isfile('result.txt'):
        os.remove('result.txt')


    while True:
        
        prev_time = cur_time
        cur_time = time.time()
        
        print(f'Processing frame: {counter}')
        tracks, frame = GetFrame.process_frame(video, model)
        
        bounding_boxes = tracks[0].boxes.xywh.cpu() # return x, y, w, h
        track_ids = tracks[0].boxes.id
        
        tracks = (bounding_boxes, track_ids)

        if track_ids is None:
            track_ids = []
        else:
            track_ids = track_ids.int().cpu().tolist()
        
        if frame is None:
            break
        images_saved.append(frame)  # 배열에 이미지 저장 -> 나중에 사고 영역 출력에 사용
        bounding_boxes, track_ids = tracks
        
        
        # 이미지 전처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(frame_rgb, (224, 224))
        img_array = np.asarray(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)  # TensorFlow 텐서로 변환
        prediction = predict_function(input_tensor)

        with open('fire.txt', 'a') as file:
            file.write(f'Fire Result: {prediction}\n')
        
        # 일단 임의로 지정함
        if prediction[0][0] < prediction[0][1]:
            accident_flag = 1
        
        times = cur_time - prev_time
        cars_dict = CrashDetection.update_car_data(cars_dict, bounding_boxes, track_ids, times)
        
        CrashDetection.remove_missing_cars(cars_dict, track_ids, frames_since_last_seen, max_frames_missing)
                      
        # 추적 ID log
        if counter % 50 == 0:
            CrashDetection.track_log(cars_dict, counter, track_ids, log_directory)  

        # 100 프레임마다 계산
        if accident_flag not in (1, 2) and counter % 100 == 0:
            
            result, checks, cars_data, overlapped, frame_overlapped = CrashDetection.analyze_cars(
                cars_dict, counter, filter_flag=1, T_var=10, k_overlap=0.5, T_acc=2, 
                frame_overlapped_interval=5, trajectory_thresold=15
            )
            
            
            reusltThreshold = 1.99
            # 이건 위치 계산
            
            # 사고 지점 위도 경도 구하기 (예: 1 픽셀 이동당 0.00001도 변화)
            if result > reusltThreshold and len(overlapped) > 0:
                accident_flag = 2
                
                # overlapped에서 첫 번째 차량을 가져옴
                first_car = list(overlapped)[0]
                
                # 사고 지점에서의 픽셀 좌표 변화
                accident_x = cars_data[first_car]['x'][frame_overlapped]
                accident_y = cars_data[first_car]['y'][frame_overlapped]

                # 픽셀 좌표를 위도 경도 변화로 변환 (간단한 선형 변환 사용)
                # 실제 환경에서는 정확한 매핑을 위해 추가적인 보정 필요
                lat_change_per_pixel = 0.00001  # 위도 변화율 (가정)
                lng_change_per_pixel = 0.00001  # 경도 변화율 (가정)

                accident_lat += (accident_y * lat_change_per_pixel)
                accident_lng += (accident_x * lng_change_per_pixel)
                
            CrashDetection.save_results(counter, result, reusltThreshold, checks=checks, 
                         overlapped=overlapped, frame_overlapped=frame_overlapped, 
                         cars_data=cars_data, lat=accident_lat, lng=accident_lng, images_saved=images_saved)
        
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        counter = counter + 1
        
        # 사고라면
        if accident_flag in (1, 2):
            # 파일에 기록
            with open('result.txt', 'a') as file:
                file.write(f'accident location: (lat, lng) -> ({accident_lat}, {accident_lng})\n')
                file.write(f'accident type: {accident_flag}\n\n')
            accident_flag = 0
            
            
            accident_info = {
                'accident_location': {
                    'latitude': accident_lat,
                    'longitude': accident_lng
                },
                'accident_type': accident_flag
            }
            
            '''
            # JSON 형식으로 파일에 기록
            with open('result.json', 'a') as file:
                json.dump(accident_info, file)
                file.write('\n')  # 각 사고 정보를 새 줄에 기록
            '''
    

if __name__ == "__main__":
    main()