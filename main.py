import time
import numpy as np
import cv2
from collections import defaultdict
import requests
from keras.models import load_model
import tensorflow as tf
import multiprocessing


import log
import CrashDetection
import GetFrame

YOLO_MODEL_PATH = 'yolov8n.pt'
FIRE_MODEL_PATH = 'C:/vgg16_BATCH_32.h5'
FIRE_MODEL_IMAGE_SIZE = 192
RAIN_AUTH_KEY = '1rEfi1JvQG6xH4tSbwBu9Q'

LATITUDE = 37.517423
LONGITUDE = 127.17903

model = tf.keras.models.load_model(FIRE_MODEL_PATH)

# 화재 예측 함수
@tf.function
def predict_function(input_tensor):
    return model(input_tensor)

def fire_detection(accident_flag, frame_queue, event):
    while True:
        frame = frame_queue.get()
        
        # 이미지 전처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(frame_rgb, (FIRE_MODEL_IMAGE_SIZE, FIRE_MODEL_IMAGE_SIZE))
        img_array = np.asarray(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
            
        input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        prediction = predict_function(input_tensor)
                
        #log.fire_value_log(prediction)
        
        with open('txt_file/fire.txt', 'a') as file:
            file.write(f'{prediction}: ')
        
        with open('txt_file/fire.txt', 'a') as file:
            if prediction[0][0] < prediction[0][1]:
                file.write(f'Fire\n\n')
            else:
                file.write(f'Not Fire\n\n')
            
        # 0: non-fire, 1: fire
        if prediction[0][0] < prediction[0][1]:
            accident_flag.value = 1
            event.set()  # 이벤트 발생


# 동기화 문제 발생 -> 해결했는데 다 해결은 아닌거 같기도 하고
def crash_detection(accident_flag, frame_queue, bounding_boxes_queue, track_ids_queue, 
                    frames_since_last_seen, max_frames_missing, images_saved, 
                    times_queue, counter_queue, accident_lat, accident_lng, cars_dict, event):
    while True:
        counter = counter_queue.get()
        
        frame = frame_queue.get()
        bounding_boxes = bounding_boxes_queue.get()
        track_ids = track_ids_queue.get()
        times = times_queue.get()

        # if frame is None:
        #    break
        
        cars_dict = CrashDetection.update_car_data(cars_dict, bounding_boxes, track_ids, times)
        CrashDetection.remove_missing_cars(cars_dict, track_ids, frames_since_last_seen, max_frames_missing)
        
        ######################################################### 여기까지 현재 cars_dict에는 추적하고 있는 car의 정보만 저장
         
        # 로그 기록: 지금 추적하고 있는 car_id             
        if counter % 100 == 0:
            log.track_log(cars_dict, counter, track_ids)  

        # 100 프레임마다 사고 여부 계산 ----------------------------------------------------------------------> 여기부터 수정 필요
        #if accident_flag.value not in (1, 2) and counter % 100 == 0:
        #if accident_flag.value not in (1, 2) :
        if counter != 0 and counter % 100 == 0:
            '''
            # overlapped: set type, 프레임 내에서 충돌하거나 겹친 차량들의 고유 ID를 저장
            # frame_overlapped: 차량들이 충돌하거나 겹쳤다고 판단된 프레임의 번호 
            result, checks, cars_data, overlapped, frame_overlapped = CrashDetection.analyze_cars(
                cars_dict, counter, filter_flag=1, T_var=10, k_overlap=0.5, T_acc=2, 
                frame_overlapped_interval=5, trajectory_thresold=15
            )
            '''
            result = False
            result = CrashDetection.stop_detection(cars_dict)
            
            with open('txt_file/crash_func.txt', 'a') as file:
                file.write(f'{counter}: {result}\n\n')
            
            resultThreshold = 1.99
            
            # len(overlapped): 오류 없는지 확인 -> 충돌 사고 판단 알고리즘 수정하기...
            #if result > resultThreshold and len(overlapped) > 0:
            if result == True:
                accident_flag.value = 2
                #accident_lat.value, accident_lng.value = CrashDetection.calculation_location(overlapped, cars_data, frame_overlapped, accident_lat, accident_lng)
                event.set()  # Set the event to notify the main process
        '''
        if counter.value % 100 == 0:
            log.final_log(counter.value, result, resultThreshold,  
                            overlapped=overlapped, frame_overlapped=frame_overlapped, 
                            cars_data=cars_data, lat=accident_lat.value, lng=accident_lng.value, images_saved=images_saved)
        '''

def main():
    accident_flag = multiprocessing.Value('i', 0)
    
    # 로그 파일 디렉토리 설정
    log.setup_log()
    
    #######################################################################
    
    # CCTV API 설정 (위도 / 경도)
    accident_lat = multiprocessing.Value('d', LATITUDE)
    accident_lng = multiprocessing.Value('d', LONGITUDE)
    
    cctv_url, processed_name = GetFrame.get_cctv_data(LATITUDE, LONGITUDE)
    
    #######################################################################
    # local 동영상 경로
    # cctv_url = 'trafficAccident1.mp4'
    # cctv_url = 'fire2.mp4'
    
    model, video = CrashDetection.init_model(YOLO_MODEL_PATH, cctv_url)

    counter = 1
    images_saved = []
    # 사라진 차량 기록 / 차량이 사라졌다고 간주되는 최대 프레임 수 (조정 필요)
    frames_since_last_seen = {}
    max_frames_missing = 15
    prev_time, cur_time, times = time.time(), time.time(), 0
    
    manager = multiprocessing.Manager()
    cars_dict = manager.dict()
    
    # 멀티프로세싱 큐 설정
    fire_frame_queue = multiprocessing.Queue()
    crash_frame_queue = multiprocessing.Queue()
    bounding_boxes_queue = multiprocessing.Queue()
    track_ids_queue = multiprocessing.Queue()
    counter_queue = multiprocessing.Queue()
    times_queue = multiprocessing.Queue()
    
    # 이벤트 설정
    event = multiprocessing.Event()
    
    crash_process = multiprocessing.Process(target=crash_detection, args=(accident_flag, crash_frame_queue, bounding_boxes_queue, 
                                                                          track_ids_queue, frames_since_last_seen, 
                                                                          max_frames_missing, images_saved, times_queue, counter_queue, 
                                                                          accident_lat.value, accident_lng.value, cars_dict, event))
    fire_process = multiprocessing.Process(target=fire_detection, args=(accident_flag, fire_frame_queue, event))

    fire_process.start()
    crash_process.start()
  
    prev_time, cur_time, times = time.time(), time.time(), 0
    event = multiprocessing.Event()
    accident_flag = multiprocessing.Value('i', 0)
    counter = 0
    
    while True:
        # 스케줄러 작업 실행
        #schedule.run_pending()  # 예약된 작업 실행
        #time.sleep(1) 
        
        cur_time = time.time()
        times = cur_time - prev_time
        prev_time = cur_time
        
        ###########################################################################################################
                
        print(f'Processing frame: {counter}')
        
        frame, bounding_boxes, track_ids = GetFrame.process_frame(video, model)
        
        if frame is None:
            break
        
        images_saved.append(frame)
        cv2.imshow("Image", frame)

        # 큐가 비어있을 때 기본적으로 대기 상태 (추가될 때까지 기다림)
        fire_frame_queue.put(frame)
        crash_frame_queue.put(frame)
        bounding_boxes_queue.put(bounding_boxes)
        track_ids_queue.put(track_ids)
        counter_queue.put(counter)
        times_queue.put(times)

        # 이벤트 대기
        # 서버 테스트
        if False:
        #if event.is_set():
            accident_data = {
                'tunnel_name': processed_name,
                'accident_type': accident_flag.value,
                'latitude': accident_lat.value,
                'longitude': accident_lng.value
            }
            
            accident_data = {
                'tunnel_name': 'temp',
                'accident_type': 10,
                'latitude': 5.0,
                'longitude': 5.0
            }

            # 서버에 사고 정보 전송
            
            try:
                response = requests.post("http://61.252.59.35:8080/api/accident/", json=accident_data)
                #response = requests.post("http://127.0.0.1:8000//api/accident/", json=accident_data)  # local address
                
                # log code
                with open('server.txt', 'a', encoding='utf-8') as file:
                    file.write(str(response.json()))
                    file.write(' -> server success\n\n')
            except Exception as e:
                print(f"Error occurred while reporting accident: {e}")
                              
            # 초기화
            accident_flag.value = 0 
            accident_lat.value = LATITUDE
            accident_lng.value = LONGITUDE
    
            event.clear()  # Reset the event after handling the accident

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        counter += 1

    fire_frame_queue.put(None)
    crash_frame_queue.put(None)
    bounding_boxes_queue.put(None)
    track_ids_queue.put(None)
    counter_queue.put(None)
    times_queue.put(None)

    
    fire_process.join()
    crash_process.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()