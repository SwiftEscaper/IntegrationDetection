# https://github.com/MuhammadMoinFaisal/Computervisionprojects/blob/main/YOLOv8_Tracking_Counting_SpeedEstimation/predict.py
# DeepSort에 대해 알아보기

import os
import shutil
import cv2
import numpy as np
import math
import time
import requests
from collections import defaultdict

from ultralytics import YOLO

import CrashFilter as trackingpy # functions for tracking
import CrashCheck as accidentspy # functions for detection

def setup_track_log():
    if os.path.exists('logs'):
        shutil.rmtree('logs')
    log_directory = 'logs'
    os.makedirs(log_directory, exist_ok=True)
    
    if os.path.isfile('logfile.txt'):
        os.remove('logfile.txt')

    return log_directory

def track_log(cars_dict, version, track_ids, log_directory):
    # 텍스트 파일 경로
    log_file_path = os.path.join(log_directory, f'version{version}.txt')
    
    # 파일에 데이터 저장
    with open(log_file_path, 'w') as file:
        file.write(f"Track ID(YOLO): {track_ids}\n\n")
        for car_id, data in cars_dict.items():
            file.write(f"Car ID(in Dict): {car_id}\n")

def get_cctv_data(lat, lng):
    # 국가교통정보센터 API
    # return으로 위도, 경도 값 알 수 있음 -> 나중에 위치 반환할 때 사용
    # coordx: 경도 좌표, coordy: 위도 좌표
    # lat: 위도, lng: 경도

    # CCTV 탐색 범위 지정을 위해 임의로 ±1
    minX = str(lng-1)  # 최소 경도 영역
    maxX = str(lng+1)  # 최대 경도 영역
    minY = str(lat-1)  # 최소 위도 영역
    maxY = str(lat+1)  # 최대 위도 영역

    # getType: 출력 결과 형식(xml, json / 기본: xml)
    api_call = 'https://openapi.its.go.kr:9443/cctvInfo?' \
            'apiKey=ea732642a365461f96f8ea3b63c00317' \
            '&type=ex&cctvType=1' \
            '&minX=' + minX + \
            '&maxX=' + maxX + \
            '&minY=' + minY + \
            '&maxY=' + maxY + \
            '&getType=json'
                
    dataset = requests.get(api_call).json()
    cctv_data = dataset['response']['data']

    coordx_list = []
    for index, data in enumerate(cctv_data):
        xy_couple = (float(cctv_data[index]['coordy']),float(cctv_data[index]['coordx']))
        coordx_list.append(xy_couple)

    # 입력한 위경도 좌표에서 가장 가까운 위치에 있는 CCTV를 찾는 과정
    coordx_list = np.array(coordx_list)
    leftbottom = np.array((lat, lng))
    distances = np.linalg.norm(coordx_list - leftbottom, axis=1)
    min_index = np.argmin(distances)

    print('CCTV:', cctv_data[min_index]['cctvname'])
    
    return cctv_data[min_index]

def init_model(model_path, video_url):
    model = YOLO(model_path)
    video = cv2.VideoCapture(video_url)
    return model, video

def process_frame(video, model):
    retval, image = video.read()  # retval: 성공하면 True
    if not retval:
        return None, None

    # conf: 최소 신뢰도 임계값 (이 임계값 미만으로 감지된 객체는 무시)
    # iou: 값이 낮을수록 겹치는 상자가 제거되어 중복을 줄이는 데 유용 (default: 0.7)
    tracks = model.track(image, persist=True, classes=[2, 3, 5, 7], conf=0.25, iou=0.7)
    image = tracks[0].plot()
    
    bounding_boxes = tracks[0].boxes.xywh.cpu() # return x, y, w, h
    track_ids = tracks[0].boxes.id

    if track_ids is None:
        track_ids = []
    else:
        track_ids = track_ids.int().cpu().tolist()

    return image, (bounding_boxes, track_ids)

def update_car_data(cars_dict, bounding_boxes, track_ids, times):
    # 각 객체의 위치를 기록 업데이트
    # 중심점, 위치 벡터, 속도, 가속도, 박스 정보(x, y, w, h)
    # 항상 현재 프레임에서 감지된 track_id만을 처리
    for box, track_id in zip(bounding_boxes, track_ids):
        x, y, w, h = map(int, box)  # 박스 좌표 (이때 x, y는 센터 좌표)
        box = [int(x - w / 2), int(y - h / 2), w, h]  # 좌측 하단 point 저장
        
        if track_id not in cars_dict:
            cars_dict[track_id] = [[[x, y]], [[0, 0]], [0], [0], box]
        else:
            prev_center = cars_dict[track_id][0][-1]
            cars_dict[track_id][0].append([x, y])  # 새로운 중심점 추가
            
            motion_vector = np.subtract([x, y], prev_center)
            distance = np.sqrt(motion_vector[0]**2 + motion_vector[1]**2)
            if distance != 0:
                motion_vector = np.divide(motion_vector, distance)
            cars_dict[track_id][1].append(motion_vector)  # 새로운 위치 벡터 추가

            prev_velocity = cars_dict[track_id][2][-1]
            current_velocity = distance / times
            cars_dict[track_id][2].append(current_velocity)  # 새로운 속도 추가

            vec_diff = abs(current_velocity - prev_velocity)
            current_acceleration = vec_diff / times
            cars_dict[track_id][3].append(current_acceleration)  # 새로운 가속도 추가

            cars_dict[track_id][4] = box  # 박스 정보 업데이트

    return cars_dict

def remove_missing_cars(cars_dict, track_ids, frames_since_last_seen, max_frames_missing):
    # cars_dict에 있는 차량 중 track_ids에 없는 차량 제거 (max_frames_missing동안 없다면)
    for track_id in list(cars_dict.keys()):
        if track_id not in track_ids:
            if track_id in frames_since_last_seen:
                frames_since_last_seen[track_id] += 1
            else:
                frames_since_last_seen[track_id] = 1

            if frames_since_last_seen[track_id] > max_frames_missing:
                del cars_dict[track_id]
                del frames_since_last_seen[track_id] 
                
def check_overlap(cars_data, cars_labels_to_analyze, k_overlap, counter):
    checks = [0, 0, 0]
    frame_overlapped = -1
    overlapped = set()
    flag = 1
    # frames = [int(i) for i in range(frame_end_with - frame_start_with)]
    frames = range(counter - 100, counter) # 분석하고자하는 frame
    for frame in frames:
        for first_car in cars_labels_to_analyze:
            for second_car in cars_labels_to_analyze:
                if (int(second_car) != int(first_car)) and (
                    accidentspy.check_overlap(
                        (cars_data[first_car]['x'][frame], cars_data[first_car]['y'][frame]),
                        (cars_data[second_car]['x'][frame], cars_data[second_car]['y'][frame]),
                        cars_data[first_car]['car diagonal'], cars_data[second_car]['car diagonal'], k_overlap
                    )
                ):
                    overlapped.add(first_car)
                    overlapped.add(second_car)
                    if flag:
                        frame_overlapped = frame
                        flag = 0
                                
    checks[0] = 1.0 if not flag else 0.5
    return checks, overlapped, frame_overlapped

def check_acceleration_anomaly(cars_data, potential_cars_labels, frame_overlapped, frame_overlapped_interval, T_acc):
    frames_before = range(frame_overlapped - frame_overlapped_interval, frame_overlapped)
    acc_average = []
    for label in potential_cars_labels:
        acc_av = 0
        t = 1
        for frame in frames_before:
            acc_av = acc_av * (t - 1) / t + cars_data[label]['acceleration'][frame] / t
            t += 1
        acc_average.append(acc_av)
        
    frames_after = range(frame_overlapped, frame_overlapped + frame_overlapped_interval)
    acc_maximum = []
    for label in potential_cars_labels:
        acc_max = 0
        for frame in frames_after:
            if cars_data[label]['acceleration'][frame] > acc_max:
                acc_max = cars_data[label]['acceleration'][frame]
        acc_maximum.append(acc_max)

    acc_diff = np.mean(np.subtract(acc_maximum, acc_average))
    return acc_diff

def check_angle_anomaly(cars_data, potential_cars_labels, frame_overlapped, frame_overlapped_interval, trajectory_thresold):
    angle_anomalies = []
    for label in potential_cars_labels:
        angle_difference = accidentspy.check_angle_anomaly(cars_data[label]['angle'], frame_overlapped, frame_overlapped_interval)
        angle_anomalies.append(angle_difference)

    return max(angle_anomalies) if len(angle_anomalies) > 0 else 0

def analyze_cars(cars_dict, counter, filter_flag, T_var, k_overlap, T_acc, frame_overlapped_interval, trajectory_thresold):
    cars_data, cars_labels_to_analyze = {}, []

    for label, data in cars_dict.items():
        x_pos = [pos[0] for pos in data[0]]
        y_pos = [pos[1] for pos in data[0]]
        angle = [np.arccos(vec[0]) for vec in data[1]]
        velocity, acceleration, (w, h) = data[2], data[3], data[4][2:4]

        x_pos, y_pos, angle, velocity, acceleration = trackingpy.filter_data(
            x_pos, y_pos, angle, velocity, acceleration, filter_flag)
        
        car_diagonal = np.sqrt(w**2 + h**2)
        cars_data[label] = {
            'x': x_pos,
            'y': y_pos,
            'angle': angle,
            'velocity': velocity,
            'acceleration': acceleration,
            'car diagonal': car_diagonal
        }
        
        movement_variance = np.var(np.sqrt(np.array(x_pos)**2 + np.array(y_pos)**2))
        if movement_variance >= T_var:
            cars_labels_to_analyze.append(label)
        else:
            del cars_data[label]

    path = trackingpy.Path(cars_data)
    for label in cars_labels_to_analyze:
        interp_points, value_error = path.interpolate(label, number=counter, method='cubic')
        if value_error == 0:
            cars_data[label]['x'], cars_data[label]['y'], \
            cars_data[label]['angle'], cars_data[label]['velocity'], \
            cars_data[label]['acceleration'] = interp_points.T

    checks, overlapped, frame_overlapped = check_overlap(cars_data, cars_labels_to_analyze, k_overlap, counter)
    
    if checks[0] == 1.0:
        acc_diff = check_acceleration_anomaly(cars_data, overlapped, frame_overlapped, frame_overlapped_interval, T_acc)
        checks[1] = 1.0 if acc_diff >= T_acc else 0.5
        
        max_angle_change = check_angle_anomaly(cars_data, overlapped, frame_overlapped, frame_overlapped_interval, trajectory_thresold)
        checks[2] = 1.0 if max_angle_change >= trajectory_thresold else 0.5

    result = sum(checks)
    return result, checks, cars_data, overlapped, frame_overlapped

def save_results(counter, result, reusltThreshold, checks, overlapped, frame_overlapped, cars_data, images_saved, lat, lng, output_dir='output'):
    log_file_path = 'logfile.txt'

    with open(log_file_path, 'a') as file:
        file.write(f'Frame: {counter}, Score: {result}\n')
        if result >= reusltThreshold:
            file.write(f'Accident happened at frame {frame_overlapped} between cars {overlapped}\n')
            file.write(f'lat: {lat} lng: {lng}\n\n')
            
            image = images_saved[frame_overlapped]
            for car_label in overlapped:
                cv2.circle(image, (int(cars_data[car_label]['x'][frame_overlapped]), int(cars_data[car_label]['y'][frame_overlapped])), 50, (255, 255, 0), 2)

            cv2.imwrite(os.path.join(output_dir, 'accident.png'), image)
            
    cv2.imwrite(os.path.join(output_dir, 'final_frame.png'), images_saved[-1])
