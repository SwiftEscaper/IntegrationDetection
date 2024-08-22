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


def init_model(model_path, video_url):
    model = YOLO(model_path)
    video = cv2.VideoCapture(video_url)
    return model, video

def update_car_data(cars_dict, bounding_boxes, track_ids, times):
    # 각 객체의 위치를 기록 업데이트
    # 중심점, 위치 벡터, 속도, 가속도, 박스 정보(x, y, w, h)
    # 항상 현재 프레임에서 감지된 track_id만을 처리
    for box, track_id in zip(bounding_boxes, track_ids):
        x, y, w, h = map(int, box)  # 박스 좌표 (이때 x, y는 센터 좌표)
        box = [int(x - w / 2), int(y - h / 2), w, h]  # 좌측 하단 point 저장
        
        if track_id not in list(cars_dict.keys()):
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
            if times > 0:
                current_velocity = distance / times
            else: current_velocity = 0
            cars_dict[track_id][2].append(current_velocity)  # 새로운 속도 추가

            vec_diff = abs(current_velocity - prev_velocity) if not np.isnan(current_velocity) and not np.isnan(prev_velocity) else 0
            if times > 0:
                current_acceleration = vec_diff / times
            else: current_acceleration = 0
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
       
    # checks[0] = 1.0 if not flag else 0.5                         
    checks[0] = 0.0 if not flag else 0.0
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
    
    ##### 이거 수정
    acc_diff = check_acceleration_anomaly(cars_data, overlapped, frame_overlapped, frame_overlapped_interval, T_acc)
    max_angle_change = check_angle_anomaly(cars_data, overlapped, frame_overlapped, frame_overlapped_interval, trajectory_thresold)
    
    # checks[1] = 1.0 if acc_diff >= T_acc else 0.5
    # checks[2] = 1.0 if max_angle_change >= trajectory_thresold else 0.5
    
    if acc_diff >= T_acc:
        checks[1] = 1.0
    else: 
        checks[1] = 0.5
        
    if max_angle_change >= trajectory_thresold :
        checks[2] = 1.0
    else: 
        checks[2] = 0.5
        
    with open('txt_file/acc.txt', 'a') as file:
        file.write(f'Acc Result: {checks[1]}\n')
    
    with open('txt_file/traj.txt', 'a') as file:
        file.write(f'Traj Result: {checks[2]}\n')
    
    ####################################################################################################################### 이거 수정
    '''
    if checks[0] == 1.0:
        acc_diff = check_acceleration_anomaly(cars_data, overlapped, frame_overlapped, frame_overlapped_interval, T_acc)
        checks[1] = 1.0 if acc_diff >= T_acc else 0.5
        
        max_angle_change = check_angle_anomaly(cars_data, overlapped, frame_overlapped, frame_overlapped_interval, trajectory_thresold)
        checks[2] = 1.0 if max_angle_change >= trajectory_thresold else 0.5
    '''
    result = sum(checks)
    return result, checks, cars_data, overlapped, frame_overlapped

def calculation_location(overlapped, cars_data, frame_overlapped, accident_lat, accident_lng):
    first_car = list(overlapped)[0]
    
    accident_x = cars_data[first_car]['x'][frame_overlapped]
    accident_y = cars_data[first_car]['y'][frame_overlapped]

    # 픽셀 좌표를 위도 경도 변화로 변환 (간단한 선형 변환 사용)
    # 예: 1 픽셀 이동당 0.00001도 변화
    # 실제 환경에서는 정확한 매핑을 위해 추가적인 보정 필요
    lat_change_per_pixel = 0.00001  # 위도 변화율 (가정)
    lng_change_per_pixel = 0.00001  # 경도 변화율 (가정)

    accident_lat += (accident_y * lat_change_per_pixel)
    accident_lng += (accident_x * lng_change_per_pixel)
    
    return accident_lat, accident_lng


# cars_dict: 현재 추적하고 있는 car의 정보
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
# counter: 프레임 번호
# 여기에서 1분동안 (100개의 프레임동안) x, y 좌표 차이가 크게 달라지지 않으면 사고라고 감지 (return true)

def stop_detection(cars_dict):
    stationary_cars = 0  # 정지한 차량 수
    total_cars = len(cars_dict)  # 전체 차량 수
    
    # 차량이 없는 경우 바로 False 반환
    if total_cars == 0:
        return False
    
    # 모든 차량에 대해 사고 여부 확인
    for car_id, data in cars_dict.items():
        
        # 각 차량의 마지막 100 프레임 동안의 x, y 좌표 리스트 가져오기
        centers = data[0][-100:]  # 중심점 리스트에서 마지막 100 프레임의 좌표 가져오기
        x_positions = [pos[0] for pos in centers]  # x 좌표 리스트 (한 차량에 대해)
        y_positions = [pos[1] for pos in centers]  # y 좌표 리스트 (한 차량에 대해)
        
        # x_positions의 길이가 1 이하일 경우 건너뛰기
        if len(x_positions) <= 1:
            continue
        
        # 변화 없는 프레임 수 카운트
        stationary_frames = 0
        threshold = 2  # 임계값 설정 (예: 2 픽셀 이하의 변화는 정지로 판단)

        # 한 차량에 대해 계산 (100프레임동안)
        for i in range(1, len(x_positions)):
            x_diff = abs(x_positions[i] - x_positions[i-1])
            y_diff = abs(y_positions[i] - y_positions[i-1])
            
            # 변화량이 임계값 이하인 경우 stationary_frames 증가
            if x_diff < threshold and y_diff < threshold:
                stationary_frames += 1
                
            print('stationary_frames: ', stationary_frames)
        
        ################### 이번 차량 계산 종료
        
        # 비율 계산 (50% 이상 멈춰있으면 사고로 간주 -> 100 프레임 동안 한 차량이)
        stationary_ratio = stationary_frames / (len(x_positions) - 1)
        print(stationary_ratio)
        
        # 차량이 50% 이상의 프레임 동안 멈춰있다면 사고로 판단 -> 한 차량에서 프레임 중 50% 이상 멈춰있으면
        if stationary_ratio >= 0.5:
            stationary_cars += 1  # 정지한 차량 수 증가
            
     # 정지한 차량이 전체의 50% 이상이면 사고로 판단
    if stationary_cars / total_cars >= 0.5:
        return True
    
    return False

if __name__ == "__main__":
    cars_dict = {
        '1': [
            [[1, 1], [2, 2]],  # 중심점 리스트
            ],
        '2' : [
            [[2, 2],[5, 5]],  # 중심점 리스트
            ],
        '3' : [
            [[4, 4], [10, 10]],  # 중심점 리스트
            ]
    }
    
    print(stop_detection(cars_dict))