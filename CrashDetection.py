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
