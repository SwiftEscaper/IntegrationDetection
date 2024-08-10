import cv2
import numpy as np
import requests

import re


def get_cctv_data(lat, lng):
    # 국가교통정보센터 API
    # coordx: 경도 좌표, coordy: 위도 좌표
    # lat: 위도, lng: 경도

    # CCTV 탐색 범위 지정
    minX = str(lng-1)  # 최소 경도 영역
    maxX = str(lng+1)  # 최대 경도 영역
    minY = str(lat-1)  # 최소 위도 영역
    maxY = str(lat+1)  # 최대 위도 영역

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
    
    cctv_url = cctv_data[min_index]['cctvurl']
    
    cctv_name = cctv_data[min_index]['cctvname']
    print('CCTV:', cctv_name)
    processed_name = re.sub(r'\[.*?\]', '', cctv_name).strip()

    return cctv_url, cctv_name

def process_frame(video, model):
    retval, frame = video.read()  # retval: 성공하면 True
    if not retval:
        return None, None

    # conf: 최소 신뢰도 임계값 (이 임계값 미만으로 감지된 객체는 무시)
    # iou: 값이 낮을수록 겹치는 상자가 제거되어 중복을 줄이는 데 유용 (default: 0.7)
    tracks = model.track(frame, persist=True, classes=[2, 3, 5, 7], conf=0.25, iou=0.7)
    
    frame = tracks[0].plot()  # track 이미지
    
    bounding_boxes = tracks[0].boxes.xywh.cpu() # return x, y, w, h
    track_ids = tracks[0].boxes.id

    if track_ids is None:
        track_ids = []
    else:
        track_ids = track_ids.int().cpu().tolist()
            
    return frame, bounding_boxes, track_ids