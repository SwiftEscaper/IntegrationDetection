import numpy as np

def check_overlap(first_car, second_car, one_diag, second_diag, k):
  dist = np.sqrt((first_car[0]-second_car[0])**2 + (first_car[1]-second_car[1])**2)
  
  # 두 차량이 겹칠 수 있는 최대 거리
  threshold = one_diag + second_diag
  
  if (dist < threshold*k):
    check = True
  else:
    check = False  
  
  return check

def check_angle_anomaly(angle_list_1st, frame, check_frames):
  # 주어진 프레임 이전과 이후 각도 변화 추출
  angle_change = angle_list_1st[frame-check_frames: frame+check_frames]
  
  # 각도 변화가 있는 경우
  if len(angle_change)>0:
    diff = max(angle_change)-min(angle_change)
  else:
    diff = 0
  
  # return: 각도 변화의 정도
  return diff

# 충돌 시 각도 차이가 일정 임계값을 초과하는지 확인
def check_crash_angle(angle_1st_car, angle_2nd_car, threshold):
  if (angle_1st_car-angle_2nd_car) > threshold:
    check=True
  else:
    check = False
    
  return check	