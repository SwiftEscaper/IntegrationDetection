import os
import shutil
import cv2

# 파일 초기화
def setup_log():
    txt_file_dir = 'txt_file'
    logs_dir = os.path.join(txt_file_dir, 'logs')
    
    # 최종 사고 결과
    if os.path.isfile(os.path.join(txt_file_dir, 'final.txt')):
        os.remove(os.path.join(txt_file_dir, 'final.txt'))
        
    if os.path.isfile(os.path.join(txt_file_dir, 'fire.txt')):
        os.remove(os.path.join(txt_file_dir, 'fire.txt'))
    
    if os.path.isfile(os.path.join(txt_file_dir, 'crash_func.txt')):
        os.remove(os.path.join(txt_file_dir, 'crash_func.txt'))
        
    # 추적하고 있는 차량 id
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)
    os.makedirs(logs_dir, exist_ok=True)
    

def track_log(cars_dict, version, track_ids):
    # 텍스트 파일 경로
    log_file_path = os.path.join('txt_file/logs', f'version{version}.txt')
    
    with open(log_file_path, 'w') as file:
        file.write(f"Track ID(YOLO): {track_ids}\n")
        file.write(f"Car ID(in Dict): {list(cars_dict.keys())}\n")
 
            
def fire_value_log(prediction):
    with open('txt_file/fire.txt', 'a') as file:
        file.write(f'Fire Result: {prediction}\n')
 
            
def final_log(counter, result, reusltThreshold, overlapped, frame_overlapped, cars_data, images_saved, lat, lng, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join('txt_file', 'final.txt'), 'a') as file:
        file.write(f'Frame: {counter}, Score: {result}\n\n')
        if result >= reusltThreshold:
            file.write(f'Accident happened at frame {frame_overlapped} between cars {overlapped}\n')
            file.write(f'lat: {lat} lng: {lng}\n')
            
            image = images_saved[frame_overlapped]
            for car_label in overlapped:
                cv2.circle(image, (int(cars_data[car_label]['x'][frame_overlapped]), int(cars_data[car_label]['y'][frame_overlapped])), 50, (255, 255, 0), 2)

            cv2.imwrite(os.path.join(output_dir, 'accident.png'), image)
            
    cv2.imwrite(os.path.join(output_dir, 'final_frame.png'), images_saved[-1])
