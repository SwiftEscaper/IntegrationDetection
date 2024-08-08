import numpy as np
from numpy import linalg as LA
import cv2

from scipy.interpolate import interp1d

# 차량의 궤적을 보간(interpolation)하는 기능을 제공
class Path():
    """
    interpolation methods: ['original', 'slinear', 'quadratic', 'cubic']
    """
    def __init__(self,data):
        self.data = data

    def interpolate(self, label, number=100, method ='cubic') :
      value_error = 0
      x = np.array(self.data[label]['x'])
      y = np.array(self.data[label]['y'])
      #time = np.array(self.data[label]['time'])
      angle = np.array(self.data[label]['angle'])
      acceleration = np.array(self.data[label]['velocity'])
      velocity = np.array(self.data[label]['acceleration'])
      # self.points = np.stack((x, y, time, angle, velocity, acceleration), axis = 1)
      self.points = np.stack((x, y, angle, velocity, acceleration), axis = 1)

      if method == 'original':
        return self.points
      
      if len(angle) < 3:
        method = 'slinear'
      # Calculate the linear length along the line:
      distance = np.cumsum(np.sqrt(np.sum(np.diff(self.points, axis=0)**2, axis=1)))
      distance = np.insert(distance, 0, 0)/distance[-1]

      # Interpolation itself:
      alpha = np.linspace(0, 1, number)
      try:
        interpolator =  interp1d(distance, self.points, kind=method, axis=0)
        interp_points = interpolator(alpha)
      except ValueError:
        value_error = 1
        interp_points = 0
      return interp_points, value_error


# 필터링을 위한 윈도우 크기와 다항식 차수를 계산
def check_odd_filter(x):
	# It's function used for window and poly order calculation
	# for moving averaging filter

	# x is the size of the window
	# y is the poly order. Should be less than x

	coeff = 1
	x = x// coeff # window size = (size of data)/coefficient
	if x <= 2: 
		x = 3
	if x % 2 == 0:
		x = x - 1
	if x <= 3:
		if x <=2:
			y = 1
		else:	
			y = 2
	else:
		y = 3	
	return (x, y)      

from scipy.signal import savgol_filter

def filter_data(x_pos, y_pos, angle, velocity, acceleration, filter_flag):
  len_on_filter = 2 # minimum length of the data list to apply filter on it
  if filter_flag:
    if len(x_pos) > len_on_filter:
      window_size, polyorder = check_odd_filter(len(x_pos))
      x_pos = savgol_filter(x_pos, window_size, polyorder)
    if len(y_pos) > len_on_filter:
      window_size, polyorder = check_odd_filter(len(y_pos))
      y_pos = savgol_filter(y_pos, window_size, polyorder)
    if len(angle) > len_on_filter:
      window_size, polyorder = check_odd_filter(len(angle))
      angle = savgol_filter(angle, window_size, polyorder)
    if len(velocity) > len_on_filter:
      window_size, polyorder = check_odd_filter(len(velocity))
      velocity = savgol_filter(velocity, window_size, polyorder)
    if len(acceleration) > len_on_filter:
      window_size, polyorder = check_odd_filter(len(acceleration))
      acceleration = savgol_filter(acceleration, window_size, polyorder)
  return x_pos, y_pos, angle, velocity, acceleration    