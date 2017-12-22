import pyscreenshot as ImageGrab
import cv2
import numpy as np
import pyautogui
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

no_parkingl=dir_path+'/1.png'
no_parking=cv2.imread(no_parkingl)
gray_no_parking=cv2.cvtColor(no_parking,cv2.COLOR_BGR2GRAY)
ret,bin_no_parking = cv2.threshold(gray_no_parking,127,255,cv2.THRESH_BINARY)

no_uturnl=dir_path+'/templates/no_uturn.png'
no_uturn=cv2.imread(no_uturnl)
gray_no_uturn=cv2.cvtColor(no_uturn,cv2.COLOR_BGR2GRAY)
ret,bin_no_uturn = cv2.threshold(gray_no_uturn,127,255,cv2.THRESH_BINARY)

res_no_uturn=cv2.matchTemplate(bin_no_parking,bin_no_uturn,cv2.TM_CCOEFF_NORMED)
loc_no_uturn=np.where(res_no_uturn>=0.2)
print loc_no_uturn
