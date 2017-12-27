
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# #
# loc=
# cv2.destroyAllWindows()
dir_path = os.path.dirname(os.path.realpath(__file__))
# im=ImageGrab.grab() # X1,Y1,X2,Y2
# imnp=np.array(im)
# gray11=cv2.cvtColor(imnp,cv2.COLOR_BGR2GRAY)
# template_win=cv2.imread('vlc_header_template.png')
# gray_template_vlc=cv2.cvtColor(template_win,cv2.COLOR_BGR2GRAY)
# w,h=gray_template_vlc.shape[::-1]
# res_win=cv2.matchTemplate(gray11,gray_template_vlc,cv2.TM_CCOEFF_NORMED)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_win)
# print(min_loc[0],max_loc[1])
# lc1=max_loc[0]
# lc2=max_loc[1]+h


trafficl=dir_path+'/templates/traffic.png'
traffic=cv2.imread(trafficl)
gray_traffic=cv2.cvtColor(traffic,cv2.COLOR_BGR2GRAY)
ret,bin_traffic = cv2.threshold(gray_traffic,60,255,cv2.THRESH_BINARY)

no_uturnl=dir_path+'/templates/no_uturn.png'
no_uturn=cv2.imread(no_uturnl)
gray_no_uturn=cv2.cvtColor(no_uturn,cv2.COLOR_BGR2GRAY)
ret,bin_no_uturn = cv2.threshold(gray_no_uturn,60,255,cv2.THRESH_BINARY)

stopl=dir_path+'/templates/stop.png'
stop=cv2.imread(stopl)
gray_stop=cv2.cvtColor(stop,cv2.COLOR_BGR2GRAY)
ret,bin_stop = cv2.threshold(gray_stop,60,255,cv2.THRESH_BINARY)

serpantinel=dir_path+'/templates/serpantine.png'
serpantine=cv2.imread(serpantinel)
gray_serpantine=cv2.cvtColor(serpantine,cv2.COLOR_BGR2GRAY)
ret,bin_serpantine = cv2.threshold(gray_serpantine,60,255,cv2.THRESH_BINARY)

count=0
vidrec=cv2.VideoCapture(0)
while True:
    #Grab Window
    # im=ImageGrab.grab(bbox=(lc1,lc2,lc1+588,lc2+362)) # X1,Y1,X2,Y2
    
    ret0,im=vidrec.read()
    imnp=np.array(im)
    #operate with cv2
    gray=cv2.cvtColor(imnp,cv2.COLOR_BGR2GRAY)
    ret,binm = cv2.threshold(imnp,60,255,cv2.THRESH_BINARY)
    # cv2.imshow('frame',imnp)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    # break
    res_traffic=cv2.matchTemplate(binm,bin_traffic,cv2.TM_CCORR_NORMED)
    res_no_uturn=cv2.matchTemplate(binm,bin_no_uturn,cv2.TM_CCORR_NORMED)
    res_stop=cv2.matchTemplate(binm,bin_stop,cv2.TM_CCORR_NORMED)
    res_serpantine=cv2.matchTemplate(binm,bin_serpantine,cv2.TM_CCORR_NORMED)

    threshold=0.7

    loc_traffic=np.where(res_traffic>=threshold)
    loc_no_uturn=np.where(res_no_uturn>=threshold)
    loc_stop=np.where(res_stop>=threshold)
    loc_serpantine=np.where(res_serpantine>=threshold)


    for pt in zip(*loc_traffic[::-1]):
        print('traffic',pt[0],pt[1])
        cv2.rectangle(binm,pt,(pt[0]+13,pt[1]+14),(255,255,255),2)
        break
    for pt in zip(*loc_no_uturn[::-1]):
        print('no_uturn',pt[0],pt[1])
        cv2.rectangle(binm,pt,(pt[0]+13,pt[1]+14),(255,255,255),2)
        break
    for pt in zip(*loc_stop[::-1]):
        print('stop',pt[0],pt[1])
        cv2.rectangle(binm,pt,(pt[0]+13,pt[1]+14),(255,255,255),2)
        break
    for pt in zip(*loc_serpantine[::-1]):
        print('serpantine',pt[0],pt[1])
        cv2.rectangle(binm,pt,(pt[0]+13,pt[1]+14),(255,255,255),2)
        break
    # ret,thresh1 = cv2.threshold(gray,60,255,cv2.THRESH_BINARY)
    # cv2.imshow('frame',binm)
    if cv2.waitKey(10)==27:
        break
