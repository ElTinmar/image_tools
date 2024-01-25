import cv2
from numpy.typing import NDArray
import numpy as np
from .convert import im2gray

def polyroi(img: NDArray) -> NDArray:

    local_image = img.copy()
    coords = []

    def click_event(event, x, y, flags, params): 
        nonlocal coords, local_image, img

        # checking for left mouse clicks 
        if event == cv2.EVENT_LBUTTONDOWN: 
            coords.append((x,y))
    
        # checking for right mouse clicks      
        if event==cv2.EVENT_RBUTTONDOWN: 
            coords.pop()

        pts = np.array(coords, np.int32)
        pts = pts.reshape((-1,1,2))
        
        # it looks like cv2.polylines modifies input inplace
        # so I make a copy
        original = img.copy() 
        local_image = cv2.polylines(original, [pts], True, (0, 0, 255), 2)
        cv2.imshow('image', local_image) 

    cv2.imshow('image', local_image) 
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0) 
    cv2.destroyWindow('image') 
    
    return np.array(coords, np.int32)

def polymask(img: NDArray) -> NDArray:
    mask = np.zeros_like(img)
    coords = polyroi(img)
    mask_RGB = cv2.fillPoly(mask, [coords], 255)
    return im2gray(mask_RGB)

