import cv2 as cv2
import numpy as np
import os
import time

# import matplotlib.pyplot as plt



image_path = "/home/work/Object_Detection_Yolo/Data/Raw/Tools"

save_path = "/home/work/Object_Detection_Yolo/Data/Training_Images/Tools"

# try: 
#     os.makedirs(save_path)
#     print("Preprocess image path created")
# except FileExistsError:
#     print("Image path already exists")

global i

i = os.listdir(image_path)

for x in os.listdir(image_path):
    
    img = cv2.imread(os.path.join(image_path,x),cv2.IMREAD_UNCHANGED)

    img = cv2.resize(img,(1920,1080))
    # kernal = np.array([[1, 1, 1],[1, 1, 1],[1, 1 ,1]])
    # img = cv2.filter2D(img,-1,kernal)
    img 

    cv2.imshow('test',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    break

