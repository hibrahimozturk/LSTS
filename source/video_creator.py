import cv2
import numpy as np
import glob
from numpy import dtype

vid_num = '608'
vid_array = []
for filename in glob.glob('../experiments/26_05_2019_smaller_lr/examples/input/*' + vid_num + '_*.jpg'):
#     print(filename.split('/')[-1])
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    vid_array.append(img)


img_array = []
for filename in glob.glob('../experiments/26_05_2019_smaller_lr/examples/estimated/*' + vid_num + '_*.jpg'):
#     print(filename.split('/')[-1])
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    
output_array = []
for i in range(len(vid_array)):
    vid_frame = np.array(vid_array[i], dtype='float32')
    smap = np.array(img_array[i], dtype='float32')
    smap = np.interp(smap, (smap.min(), smap.max()), (0.2, 1))
    output_array.append(smap*vid_frame)
    

 
out = cv2.VideoWriter('output_videos/' + vid_num + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(output_array)):
    out.write(output_array[i].astype(np.uint8))
out.release()