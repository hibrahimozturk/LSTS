import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
import glob 
import cv2  
import os
from scipy import ndimage

def add(image, heat_map, alpha=0.6, display=False, save=None, cmap='viridis', axis='on', verbose=False):

    height = image.shape[0]
    width = image.shape[1]

    # resize heat map
    heat_map_resized = transform.resize(heat_map, (height, width))

    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)

    # display
    plt.axis('off')
    plt.imshow(image)
    fig = plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
#     plt.axis(axis)

    if display:
        plt.show()

    if save is not None:
        if verbose:
            print('save image: ' + save)
            plt.savefig(save, bbox_inches='tight', pad_inches=0)
            
            
def get_frames_smaps(vid_num, example_dir):
#     vid_num = '608'
    vid_array = []
    smap_array = []
    framenum_array = []
 
    print("Example dir: {}".format(example_dir))
    # Images and saliency maps were already generated and stored in experiment file
    for filename in glob.glob( os.path.join( example_dir, 'input/*' + str(vid_num) + '_*.jpg') ):
    #     print(filename.split('/')[-1])
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, layers = img.shape
        size = (width,height)
        vid_array.append(img)
    
    
    for filename in glob.glob( os.path.join( example_dir, 'estimated/*' + str(vid_num) + '_*.jpg') ):
    #     print(filename.split('/')[-1])
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, layers = img.shape
        size = (width,height)
        smap_array.append(img)
        
        filename = filename.split('/')[-1].split('_')[2].split('.')[0]
        framenum_array.append( int(filename) )
          
#     add(vid_array[0], img[:,:,0], save='/home/ibrahim/workspace/CMP_717/Project/Development/FCN-pytorch/python/exp.png')
    return vid_array, smap_array, framenum_array
if __name__ == '__main__':
    example_dir='../experiments/rgb_flow_fix/examples'
    vid_array,smap_array,framenum_array = get_frames_smaps(607, example_dir=example_dir)
    
    output_dir = os.path.join( example_dir, 'visualize', str(607) , 'output')
    input_dir = os.path.join( example_dir, 'visualize', str(607) , 'input')

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    
    for img,smap,framenum in zip(vid_array,smap_array,framenum_array):
        add(img, smap[:,:,0], save= os.path.join( output_dir, str(framenum) ), verbose=True)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite( os.path.join( input_dir, str(framenum) + '.jpg' ), img )
    
    print('Finished')