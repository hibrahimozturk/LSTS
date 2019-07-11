import numpy as np
import cv2
import os
import shutil
import torch
import argparse


from src.i3dpt import I3D

def extract_frames(args):

    video_path = '../../dataset/DHF1K/video'
    data_path = '../../experiments/flow_val_train_fix/examples/input'
    
    
    for video_num in range(601,610):
        
        capture = cv2.VideoCapture(os.path.join(video_path, str(video_num).zfill(3))+'.AVI')
        i = 0
        
#         vid = []
        while True:
            ret, img = capture.read()
            if not ret:
                break
#             img = cv2.resize(img, (224,224))
#             vid.append(img)
            if i>=79:
                # Prepare I3D Input
                output_name = str(video_num).zfill(4)+ '_' + str(i+1).zfill(4)
                print(output_name)
                cv2.imwrite(os.path.join(data_path,output_name+'.jpg'), img)
                
            i += 1

def fix_wrong_calc():
    # frame numbers were calculated wrongly in training, the function fix
    data_path = '../dataset/DHF1K/train/data'
    fix_path = '/workspace/CMP_717/Project/Dataset/DHF1K/train/data_fix'
    target_path = '/workspace/CMP_717/Project/Dataset/DHF1K/train/target_fix'
    annotation_path = '../../Dataset/DHF1K/annotation'

#     data_paths = os.listdir(data_path)
    for path in os.listdir(data_path):
        video_num = int(path.split('_')[0])
        frame_index = int(path.split('_')[1].split('.')[0])
        shutil.copy(os.path.join(data_path, path),
            os.path.join(fix_path, str(video_num).zfill(4) + '_' +str(frame_index-1).zfill(4) )+'.npy')
        
        output_name = path.split('.')[0]
        shutil.copy(os.path.join(annotation_path, str(video_num).zfill(4), 'maps', str(frame_index-1).zfill(4))+'.png',
                    os.path.join(target_path, 'map' , str(video_num).zfill(4) + '_' +str(frame_index-1) )+'.png')
        shutil.copy(os.path.join(annotation_path, str(video_num).zfill(4), 'fixation', str(frame_index-1).zfill(4))+'.png',
                    os.path.join(target_path, 'fixation' ,  str(video_num).zfill(4) + '_' +str(frame_index-1) )+'.png')
        print(frame_index)
    return
            
def rgb_features(args):

    # Initialize I3D Network
#     i3d_rgb = I3D(num_classes=400, modality='rgb')
#     i3d_rgb.eval()
#     i3d_rgb.load_state_dict(torch.load(args.rgb_weights_path))
#     i3d_rgb.cuda()
    
    video_path = '../../dataset/DHF1K/video'
    data_path = '../../dataset/DHF1K/train/data_fix'
    target_path = '../../dataset/DHF1K/val/target_fix'
    annotation_path = '../../dataset/DHF1K/annotation'
    
    
    for video_num in range(501,601):
        
        capture = cv2.VideoCapture(os.path.join(video_path, str(video_num).zfill(3))+'.AVI')
        i = 0
        
        vid = []
        while True:
            ret, img = capture.read()
            if not ret:
                break
            img = cv2.resize(img, (224,224))
            vid.append(img)
            i+=1
            if i>=79:
                # Annotations start from 1, for this reason not i-1
                output_name = str(video_num).zfill(4)+ '_' + str(i).zfill(4)
                print(output_name)

                if not os.path.exists(os.path.join(data_path, output_name+'.npy')):
                    # Prepare I3D Input
                    video_clip = vid[i-79:i]
                    video_clip_np = np.array(video_clip, dtype='float32')
                    video_clip_np = np.interp(video_clip_np, (video_clip_np.min(), video_clip_np.max()), (-1, +1))
                    video_clip_np = np.expand_dims(video_clip_np, axis=0)
                    video_clip_np = np.transpose(video_clip_np, (0, 4, 1, 2, 3)).astype(np.float32)
 
                    # Get RGB Output
                    sample_var = torch.autograd.Variable(torch.from_numpy(video_clip_np).cuda())
                    out_var = i3d_rgb.feature_extract(sample_var)
                    out_tensor = out_var.data.cpu().numpy()
                     
                    # Save To Disk and Copy GroundTruths
                     
                    np.save(os.path.join(data_path, output_name), out_tensor)
                print(output_name)
                shutil.copy(os.path.join(annotation_path, str(video_num).zfill(4), 'maps', str(i).zfill(4))+'.png',
                            os.path.join(target_path, 'map' , output_name)+'.png')
                shutil.copy(os.path.join(annotation_path, str(video_num).zfill(4), 'fixation', str(i).zfill(4))+'.png',
                            os.path.join(target_path, 'fixation' , output_name)+'.png')
                 
            
            
            
def flow_extraction(args):

    video_path = '../dataset/DHF1K/video'
    flows_path = '../dataset/DHF1K/flows'
    
    for video_num in range(501,601):
        
        
        optical_flow = cv2.DualTVL1OpticalFlow.create()
        capture = cv2.VideoCapture(os.path.join(video_path, str(video_num).zfill(3))+'.AVI')
        i = 0
        
        flows = []
        while True:
            ret, img = capture.read()
            if not ret:
                print(str(video_num))
                flows_np = np.array(flows, dtype='float32')
                np.save(os.path.join(flows_path, str(video_num).zfill(3)), flows_np)
                break
            
            if i == 0:
                nextImg = cv2.resize(img, (112,112))
                nextImg = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)
            else:
                prevImg = nextImg
                nextImg = cv2.resize(img, (112,112))
                nextImg = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)
                flow = optical_flow.calc(prevImg, nextImg, None)
                flow = cv2.resize(flow, (224,224))
                flows.append(flow)
                print(str(video_num).zfill(3) + '_' + str(i).zfill(4))
            
            i+=1

                
            
def flow_features(args):

    # Initialize I3D Network
    i3d_flow = I3D(num_classes=400, modality='flow')
    i3d_flow.eval()
    i3d_flow.load_state_dict(torch.load(args.flow_weights_path))
    i3d_flow.cuda()
    
    data_path = '../dataset/DHF1K/test/flows_fix2'
    flows_path = '../dataset/DHF1K/flows'
    
    for video_num in range(601,701):
        
        video_flows = np.load( os.path.join( flows_path, str(video_num).zfill(3) + '.npy' ) )
        for idx in range(video_flows.shape[0]+1):
            if idx < 79:
                continue
            
            output_name = str(video_num).zfill(4)+ '_' + str(idx+1).zfill(4)
            
            flows = video_flows[idx-79:idx, :, :, :]
            flow_clip = np.expand_dims(flows, axis=0)
            flow_clip[flow_clip>20] = 20.
            flow_clip[flow_clip<-20] = -20. 
            flow_clip = np.interp(flow_clip, (flow_clip.min(), flow_clip.max()), (-1, +1))
            flow_clip = np.transpose(flow_clip, (0, 4, 1, 2, 3)).astype(np.float32)
            
            sample_var = torch.autograd.Variable(torch.from_numpy(flow_clip).cuda())
            out_var = i3d_flow.feature_extract(sample_var)
            out_tensor = out_var.data.cpu().numpy()

            # Save To Disk and Copy GroundTruths
            np.save(os.path.join(data_path, output_name), out_tensor)
            print(output_name)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Runs inflated inception v1 network on\
    cricket sample from tensorflow demo (generate the network weights with\
    i3d_tf_to_pt.py first)')

    # RGB arguments
    parser.add_argument(
        '--rgb', action='store_true', help='Evaluate RGB pretrained network')
    parser.add_argument(
        '--rgb_weights_path',
        type=str,
        default='model/model_rgb.pth',
        help='Path to rgb model state_dict')
    
    # Flow arguments
    parser.add_argument(
        '--flow', action='store_true', help='Evaluate flow pretrained network')
    parser.add_argument(
        '--flow_weights_path',
        type=str,
        default='model/model_flow.pth',
        help='Path to flow model state_dict')
    
    os.environ["CUDA_VISIBLE_DEVICES"]= '2'
    args = parser.parse_args()
#     rgb_features(args)
#     fix_wrong_calc()
#     flow_features(args)
    extract_frames(args)