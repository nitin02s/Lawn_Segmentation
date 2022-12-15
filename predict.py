import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
import cv2
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour

def write_video(orig,final_img,output_orig,output_seg):
    output_orig.write(orig)
    output_seg.write(final_img)

def get_canny_coor(edges):
    indices = np.where(edges != 0)
    coordinates = zip(indices[0], indices[1])
    return coordinates

def active_contour_det(img,snake_coor):
    snake_coor=np.array(snake_coor)
    snake = active_contour(img,
                       np.array(snake_coor).T, alpha=0.015, beta=10, gamma=0.001) 
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(snake_coor[:, 1], snake_coor[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()
                      

def feature(orig_img,img,edge,prev_frame,curr_frame):
    orb = cv2.ORB_create()
    seg_img = np.array(img,copy=True)
    first_img = np.array(orig_img,copy=True)
    kp1, des1 = orb.detectAndCompute(prev_frame,None)
    kp2, des2 = orb.detectAndCompute(curr_frame,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    for i in matches[:200]:
        x_prev,y_prev=kp1[i.queryIdx].pt
        x_curr,y_curr=kp2[i.trainIdx].pt
        x_prev,y_prev = int(x_prev),int(y_prev)
        x_curr,y_curr = int(x_curr),int(y_curr)
        if seg_img[x_prev][y_prev] == 255:
            seg_img[x_curr-4:x_curr+4,y_curr-4:y_curr+4] = 255
        elif edge[x_prev][y_prev] == 255:
            edge[x_curr-3:x_curr+3,y_curr-3:y_curr+3] = 255
        first_img[seg_img==255]=(0,255,0)    
        first_img[edge==255]=(255,0,0)
    cv2.imshow("circle",first_img)


def boundary(seg_img,orig_img,blur,prev_frame,curr_frame):
    thresh = 80
    if blur:
        
        temp_image = np.zeros((orig_img.shape), dtype=np.uint8)
        im_bw = cv2.threshold(seg_img, thresh, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(seg_img, kernel, iterations=2)
        closing = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel)
        blur = cv2.blur(closing,(5,5))                   
        median_blur=cv2.medianBlur(blur,5)                 
        gaus_blur= cv2.GaussianBlur(median_blur,(5,5),0)       
        smooth= cv2.bilateralFilter(gaus_blur,9,75,75)    
        img_erode = cv2.erode(smooth,(9,9),iterations=4)
    else:
        kernel = np.ones((3, 3), np.uint8)
        img_dilation = cv2.dilate(seg_img, kernel, iterations=2)
        blur = cv2.blur(img_dilation,(5,5))          
        median_blur=cv2.medianBlur(blur,5)                 
        gaus_blur= cv2.GaussianBlur(median_blur,(5,5),0)       
        smooth= cv2.bilateralFilter(gaus_blur,9,75,75)    
        img_erode = cv2.erode(smooth,(3,3),iterations=4)

    im_bw = cv2.threshold(img_erode, thresh, 255, cv2.THRESH_BINARY)[1]
    edge = cv2.Canny(im_bw, 50,200)
    # snake=np.array(list(get_canny_coor(edge)))
    # active_contour_det(orig_img,snake)
    # snake = get_canny_coor(snake)
    # contour_snake=cv2.drawContours(orig_img, snake, -1, (255,0,0), 3)
    # feature(orig_img,im_bw,edge_bw,prev_frame,curr_frame)
    contours, hierarchy = cv2.findContours(edge, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_1=cv2.drawContours(edge, contours, -1, (255,0,0), 3)
    contour=cv2.drawContours(orig_img, contours, -1, (255,0,0), 3)
    return contour,im_bw

def preprocess(pil_img, scale, is_mask):
        img_ndarray = np.asarray(pil_img)
        # img_ndarray = np.resize(img_ndarray,(540,500,3))
        # print(img_ndarray.shape)

        # if is_mask:
        #     data = np.array(img_ndarray)
        #     data[data!=0] = 1
        #     img_ndarray = np.array(data,copy=True)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))
            img_ndarray = img_ndarray / 255

        return img_ndarray

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()

def predict_img_2(net,
                full_img,
                device,
                scale_factor=0.5,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear',align_corners=True)
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].cpu().long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='weights\checkpoint_epoch5.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default=['test_images\lawn_1.jpg','test_images\lawn_3.jpg','test_images\lawn_4.jpg','test_images\lawn_5.jpg'], metavar='FILE', nargs='+', help='Filenames of input images present in test_images directory')
    # parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    # parser.add_argument('--viz', '-v', action='store_true',
    #                     help='Visualize the images as they are processed')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    input = 0
    # in_files = args.input
    # out_files = get_output_filenames(args)

    #image directory
    img_dir = args.input
    #model directory
    model_dir = args.model
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model_dir}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(model_dir, map_location=device))

    logging.info('Model loaded!')
    
    scale = 1
    mask_threshold = 0.5
    i=0
    if input ==1:
        cap = cv2.VideoCapture(r'D:\Repo\Pytorch-UNet\weights\more_grass\IMG-4948.MOV')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_orig = cv2.VideoWriter(r'D:\Repo\Pytorch-UNet\weights\more_grass\video\output_orig_1.mp4', fourcc, 30, (540,550))
        output_seg = cv2.VideoWriter(r'D:\Repo\Pytorch-UNet\weights\more_grass\video\output_seg_1.mp4', fourcc, 30, (540,550))
 
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _,prev_frame = cap.read() 
        prev_frame=cv2.resize(prev_frame,(540,550))

        while(cap.isOpened()):
            i+=1
            cap.set(cv2.CAP_PROP_POS_FRAMES, i) 
            ret, frame = cap.read()
            if ret == True:
                converted = cv2.resize(frame,(540,550))
                orig = np.array(converted,copy=True)
                med_blur=cv2.medianBlur(converted,5)                 
                gaus_blur= cv2.GaussianBlur(med_blur,(5,5),0)
                img = Image.fromarray(gaus_blur)
                mask = predict_img_2(net=net,
                            full_img=img,
                            scale_factor=scale,
                            out_threshold=mask_threshold,
                            device=device)
                result = mask_to_image(mask)
                disp = np.array(result)
                converted[disp==255]=(0,255,0)
                final_img, thresh_img = boundary(disp,converted,True,None,None)
                prev_frame=orig
                final_img[thresh_img==255] = (0,255,0)
                # write_video(orig,final_img,output_orig,output_seg)
                cv2.imshow('thres _ final',final_img)
                cv2.imshow("orig",orig)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else: 
                break
        
        cap.release()
        cv2.destroyAllWindows()   
    else:     
        # img_dir = r"test_images\lawn_1.jpg"    
        for i, filename in enumerate(img_dir): 
            img = Image.open(filename)     
            orig = np.array(img)
            mask = predict_img_2(net=net,

                                full_img=img,
                                scale_factor=0.5,
                                out_threshold=0.5,
                                device=device)
            # result = mask_to_image(mask)
            # plot_img_and_mask(img, mask)
            result = mask_to_image(mask)
            out = np.array(result) 
            out=cv2.resize(out,(540,500))
            converted = np.array(img)
            converted = cv2.cvtColor(converted,cv2.COLOR_BGR2RGB)
            converted=cv2.resize(converted,(540,500))
            converted[out==255]=(0,255,0)
            final_img, thresh_img = boundary(out,converted,False,None,None)
            final_img[thresh_img==255] = (0,255,0)
            plot_img_and_mask(img,mask,final_img)
            # plot_img_and_mask(img, mask,final_img)
            # cv2.imshow('final result',final_img)

            # cv2.waitKey(0)    
            # cv2.destroyAllWindows()    

