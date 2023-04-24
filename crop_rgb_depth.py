import cv2
import numpy as np
import os
import torch, detectron2
print(detectron2.__version__)

import traceback


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import cv2
from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth

def crop_rgb_depth(dalle_folder, depth_folder, cropped_dalle_folder, cropped_depth_folder):

    dalle_files = os.listdir(dalle_folder)
    objs = [dalle_files[i].split('.')[0] for i in range(len(dalle_files))]
    midas_files = [os.path.join(depth_folder, objs[i] + '-dpt_beit_large_512.png') for i in range(len(dalle_files))]
    dalle_files = [os.path.join(dalle_folder, dalle_files[i]) for i in range(len(dalle_files))]
    cropped_dalle_files = [os.path.join(cropped_dalle_folder, objs[i]+'.png') for i in range(len(dalle_files))]
    cropped_midas_files = [os.path.join(cropped_depth_folder, objs[i]+'.png') for i in range(len(dalle_files))]
    print(midas_files)
    print(dalle_files)
    print(cropped_dalle_files)
    print(cropped_midas_files)

    for i in range(len(dalle_files)):
        # read the depth map image
        depth_map = cv2.imread(midas_files[i])
        apple_img = cv2.imread(dalle_files[i])

        # convert the depth map image to grayscale
        gray = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)


        # Apply Gaussian filter to the image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detector to the image
        edges = cv2.Canny(blur, 10, 100)


        # apply thresholding to segment the object
        _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        # find the contour of the object
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        # draw the bounding rectangle on the object
        x,y,w,h = cv2.boundingRect(contours[0])


        # Get the largest contour (assuming it is the object we want to crop)
        max_contour = max(contours, key=cv2.contourArea)

        # Draw the contour on the input image (optional)
        cv2.drawContours(gray, [max_contour], -1, (0, 255, 0), 2)

        # crop the depth map image using the bounding rectangle coordinates
        mask = np.zeros((depth_map.shape))
        mask = cv2.fillConvexPoly(mask,max_contour,(1,1,1))

        # masked = depth_map * mask
        masked = np.where(mask == 1,depth_map,0)
        masked_rgb = np.where(mask == 1, apple_img, 255)
        # gray = np.where(mask[:,:,0] == 1,gray,0)

        cropped = gray[y:y+h, x:x+w]
        cropped_rgb = masked_rgb[y:y+h, x:x+w]



        minval = np.min(cropped[cropped > 80])
        maxval = np.max(cropped[cropped > 80])
        mintar = 0
        maxtar = 255
        cropped = (np.float32(cropped) - minval) / (maxval - minval) * (maxtar - mintar)
        cropped[cropped < 0] = 0
        cropped[cropped > 255] = 255

        cropped = np.where(mask[y:y+h, x:x+w, 0] == 1,cropped,0)

        cropped = np.uint8(cropped)


        cv2.imwrite(cropped_dalle_files[i], cropped_rgb)
        cv2.imwrite(cropped_midas_files[i], cropped)


def crop_rgb_depth_2(dalle_folder, depth_folder, cropped_dalle_folder, cropped_depth_folder):

    dalle_files = os.listdir(dalle_folder)
    objs = [dalle_files[i].split('.')[0] for i in range(len(dalle_files))]
    midas_files = [os.path.join(depth_folder, objs[i] + '-dpt_beit_large_512.png') for i in range(len(dalle_files))]
    dalle_files = [os.path.join(dalle_folder, dalle_files[i]) for i in range(len(dalle_files))]
    cropped_dalle_files = [os.path.join(cropped_dalle_folder, objs[i]+'.png') for i in range(len(dalle_files))]
    cropped_midas_files = [os.path.join(cropped_depth_folder, objs[i]+'.png') for i in range(len(dalle_files))]
    print(midas_files)
    print(dalle_files)
    print(cropped_dalle_files)
    print(cropped_midas_files)

    # # Inference with a panoptic segmentation model
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    # predictor = DefaultPredictor(cfg)

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    for i in range(len(dalle_files)):
        # read the depth map image
        depth_map = cv2.imread(midas_files[i])
        apple_img = cv2.imread(dalle_files[i])
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)


        ######################################## Object Detection and Segmentation #################################################
        # cfg = get_cfg()
        # # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # predictor = DefaultPredictor(cfg)

        # while True:
        #     ret, im = cam.read()
        #     outputs = predictor(im)

        #     print(outputs["instances"].pred_classes)
        #     print(outputs["instances"].pred_boxes)

        #     v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #     cv2.imshow('pred', out.get_image()[:, :, ::-1])
        #     cv2.waitKey(1)
        ######################################## Object Detection and Segmentation #################################################

        try:
            outputs = predictor(apple_img)


            print(outputs["instances"].pred_classes)
            print(outputs["instances"].pred_boxes)
            print(outputs['instances'].pred_masks)

            mask = outputs['instances'].pred_masks.detach().cpu().numpy()[0]
            # for chnl_idx in range(depth_map.shape[2]):
            #     depth_map[:,:,chnl_idx] = depth_map[:,:,chnl_idx] * mask
            
            depth_map[:,:] = depth_map[:,:] * mask
            for chnl_idx in range(apple_img.shape[2]):
                apple_img[:,:,chnl_idx] = apple_img[:,:,chnl_idx] * mask
            box = outputs["instances"].pred_boxes.tensor.detach().cpu().numpy()[0]
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            depth_map_cropped = depth_map[y1:y2, x1:x2]
            apple_img_cropped = apple_img[y1:y2, x1:x2]





            depth_map_cropped_mean_shift = depth_map_cropped.reshape(-1)
            depth_map_cropped_mean_shift = depth_map_cropped_mean_shift[depth_map_cropped_mean_shift>0]
            # depth_map_cropped_mean_shift = np.sort(depth_map_cropped_mean_shift).reshape(-1, 1)
            # ms = MeanShift(bandwidth=None, bin_seeding=True)
            # ms.fit(depth_map_cropped_mean_shift)
            # print(ms.labels_)
            # print(depth_map_cropped_mean_shift)

            depth_map_cropped_mean_shift = np.sort(depth_map_cropped_mean_shift).reshape(-1)
            threshold = depth_map_cropped_mean_shift[len(depth_map_cropped_mean_shift) // 10]


            minval = np.min(depth_map_cropped[depth_map_cropped > threshold])
            maxval = np.max(depth_map_cropped[depth_map_cropped > threshold])
            mintar = 0
            maxtar = 255
            depth_map_cropped = (np.float32(depth_map_cropped) - minval) / (maxval - minval) * (maxtar - mintar)
            depth_map_cropped[depth_map_cropped < 0] = 0
            depth_map_cropped[depth_map_cropped > 255] = 255

            depth_map_cropped = np.uint8(depth_map_cropped)


            cv2.imwrite(cropped_dalle_files[i], apple_img_cropped)
            cv2.imwrite(cropped_midas_files[i], depth_map_cropped)
        except:
            continue

        # cv2.imshow('cropped_depth', depth_map_cropped)
        # cv2.imshow('cropped_rgb', apple_img_cropped)
        # cv2.waitKey(-1)

        # v = Visualizer(apple_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow('pred', out.get_image()[:, :, ::-1])
        # cv2.waitKey(-1)

        # panoptic_seg, segments_info = predictor(apple_img)["panoptic_seg"]

        # v = Visualizer(apple_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        # cv2.imshow('pred', out.get_image()[:, :, ::-1])
        # cv2.waitKey(-1)

        # panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)




def crop_rgb_depth_3(dalle_folder, depth_folder, cropped_dalle_folder, cropped_depth_folder):

    dalle_files = os.listdir(dalle_folder)
    objs = [dalle_files[i].split('.')[0] for i in range(len(dalle_files))]
    midas_files = [os.path.join(depth_folder, objs[i] + '-dpt_beit_large_512.png') for i in range(len(dalle_files))]
    dalle_files = [os.path.join(dalle_folder, dalle_files[i]) for i in range(len(dalle_files))]
    cropped_dalle_files = [os.path.join(cropped_dalle_folder, objs[i]+'.png') for i in range(len(dalle_files))]
    cropped_midas_files = [os.path.join(cropped_depth_folder, objs[i]+'.png') for i in range(len(dalle_files))]
    print(midas_files)
    print(dalle_files)
    print(cropped_dalle_files)
    print(cropped_midas_files)

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    for i in range(len(dalle_files)):
        # read the depth map image
        depth_map = cv2.imread(midas_files[i])
        apple_img = cv2.imread(dalle_files[i])
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)


        # cv2.imshow('rgb_raw', apple_img)
        # cv2.imshow('depth_raw', depth_map)
        # cv2.waitKey(-1)

        try:
            outputs = predictor(apple_img)


            print(outputs["instances"].pred_classes)
            print(outputs["instances"].pred_boxes)
            print(outputs['instances'].pred_masks)

            mask = outputs['instances'].pred_masks.detach().cpu().numpy()[0]
            
            depth_map[:,:] = depth_map[:,:] * mask
            for chnl_idx in range(apple_img.shape[2]):
                # apple_img[:,:,chnl_idx] = apple_img[:,:,chnl_idx] * mask + 255 * np.ones(apple_img[:,:,chnl_idx].shape) * (1 - mask)
                apple_img[:,:,chnl_idx] = apple_img[:,:,chnl_idx] * mask
            box = outputs["instances"].pred_boxes.tensor.detach().cpu().numpy()[0]


            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            radias_x_depth = (x2 - x1) / 2 * 1.2
            radias_y_depth = (y2 - y1) / 2 * 1.2
            radias_x_rgb = (x2 - x1) / 2 * 1.1
            radias_y_rgb = (y2 - y1) / 2 * 1.1
            x1_depth = int(max(center_x - radias_x_depth, 0))
            x2_depth = int(min(center_x + radias_x_depth, depth_map.shape[1]))
            y1_depth = int(max(center_y - radias_y_depth, 0))
            y2_depth = int(min(center_y + radias_y_depth, depth_map.shape[0]))
            x1_rgb = int(max(center_x - radias_x_rgb, 0))
            x2_rgb = int(min(center_x + radias_x_rgb, depth_map.shape[1]))
            y1_rgb = int(max(center_y - radias_y_rgb, 0))
            y2_rgb = int(min(center_y + radias_y_rgb, depth_map.shape[0]))



            print(x1, x2, y1, y2)
            depth_map_cropped = depth_map[y1_depth:y2_depth, x1_depth:x2_depth]
            apple_img_cropped = apple_img[y1_rgb:y2_rgb, x1_rgb:x2_rgb]

            depth_map_cropped_mean_shift = depth_map_cropped.reshape(-1)
            depth_map_cropped_mean_shift = depth_map_cropped_mean_shift[depth_map_cropped_mean_shift>0]

            depth_map_cropped_mean_shift = np.sort(depth_map_cropped_mean_shift).reshape(-1)
            threshold = depth_map_cropped_mean_shift[len(depth_map_cropped_mean_shift) // 14]

            minval = np.min(depth_map_cropped[depth_map_cropped > threshold])
            maxval = np.max(depth_map_cropped[depth_map_cropped > threshold])
            mintar = 0
            maxtar = 255
            depth_map_cropped = (np.float32(depth_map_cropped) - minval) / (maxval - minval) * (maxtar - mintar)
            depth_map_cropped[depth_map_cropped < 0] = 0
            depth_map_cropped[depth_map_cropped > 255] = 255

            depth_map_cropped = np.uint8(depth_map_cropped)

            # cv2.imshow('rgb', apple_img_cropped)
            # cv2.imshow('depth', depth_map_cropped)
            # cv2.waitKey(-1)


            cv2.imwrite(cropped_dalle_files[i], apple_img_cropped)
            cv2.imwrite(cropped_midas_files[i], depth_map_cropped)
        except Exception as e: 
            traceback.print_exc()


def postprocess_depth(cropped_depth_folder):

    dalle_files = os.listdir(cropped_depth_folder)
    objs = [dalle_files[i].split('.')[0] for i in range(len(dalle_files))]
    cropped_midas_files = [os.path.join(cropped_depth_folder, objs[i]+'.png') for i in range(len(dalle_files))]

    print(dalle_files)
    print(cropped_midas_files)


    for i in range(len(dalle_files)):
        # read the depth map image
        depth_map = cv2.imread(cropped_midas_files[i])
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)


        # cv2.imshow('rgb_raw', apple_img)
        # cv2.imshow('depth_raw', depth_map)
        # cv2.waitKey(-1)

        try:
            depth_map_cropped = depth_map
            depth_map_cropped_mean_shift = depth_map_cropped.reshape(-1)
            depth_map_cropped_mean_shift = depth_map_cropped_mean_shift[depth_map_cropped_mean_shift>0]

            depth_map_cropped_mean_shift = np.sort(depth_map_cropped_mean_shift).reshape(-1)
            threshold = depth_map_cropped_mean_shift[len(depth_map_cropped_mean_shift) // 120]
            # threshold = 2

            minval = np.min(depth_map_cropped[depth_map_cropped > threshold])
            maxval = np.max(depth_map_cropped[depth_map_cropped > threshold])
            mintar = 0
            maxtar = 255
            depth_map_cropped = (np.float32(depth_map_cropped) - minval) / (maxval - minval) * (maxtar - mintar)
            depth_map_cropped[depth_map_cropped < 0] = 0
            depth_map_cropped[depth_map_cropped > 255] = 255

            depth_map_cropped = np.uint8(depth_map_cropped)

            # cv2.imshow('rgb', apple_img_cropped)
            # cv2.imshow('depth', depth_map_cropped)
            # cv2.waitKey(-1)

            cv2.imwrite(cropped_midas_files[i], depth_map_cropped)
        except Exception as e: 
            traceback.print_exc()





def only_crop_rgb_depth_3(dalle_folder, depth_folder, cropped_dalle_folder, cropped_depth_folder):

    dalle_files = os.listdir(dalle_folder)
    objs = [dalle_files[i].split('.')[0] for i in range(len(dalle_files))]
    midas_files = [os.path.join(depth_folder, objs[i] + '-dpt_beit_large_512.png') for i in range(len(dalle_files))]
    dalle_files = [os.path.join(dalle_folder, dalle_files[i]) for i in range(len(dalle_files))]
    cropped_dalle_files = [os.path.join(cropped_dalle_folder, objs[i]+'.png') for i in range(len(dalle_files))]
    cropped_midas_files = [os.path.join(cropped_depth_folder, objs[i]+'.png') for i in range(len(dalle_files))]
    print(midas_files)
    print(dalle_files)
    print(cropped_dalle_files)
    print(cropped_midas_files)

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    for i in range(len(dalle_files)):
        # read the depth map image
        depth_map = cv2.imread(midas_files[i])
        apple_img = cv2.imread(dalle_files[i])
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)


        # cv2.imshow('rgb_raw', apple_img)
        # cv2.imshow('depth_raw', depth_map)
        # cv2.waitKey(-1)

        try:
            outputs = predictor(apple_img)


            print(outputs["instances"].pred_classes)
            print(outputs["instances"].pred_boxes)
            print(outputs['instances'].pred_masks)

            mask = outputs['instances'].pred_masks.detach().cpu().numpy()[0]
            
            # depth_map[:,:] = depth_map[:,:] * mask
            # for chnl_idx in range(apple_img.shape[2]):
            #     # apple_img[:,:,chnl_idx] = apple_img[:,:,chnl_idx] * mask + 255 * np.ones(apple_img[:,:,chnl_idx].shape) * (1 - mask)
            #     apple_img[:,:,chnl_idx] = apple_img[:,:,chnl_idx] * mask
            box = outputs["instances"].pred_boxes.tensor.detach().cpu().numpy()[0]


            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            radias_x_depth = (x2 - x1) / 2 * 1.4
            radias_y_depth = (y2 - y1) / 2 * 1.4
            radias_x_rgb = (x2 - x1) / 2 * 1.2
            radias_y_rgb = (y2 - y1) / 2 * 1.2
            x1_depth = int(max(center_x - radias_x_depth, 0))
            x2_depth = int(min(center_x + radias_x_depth, depth_map.shape[1]))
            y1_depth = int(max(center_y - radias_y_depth, 0))
            y2_depth = int(min(center_y + radias_y_depth, depth_map.shape[0]))
            x1_rgb = int(max(center_x - radias_x_rgb, 0))
            x2_rgb = int(min(center_x + radias_x_rgb, depth_map.shape[1]))
            y1_rgb = int(max(center_y - radias_y_rgb, 0))
            y2_rgb = int(min(center_y + radias_y_rgb, depth_map.shape[0]))



            print(x1, x2, y1, y2)
            depth_map_cropped = depth_map[y1_depth:y2_depth, x1_depth:x2_depth]
            apple_img_cropped = apple_img[y1_rgb:y2_rgb, x1_rgb:x2_rgb]

            depth_map_cropped_mean_shift = depth_map_cropped.reshape(-1)
            depth_map_cropped_mean_shift = depth_map_cropped_mean_shift[depth_map_cropped_mean_shift>0]

            depth_map_cropped_mean_shift = np.sort(depth_map_cropped_mean_shift).reshape(-1)
            threshold = depth_map_cropped_mean_shift[len(depth_map_cropped_mean_shift) // 20]

            minval = np.min(depth_map_cropped[depth_map_cropped > threshold])
            maxval = np.max(depth_map_cropped[depth_map_cropped > threshold])
            mintar = 0
            maxtar = 255
            depth_map_cropped = (np.float32(depth_map_cropped) - minval) / (maxval - minval) * (maxtar - mintar)
            depth_map_cropped[depth_map_cropped < 0] = 0
            depth_map_cropped[depth_map_cropped > 255] = 255

            depth_map_cropped = np.uint8(depth_map_cropped)

            # cv2.imshow('rgb', apple_img_cropped)
            # cv2.imshow('depth', depth_map_cropped)
            # cv2.waitKey(-1)


            cv2.imwrite(cropped_dalle_files[i], apple_img_cropped)
            cv2.imwrite(cropped_midas_files[i], depth_map_cropped)
        except Exception as e: 
            traceback.print_exc()
