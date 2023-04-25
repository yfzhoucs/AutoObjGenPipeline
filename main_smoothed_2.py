target_object = 'donut'
n = 10
batch_idx = 'donut_3'
dalle_folder = f'./results/{batch_idx}/dalle_output'
depth_folder = f'./results/{batch_idx}/midas_output_3'
cropped_dalle_folder = f'./results/{batch_idx}/final_rgb'
cropped_depth_folder = f'./results/{batch_idx}/final_depth'
cropped_smoothed_depth_folder = f'./results/{batch_idx}/final_depth_smoothed_3'
stl_folder = f'./results/{batch_idx}/blender_stl'
# obj_folder = f'./results/{batch_idx}/blender_obj_smoothed_3'
obj_folder = f'./results/{batch_idx}/blender_obj_smoothed_3_wo_collapse'
blender_path = '/home/slocal/blender-3.3.5-linux-x64/blender'
blender_process_version = '3.0'

from query_dalle import query_dalle
from run_midas import run as query_midas
from crop_rgb_depth import crop_rgb_depth, crop_rgb_depth_2, crop_rgb_depth_3, postprocess_depth, only_crop_rgb_depth_3
from blender_caller import blender_process
from smoothing import smooth_folder

import os
if not os.path.isdir(f'./results/{batch_idx}/'):
    os.mkdir(f'./results/{batch_idx}/') 
if not os.path.isdir(dalle_folder):
    os.mkdir(dalle_folder) 
if not os.path.isdir(depth_folder):
    os.mkdir(depth_folder)
if not os.path.isdir(cropped_dalle_folder):
    os.mkdir(cropped_dalle_folder) 
if not os.path.isdir(cropped_depth_folder):
    os.mkdir(cropped_depth_folder) 
if not os.path.isdir(obj_folder):
    os.mkdir(obj_folder)


# query_dalle(dalle_folder, target_object, n)
# query_midas(dalle_folder, depth_folder, 'dpt_beit_large_512', model_type='dpt_beit_large_512', grayscale=True)
# crop_rgb_depth_3(dalle_folder, depth_folder, cropped_dalle_folder, cropped_depth_folder)
blender_process(blender_path, cropped_dalle_folder, cropped_depth_folder, stl_folder, obj_folder, blender_process_version)

# only_crop_rgb_depth_3(dalle_folder, depth_folder, cropped_dalle_folder, cropped_depth_folder)
# postprocess_depth(cropped_depth_folder)

