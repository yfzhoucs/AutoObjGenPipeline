target_object = 'teapot'
n = 10
batch_idx = 'teapot'
dalle_folder = f'./results/{batch_idx}/dalle_output'
depth_folder = f'./results/{batch_idx}/midas_output'
cropped_dalle_folder = f'./results/{batch_idx}/final_rgb'
cropped_depth_folder = f'./results/{batch_idx}/final_depth'
stl_folder = f'./results/{batch_idx}/blender_stl'
obj_folder = f'./results/{batch_idx}/blender_obj'
blender_path = '/home/local/ASUAD/yzhou298/Downloads/blender-3.3.0-linux-x64/blender'

from query_dalle import query_dalle
from run_midas import run as query_midas
from crop_rgb_depth import crop_rgb_depth, crop_rgb_depth_2
from blender_caller import blender_process

import os
os.mkdir(f'./results/{batch_idx}/')
os.mkdir(dalle_folder)
os.mkdir(depth_folder)
os.mkdir(cropped_dalle_folder)
os.mkdir(cropped_depth_folder)
os.mkdir(stl_folder)
os.mkdir(obj_folder)


query_dalle(dalle_folder, target_object, n)
query_midas(dalle_folder, depth_folder, 'dpt_beit_large_512', model_type='dpt_beit_large_512')
crop_rgb_depth_2(dalle_folder, depth_folder, cropped_dalle_folder, cropped_depth_folder)

input('Please confirm the wanted cropped rgb files in ./final_rgb folder')
blender_process(blender_path, cropped_dalle_folder, cropped_depth_folder, stl_folder, obj_folder)