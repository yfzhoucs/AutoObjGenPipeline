import os
from joblib import Parallel, delayed
import cv2


def execute_blender(cmd):
    print(f'starting job {cmd.split()[3]}')
    os.system(cmd)
    print(f'finishing job {cmd.split()[3]}')


def blender_process(blender_path, rgb_folder, depth_folder, stl_folder, obj_folder, blender_process_version='1.0'):
    # folders = open('folders.txt', 'w')
    # folders.write(rgb_folder + '\n')
    # folders.write(depth_folder + '\n')
    # folders.write(stl_folder + '\n')
    # folders.write(obj_folder + '\n')
    # folders.close()

    if blender_process_version == '1.0':
        cmd = rf'{blender_path} --python ./blender_process.py'
        os.system(cmd)
    elif blender_process_version == '2.0':
        # folders = open('folders.txt', 'r')
        # rgb_folder = folders.readline().strip()
        # depth_folder = folders.readline().strip()
        # stl_folder = folders.readline().strip()
        # obj_folder = folders.readline().strip()
        # folders.close()

        rgb_files = os.listdir(rgb_folder)
        cmds = []
        for filename in rgb_files:
            depth_file = os.path.join(depth_folder, filename)
            rgb_file = os.path.join(rgb_folder, filename)

            W, H, _ = cv2.imread(depth_file).shape
            cmd = rf'{blender_path} --python ./blender_process_2.py -- "{depth_file}" "{rgb_file}" "{stl_folder}" "{obj_folder}" {W} {H}'
            cmds.append(cmd)
        Parallel(n_jobs=len(cmds))(delayed(execute_blender)(cmds[i]) for i in range(len(cmds)))
            
    elif blender_process_version == '3.0':
        rgb_files = os.listdir(rgb_folder)
        cmds = []
        for filename in rgb_files:
            depth_file = os.path.join(depth_folder, filename)
            rgb_file = os.path.join(rgb_folder, filename)

            W, H, _ = cv2.imread(depth_file).shape
            scale = (10000 / (W * H)) ** 0.5
            W = int(W * scale)
            H = int(H * scale)
            cmd = rf'{blender_path} --python ./blender_process_3.py -- "{depth_file}" "{rgb_file}" "{stl_folder}" "{obj_folder}" {W} {H}'
            cmds.append(cmd)
        # os.system(cmds[0])
        Parallel(n_jobs=len(cmds))(delayed(execute_blender)(cmds[i]) for i in range(len(cmds)))
    else:
        cmd = r'echo "blender process version not correct"'
        os.system(cmd)

