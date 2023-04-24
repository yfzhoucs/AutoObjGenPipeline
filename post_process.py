import os

results_folder = './results'

# categories = os.listdir(results_folder)
categories = [
    'lemon',
    'avocado',
    'strawberry',
    'apple',
    'orange'
]

for category in categories:
    os.mkdir(f'samples_smoothed_3/{category}')
    os.system(f'cp -r {os.path.join(results_folder, category, "blender_obj_smoothed_3/")} ./samples_smoothed_3/{category}')

    # if category == 'apple' or category == 'avocado' or category == 'banana' or category == 'bottle_of_drink':
    #     continue

    # if True:
    #     print(category)
        # # os.system(f'mv {os.path.join(results_folder, category, "final_rgb/")}* {os.path.join(results_folder, category, "blender_obj/")}')
        # os.system(f'cp {os.path.join(results_folder, category, "blender_obj/")}*.png {os.path.join(results_folder, category, "final_rgb/")}')
        
        # models = os.listdir(os.path.join(results_folder, category, "blender_obj/"))
        # models = [x for x in models if x.endswith('mtl')]
        # for model in models:
        #     print(model)
        #     f = open(os.path.join(results_folder, category, "blender_obj/", model))
        #     lines = f.readlines()
        #     f.close()
        #     start = lines[-1].split(' ')[0]
        #     end = lines[-1].split(r'/')[-1]
        #     lines[-1] = f'{start} {end} \n'

        #     f = open(os.path.join(results_folder, category, "blender_obj/", model), 'w')
        #     f.writelines(lines)
        #     f.close()

