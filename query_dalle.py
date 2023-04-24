import openai
import requests
import os


def query_dalle(dalle_output_folder, target_object, n):

    num_batches = (n - 1) // 10 + 1
    last_batch = n % 10

    for batch_idx in range(num_batches):
        if batch_idx < num_batches:
            batch_size = 10
        else:
            batch_size = last_batch
        response = openai.Image.create(
            # prompt=f"A photo of a {target_object}. Only a {target_object} in the center of the image. The whole body of the {target_object} is seeable in the image. Natural {target_object} color. No any background at all. The {target_object} can be fully seen. Every part of it can be seen perfectly. No other objects in the image. Looked from the side. Exact side view. Bottom bun of the burger is larger Show the bottom bun more. The {target_object} should take up the whole image. Pure white background.",
            prompt=f"A photo of a plain classic {target_object}. Only a {target_object} in the center of the image. The whole body of the {target_object} is seeable in the image. Natural {target_object} color. No any background at all. The {target_object} can be fully seen. Every part of it can be seen perfectly. No other objects in the image. Looked from the top. Exact top view. The {target_object} should take up the whole image. Pure white background.",
            # prompt=f"A photo of an water bottle. The water bottle is opaque. It is either metal or ceramic, or colorful plastic. Only the bottle in the center of the image. No any background at all. The water bottle can be fully seen. Every part of it can be seen perfectly. No other objects in the image. Looked from the front. Exact front view. The water bottle should take up the whole image. Pure white background.",
            # prompt=f"Create a detailed side view illustration of a scrumptious dessert cake, showcasing its various layers and textures, highlighting the richness and deliciousness of the cake.",
            n=batch_size,
            size="512x512"
            )

        for i in range(batch_size):
            global_idx = 10 * batch_idx + i
            image_url = response['data'][i]['url']
            print(global_idx, image_url)

            img_data = requests.get(image_url).content
            img_name = os.path.join(dalle_output_folder, f'{target_object}_{global_idx}_5.png')
            with open(img_name, 'wb') as handler:
                handler.write(img_data)