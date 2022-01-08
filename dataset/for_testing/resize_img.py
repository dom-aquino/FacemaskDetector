import os
import numpy as np
from PIL import Image

for file in os.listdir("."):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
        try:
            img_in_np = np.array(Image.open(file).resize((200, 200)))
            Image.fromarray(img_in_np).save(file)
        except:
            print("ERROR CONVERTING: {}".format(file))


