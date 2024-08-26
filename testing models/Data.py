import os
import random
from PIL import Image
from pathlib import Path
path_to_images = Path('F:/pk/images/refined/african-grey')
path_to_result_test = Path('F:/pk/images/used-in-split/test/african-grey')
path_to_result_train = Path('F:/pk/images/used-in-split/train/african-grey')
for i in os.listdir(path_to_images):
    if random.randint(1, 10) == 10:
        Image.open(os.path.join(path_to_images, i)).save(os.path.join(path_to_result_test, i + ".jpg"), "JPEG")
    else:
        Image.open(os.path.join(path_to_images, i)).save(os.path.join(path_to_result_train, i + ".jpg"), "JPEG")
