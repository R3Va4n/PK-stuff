import os

path = os.path.dirname(__file__)

def print_dir(dir_path, only_images = False):
    image_formats = (".jpg", ".jpeg", ".ico", ".png", ".PNG", ".webp")
    for i in os.listdir(dir_path):
        if not only_images or os.path.splitext(i)[1] in image_formats:
            print(i)
print_dir(path, True)