from PIL import Image
import os
from pathlib import Path

# Example usage
input_folder = 'F:/pk/images/raw/guinea-pigs/adobe'
output_folder = 'F:/pk/images/refined'
size = (256, 256)


def is_usable_file_type(filename) -> bool:
    return filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))


def get_image_files(my_input_folder):
    """Get a list of image files in the input folder."""
    return list(filter(is_usable_file_type, os.listdir(my_input_folder)))

def resize_image(input_path, output_folder, my_size):
    """Resize an image from the input path to the specified size and save it to the output folder."""
    image_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_folder, f"{image_name}_{my_size[0]}x{my_size[1]}.jpg")
    print(f"Resizing image: {input_path}")
    image = Image.open(input_path)
    image = image.convert("RGB").resize(my_size, Image.LANCZOS)
    image.save(output_path, "JPEG")

def directory_resize_images(my_input_folder, my_output_folder, my_size):
    """Resize all images in the input folder and save them to the output folder."""
    if not os.path.exists(my_input_folder):
        raise RuntimeError("Could not find input-folder")

    if not os.path.exists(my_output_folder):
        os.makedirs(my_output_folder)
        print("Could not find output-folder, creating a new one")

    print(f"found {len(get_image_files(my_input_folder))} usable images")

    list(map(lambda image_path: resize_image(Path(os.path.join(my_input_folder, image_path)), my_output_folder, my_size), get_image_files(my_input_folder)))


directory_resize_images(input_folder, output_folder, size)
