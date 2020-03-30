from recursivePhoto.helpers.interface_helpers import *
from recursivePhoto.helpers.image_processing import *

def get_user_input():
    raw_photo = get_photo()
    images_per_row, subimage_width = get_recursive_photo_params()
    return raw_photo, images_per_row, subimage_width

def create_photo(raw_photo, images_per_row, subimage_width):
    print("Step 1 of 1: Creating Image")
    grayscale_img = convert_img_to_grayscale(raw_photo)
    block_averages = get_block_averages_no_square(grayscale_img, images_per_row)
    averaged_images = create_averaged_imagesV1(grayscale_img, block_averages, subimage_width)
    result = stitch_images(averaged_images)
    return result

def recursivePhotoV1():
    raw_photo, images_per_row, subimage_width = get_user_input()
    result = create_photo(raw_photo, images_per_row, subimage_width)
    save_photo(result)

if __name__ == "__main__":
    recursivePhotoV1()