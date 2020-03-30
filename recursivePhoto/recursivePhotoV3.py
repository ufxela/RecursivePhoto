from recursivePhoto.helpers.interface_helpers import *
from recursivePhoto.helpers.image_processing import *

def get_user_input():
    raw_photo = get_photo()
    images_directory = get_images_directory()
    images_per_row, subimage_width = get_recursive_photo_params()
    return raw_photo, images_directory, images_per_row, subimage_width

def create_photo(raw_photo, images_directory, images_per_row, subimage_width):
    # process raw photo
    block_averages = get_block_averages_colored(raw_photo, images_per_row)
    # process images in images_directory
    print("step 1 of 3: process photos directory") # todo: there's probably a better way of doing this, like having a global stuct w/ steps along with descriptions and then using index/lengths to print step i of n.
    photos_paths = read_images_directory(images_directory) 
    photos = make_mini_photos_colored(photos_paths, subimage_width)
    # todo: probably (for all versions) should add an assertion to ensure len(photos) > 0
    print("step 2 of 3: rate photos directory")
    rated_photos = rate_colored(photos)

    # stitch together final result
    print("step 3 of 3: stitch together final result")
    images = create_averaged_imagesV3(block_averages, rated_photos)
    result = stitch_images(images)

    return result

def recursivePhotoV3():
    raw_photo, images_directory, images_per_row, subimage_width = get_user_input()
    result = create_photo(raw_photo, images_directory, images_per_row, subimage_width)
    save_photo(result)

if __name__ == '__main__':
    recursivePhotoV3()