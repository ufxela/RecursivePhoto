from recursivePhoto.helpers.interface_helpers import *
from recursivePhoto.helpers.image_processing import *

def get_user_input():
    raw_photo = get_photo()
    images_directory = get_images_directory()
    images_per_row, subimage_width = get_recursive_photo_params()
    return raw_photo, images_directory, images_per_row, subimage_width

def create_photo(raw_photo, images_directory, images_per_row, subimage_width):
    # process raw photo
    print("step 1 of 4: process raw photo")
    printProgressBar(0, 2)
    grayscale_img = convert_img_to_grayscale(raw_photo)
    printProgressBar(1, 2)
    block_averages = get_block_averages_grayscale(grayscale_img, images_per_row)
    printProgressBar(2, 2)

    # process images in images_directory
    print("step 2 of 4: process photos directory")
    photos_paths = read_images_directory(images_directory)
    photos = make_mini_photos_grayscale(photos_paths, subimage_width)
    print("step 3 of 4: rate and sort photos directory")
    rated_sorted_photos = rate_and_sort_grayscale(photos)

    # stitch together final result
    print("step 4 of 4: stitch together final result")
    images = create_averaged_imagesV2(block_averages, rated_sorted_photos)
    result = stitch_images(images)

    return result

def recursivePhotoV2():
    raw_photo, images_directory, images_per_row, subimage_width = get_user_input()
    result = create_photo(raw_photo, images_directory, images_per_row, subimage_width)
    save_photo(result)

if __name__ == '__main__':
    recursivePhotoV2()
