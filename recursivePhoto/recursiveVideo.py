from recursivePhoto.helpers.interface_helpers import *
from recursivePhoto.helpers.image_processing import *
import cv2
import os

def get_user_input():
    raw_photo = get_photo()
    images_per_row_start, images_per_row_end = get_recursive_video_params()
    return raw_photo, images_per_row_start, images_per_row_end

def create_photo(raw_photo, images_per_row, subimage_width):
    grayscale_img = convert_img_to_grayscale(raw_photo)
    block_averages = get_block_averages_no_square(grayscale_img, images_per_row)
    averaged_images = create_averaged_imagesV1(grayscale_img, block_averages, subimage_width)
    result = stitch_images(averaged_images)
    return result

def create_video(raw_photo, images_per_row_start, images_per_row_end):
    result_fp = get_video_fp()
    
    size = raw_photo.size
    width = size[0]

    out = cv2.VideoWriter("./results/" + result_fp,cv2.VideoWriter_fourcc(*'MP4V'), 15, size)
    
    for i in range(images_per_row_start, images_per_row_end):
        print("step " + str(i - images_per_row_start) + " of " + str(images_per_row_end - images_per_row_start))
        print("processing photo " + str(i - images_per_row_start + 1) + " of " + str(images_per_row_end - images_per_row_start))
        result = create_photo(raw_photo, i, int(float(width)/i))
        img_name = "./images/" + str(i) + "intermediate.png"
        save_photo_with_filepath(result, img_name)
        int_img = cv2.resize(cv2.imread(img_name), size)
        repeats = int(images_per_row_end / (i * 10) + 1)
        for i in range(repeats): # so that the initial images are displayed for a longer time, looks nicer
            out.write(int_img)
        printProgressBar(100, 100)
    out.release()

def cleanup_video(images_per_row_start, images_per_row_end):
    for i in range(images_per_row_start, images_per_row_end):
        img_name = "./images/" + str(i) + "intermediate.png"
        os.remove(img_name)

def recursiveVideo():
    raw_photo, images_per_row_start, images_per_row_end = get_user_input()

    create_video(raw_photo, images_per_row_start, images_per_row_end)
    cleanup_video(images_per_row_start, images_per_row_end)

if __name__ == "__main__":
    recursiveVideo()