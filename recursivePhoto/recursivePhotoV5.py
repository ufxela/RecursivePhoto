from recursivePhoto.helpers.interface_helpers import *
from recursivePhoto.helpers.image_processing import *
import random

# gets a list of photos 
def get_photos_list(pixels_per_photo_row):
    images_directory = get_images_directory()
    photo_paths = read_images_directory(images_directory)
    return make_mini_photos_colored(photo_paths, pixels_per_photo_row, split=1) # set split to 0 if you don't want any photos to be cropped/split into quarters in the final result.

def recursivePhotoV5(): 
    # get photos from photos directory 
    images_per_row, subimage_width = get_recursive_photo_params()
    photos = get_photos_list(subimage_width)   
    random.shuffle(photos) # turns out to be helpful to current implemetnion of get_close_images_by_avg_color (an implementation which needs to be fixed)

    # process photos from photos directory. note low_res_photos is list of tuples (low_res_photo, orig. photo)
    low_res_photos = create_low_res_photos(photos)
    rated_photos = rate_low_res_colored(low_res_photos)
    
    # ask user for photo to recreate from photos from photos directory & then recreate.
    while(1):
        target_photo = get_photo().convert("RGB") # get photo that user wants to recursify
        
        # get target blocks
        target_images = get_target_images(target_photo, images_per_row)
        result_images = create_averaged_imagesV5(target_images, rated_photos)
        result = stitch_images(result_images)
        save_photo(result)

        # ask user if they want to continue
        try:
            stop = input("continue? (Y/N)")
            if(stop.lower() == 'n'):
                break
        except:
            print("invalid input")
            break


if __name__ == "__main__":
    recursivePhotoV5()