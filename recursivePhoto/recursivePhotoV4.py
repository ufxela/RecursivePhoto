from recursivePhoto.helpers.interface_helpers import *
from recursivePhoto.helpers.image_processing import *


# gets a list of photos 
def get_photos_list(pixels_per_photo_row):
    images_directory = get_images_directory()
    photo_paths = read_images_directory(images_directory)
    return make_mini_photos_grayscale(photo_paths, pixels_per_photo_row)

def recursivePhotoV4(): 
    # get photos from photos directory 
    images_per_row, subimage_width = get_recursive_photo_params()
    photos = get_photos_list(subimage_width)   
    
    # process photos from photos directory. note low_res_photos is list of tuples (low_res_photo, orig. photo)
    low_res_photos = create_low_res_photos(photos)
    rated_and_sorted_photos = rate_and_sort_low_res_grayscale(low_res_photos)
    
    # ask user for photo to recreate from photos from photos directory & then recreate.
    while(1):
        target_photo = convert_img_to_grayscale(get_photo()) # get photo that user wants to recursify
        
        # get target blocks
        target_images = get_target_images(target_photo, images_per_row)
        result_images = create_averaged_imagesV4(target_images, rated_and_sorted_photos)
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
    recursivePhotoV4()