from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
from PIL import Image
import os, os.path
import sys

def get_photo():
    print("Select an image to recurse-ify from the pop up.")
    while 1:
        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        image_filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
        try:
            im = Image.open(image_filename)
            return im
        except:
            print("The file path '" + image_filename + "' is not valid. Try again.")

def get_images_directory():
    print("From the pop up, select a directory with images to recurse-ify the original image with (i.e. the recursivePhoto result will be made from pictures in this directory). The more photos in this directory, the better!")
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    images_directory = askdirectory() # show an "Open" dialog box and return the path to the selected file
    return images_directory

"""
reads in photos from directory with name read_images_directory
"""
def read_images_directory(read_images_directory):
    photos = []
    img_file_types = ['.jpg', '.jpeg', '.png'] # can add more image types
    for filename in os.listdir(read_images_directory):
        ext = os.path.splitext(filename)[1]
        if(ext.lower() in img_file_types):
            photos.append(os.path.join(read_images_directory, filename))
    return photos
    
def get_recursive_photo_params():
    images_per_row = 50
    print("how many mini-photos do you want in each row of the result? Recommended: 50")
    while 1:
        try:
            images_per_row = int(input())
            break
        except:
            print("need to type a number e.g. '50'")

    subimage_width = 40
    print("how many pixels wide do you want each mini-photo? Recommended: 40")
    while 1:
        try:
            subimage_width = int(input())
            break
        except:
            print("need to type a number e.g. '40'")

    return images_per_row, subimage_width

def get_recursive_video_params():
    images_per_row_start = 50
    images_per_row_end = 50
    print("how many mini-photos do you want in each row of the result to start? Recommended: 1")
    while 1:
        try:
            images_per_row_start = int(input())
            break
        except:
            print("need to type a number e.g. '50'")

    print("how many mini-photos do you want in each row of the result to start? Recommended: 50")
    while 1:
        try:
            images_per_row_end = int(input())
            break
        except:
            print("need to type a number e.g. '50'")

    subimage_width = 40

    return images_per_row_start, images_per_row_end
        
def save_photo_with_filepath(img, filepath):
    img.save(filepath)

def save_photo(img):
    print("Proccessing complete! Have a look at your picture!")
    img.show()
    print("Give me a filename to save this image to. Will be saved to your results. Type `CANCEL` if you don't want to save anything")
    while 1:
        try:
            fp = input()
            if(fp == 'CANCEL'):
                break
            img.save("./results/" + fp)
            break
        except:
            print("need to add file extension (e.g. .png, test.png)")

def get_video_fp():
    print("What shall I call your video? Use .mp4 e.g. project.mp4")
    while 1:
        fp = input()
        if fp[len(fp)-4:] == ".mp4":
            return fp
        print("didn't end with .mp4")

# Print iterations progress. From https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar(iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 1, length = 50, fill = '.', printEnd = "\r"):
    if(total == 0):
        total = 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.write("\033[F") # Cursor up one line
    # Print New Line on Complete
    if iteration == total: 
        print("\n")