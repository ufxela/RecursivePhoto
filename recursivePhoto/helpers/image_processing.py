from recursivePhoto.helpers.interface_helpers import printProgressBar
from PIL import Image
import copy 
import operator
import bisect
import sys
import time
import math
import random

"""
for searching
"""
class KeyList(object):
    # bisect doesn't accept a key function, so we build the key into our sequence.
    def __init__(self, l, key):
        self.l = l
        self.key = key
    def __len__(self):
        return len(self.l)
    def __getitem__(self, index):
        return self.key(self.l[index])

"""
given a list of mini-photos, rates them by gray scale then sorts them
"""
def rate_and_sort_grayscale(mini_photos_grayscale):
    num_photos = len(mini_photos_grayscale)
    counter = 0
    rated_sorted_photos = []
    # rate
    for photo in mini_photos_grayscale:
        printProgressBar(counter, num_photos)
        counter = counter + 1
        level = get_avg_grayscale(photo)
        rated_sorted_photos.append((level, photo))

    # sort
    rated_sorted_photos.sort(key = operator.itemgetter(0))
    printProgressBar(num_photos, num_photos)
    return rated_sorted_photos

"""
photos is a list of (photo, path) tuples
"""
def rate_and_sort_low_res_grayscale(photos):
    print("Rating and sorting photos")
    num_photos = len(photos)
    counter = 0
    printProgressBar(counter, num_photos)

    rated_sorted_photos = []

    # rate
    for photo in photos:
        level = get_avg_grayscale(photo[0])
        rated_sorted_photos.append((level, photo))

        counter = counter + 1
        printProgressBar(counter, num_photos)

    # sort
    rated_sorted_photos.sort(key = operator.itemgetter(0))
    return rated_sorted_photos

"""
photos is a list of (photo, path) tuples

Rates colored photos, but doesn't sort them
"""
def rate_low_res_colored(photos):
    print("Rating photos")
    num_photos = len(photos)
    counter = 0
    printProgressBar(counter, num_photos)

    rated_photos = []
    for photo in photos:
        level = get_avg_RGB(photo[0])
        rated_photos.append((level, photo))

        counter = counter + 1
        printProgressBar(counter, num_photos)

    return rated_photos

"""
photos is a list of photos

Rates colored photos, but doesn't sort them

note: should probably be unified with rate_low_res_colored().
"""
def rate_colored(mini_photos):
    print("Rating photos")
    num_photos = len(mini_photos)
    counter = 0
    rated_photos = []
    # rate
    for photo in mini_photos:
        level = get_avg_RGB(photo)
        rated_photos.append((level, photo))

        printProgressBar(counter, num_photos - 1)
        counter = counter + 1

    # we will not be sorting b/c it's hard to sort color images (return to this in future...)
    return rated_photos

"""
Rescale w/ cropping and scaling (no distoring) to 
desired width (d_width) and desired height (d_height)
"""
def rescale(img, d_width, d_height):
    o_width, o_height = img.size
    if(o_width == 0): #hacky... how else to avoid divide by zero?
        o_width = 1
    if(o_height == 0):
        o_height = 1
    o_ratio = float(o_width)/float(o_height)
    d_ratio = float(d_width)/float(d_height)
    if d_ratio < o_ratio:
        crop_height = o_height
        crop_width = crop_height * d_ratio
        x_off = float(o_width - crop_width) / 2
        y_off = 0
    else:
        crop_width = o_width
        crop_height = crop_width / d_ratio
        x_off = 0
        y_off = float(o_height - crop_height) / 3
    img = img.crop((x_off, y_off, x_off+int(crop_width), y_off+int(crop_height)))
    return img.resize((d_width, d_height), Image.NEAREST) # Image.NEAREST is faster, but looks noticeably worse; Image.LANCZOS is slower, but looks better
    
def convert_img_to_grayscale(img):
    return img.convert('L')

def get_pixelMap_avg_grayscale(pixelMap, width, height):
    total = 0
    for i in range(0, width):
        for j in range(0, height):
            total += pixelMap[i, j]
    total_pix = width * height 
    if (total_pix == 0): # check for divide by zero errors.
        total_pix = 1 # todo: figure out why there are such errors in the first place...
    return total / total_pix

def get_avg_grayscale(grayscale_img):
    width, height = grayscale_img.size
    pixelMap = grayscale_img.load()
    return get_pixelMap_avg_grayscale(pixelMap, width, height)

def luminance(pixel):
    return (0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])

def get_pixelMap_avg_RGB(pixelMap, width, height):
    R_tot = 0
    G_tot = 0
    B_tot = 0
    for i in range(0, width):
        for j in range(0, height):
            R_tot += pixelMap[i, j][0]
            G_tot += pixelMap[i, j][1]
            B_tot += pixelMap[i, j][2]
    if(width * height == 0):
        return 0, 0, 0
    return R_tot / (width * height), G_tot / (width * height), B_tot / (width * height)

def get_avg_RGB(color_img):
    width, height = color_img.size
    pixelMap = color_img.load()
    return get_pixelMap_avg_RGB(pixelMap, width, height)

"""
adds val to the grayscale value of each pixel of img
"""
def raise_img_by_val(img, val):
    pixelMap = img.load()
    width, height = img.size
    for i in range(0, width):
        for j in range(0, height):
            pixel_val = pixelMap[i,j]
            pixel_val += int(val)
            # make sure pixel_val is in bounds
            if(pixel_val > 255):
                pixel_val = 255
            if(pixel_val < 0):
                pixel_val = 0
            pixelMap[i,j] = pixel_val
    return img

def raise_img_by_color(img, color_RGB):
    pixelMap = img.load()
    width, height = img.size
    for i in range(0, width):
        for j in range(0, height):
            pixel_val = list(pixelMap[i,j])
            for k in range(0, 3):
                pixel_val[k] += color_RGB[0]
                pixel_val[k] += color_RGB[1]
                pixel_val[k] += color_RGB[2] 
                # make sure pixel_val is in bounds
                if(pixel_val[k] > 255):
                    pixel_val[k] = 255
                if(pixel_val[k] < 0):
                    pixel_val[k] = 0
            pixelMap[i,j] = tuple(pixel_val)
    return img

"""
computes a score where the lower the score, the more similar
threshold - if score above this, then ignore and return INF
"""
def img_similarity_grayscale(target, compare, threshold):
    score = 0
    target_pm = target.load()
    compare_pm = compare.load()
    width, height = target.size
    for row in range(width):
        for col in range(height):
            score += abs(target_pm[row, col] - compare_pm[row, col])
            if(score > threshold):
                return float('inf')
    return score

"""
searching for image similar to target from photos, 
returns a tuple (low res photo, path_name)

low_res_photos is a list of ((low res photo, orig_photo), index in rsp_copy) tuples
"""
def low_res_match_grayscale(target, low_res_photos, res):
    closest_score = float('inf')
    match = low_res_photos[0]
    for photo in low_res_photos:
        score = img_similarity_grayscale(target, photo[0][0], closest_score)
        if(score < closest_score):
            closest_score = score
            match = photo
    return match

"""
returns a list which is a sublist of photos containing N closest images in photos
w/ respect to average grayscale value

returned list also contains indices s.t. we can remove match from photos once its chosen

does not modify photos
"""
def get_closest_grayscale(target, photos, N):
    target_grayscale = get_avg_grayscale(target)

    closest = []

    # find closest value; then get surroundings b/c we know that photos is already sorted
    closest_index = bisect.bisect_right(KeyList(photos, key=lambda x: x[0]), target_grayscale) 
    if(closest_index >= len(photos)): # deal w/ case that target_grayscale > all photos grayscale scores
        closest_index = len(photos) - 1

    closest.append((photos[closest_index][1], closest_index))
    radius = int(N/2)
    for i in range(1, radius):
        if(closest_index - radius > 0):
            closest.append((photos[closest_index - radius][1], closest_index - radius))
        if(closest_index + radius < len(photos)):
            closest.append((photos[closest_index + radius][1], closest_index + radius))

    return closest

"""
computes a score where the lower the score, the more similar
threshold - if score above this, then ignore and return INF
"""
def img_similarity_color(target, compare, threshold):
    score = 0
    target_pm = target.load()
    compare_pm = compare.load()
    width, height = target.size
    for row in range(width):
        for col in range(height):
            score += RGB_dist(target_pm[row, col], compare_pm[row, col])
            if(score > threshold):
                return float('inf')
    return score


"""
searching for image similar to target from photos, 
returns a tuple (low res photo, path_name)

low_res_photos is a list of ((low res photo, orig_photo), index in rsp_copy) tuples
"""
def low_res_match_colored(target, low_res_photos, res):
    closest_score = float('inf')
    match = low_res_photos[0]
    for photo in low_res_photos:
        score = img_similarity_color(target, photo[0][0], closest_score)
        if(score < closest_score):
            closest_score = score
            match = photo
    return match, closest_score

"""
returns a list which is a sublist of photos containing images in photos
which have distance from target which is less than threshold

N is the max length that we allow closest to be

returned list also contains indices s.t. we can remove match from photos once its chosen

does not modify photos

reason this uses threshold and not N (as in get_closest_grayscale) is b/c
its easier for me to implement this way, but should probably change in the future.
^ to do it the way with N, possibly use kselect

! definitel;y need to strike a balance between threshold and N. Perhaps there's a better way to write this function which doesn't make that balance so important / put that burden on the user (put burden on implementation instaed)
"""
def get_close_images_by_avg_color(target, photos):
    N = len(photos)/20 + 1 # larger N => end result has less color match, more pattern match, longer runtime 
    threshold = 20 # larger threshold => less repeated photos, but worse color match, longer runtime

    target_RGB = get_avg_RGB(target)

    closest = []

    """
    idea: have closest be a list sorted by dist
    then, compute each dist, and pop off of list (a heap) if smaller etc.
    """
    # find all images with distance less than threshold to target_RGB
    closest_photo = 0
    closest_index = 0
    closest_dist = float("inf")
    for i in range(len(photos)):
        dist = RGB_dist(target_RGB, photos[i][0])
        if(dist < threshold): 
            closest.append((photos[i][1], i))
        if(dist < closest_dist):
            closest_dist = dist
            closest_index = i
            closest_photo = photos[i][1]
        if(len(closest) >= N): # return if already found enough photos
            return closest

    # in the case that we found nothing within threshold, then append the closest.
    if(len(closest) <= 0):
        closest.append((closest_photo, closest_index))
        
    return closest

"""
returns a 2d array where the (i, j) element contains
the average of the sub-image defined by with upper left corner at
(img.width/images_per_row * i, img.height/images_per_row * j)
and lower right corner at
(img.width/images_per_row * (i+1), img.height/images_per_row * (j + 1))

img is grayscale img and images_per_row is an int
"""
def get_block_averages_grayscale(img, images_per_row):
    width, height = img.size
    asp_ratio = float(height) / width
    simg_w = float(width) / images_per_row
    simg_h = simg_w
    block_averages = []
    for row in range(images_per_row):
        block_averages.append([])
        for col in range(int(images_per_row * asp_ratio)):
            subimg = img.crop((int(simg_w*row), int(simg_h*col), 
                int(simg_w*(row+1)), int(simg_h*(col+1))))
            block_averages[row].append(get_avg_grayscale(subimg))
    return block_averages

"""
returns a 2d array where the (i, j) element contains
the average of the sub-image defined by with upper left corner at
(img.width/images_per_row * i, img.height/images_per_row * j)
and lower right corner at
(img.width/images_per_row * (i+1), img.height/images_per_row * (j + 1))

img is grayscale img and images_per_row is an int

keeps original aspect ratio (no square)
"""
def get_block_averages_no_square(img, images_per_row):
    width, height = img.size
    simg_w = float(width) / images_per_row
    simg_h = float(height) / images_per_row
    block_averages = []
    for row in range(images_per_row):
        block_averages.append([])
        for col in range(images_per_row):
            subimg = img.crop((int(simg_w*row), int(simg_h*col), 
                int(simg_w*(row+1)), int(simg_h*(col+1))))
            block_averages[row].append(get_avg_grayscale(subimg))
    return block_averages

"""
returns 2D array where (i, j) element contains a 
low res (8 x 8) copy of the sub-image defined with upper left corner at
(img.width/images_per_row * i, img.height/images_per_row * j)
and lower right corner at
(img.width/images_per_row * (i+1), img.height/images_per_row * (j + 1))
"""
def get_target_images(img, images_per_row):
    width, height = img.size
    asp_ratio = float(height) / width
    simg_w = float(width) / images_per_row
    simg_h = simg_w
    target_images = []
    for row in range(images_per_row):
        target_images.append([])
        for col in range(int(images_per_row * asp_ratio)):
            subimg = img.crop((int(simg_w*row), int(simg_h*col), 
                int(simg_w*(row+1)), int(simg_h*(col+1))))
            target_images[row].append(rescale(subimg, 8, 8))
    return target_images

"""
colored version of get_block_averages
Still returns 2d array of ints, except instead of containing average of grayscale
contains tuple of (R, G, B)
"""
def get_block_averages_colored(img, images_per_row):
    width, height = img.size
    asp_ratio = float(height) / width
    simg_w = float(width) / images_per_row
    simg_h = simg_w
    block_averages = []
    for row in range(images_per_row):
        block_averages.append([])
        for col in range(int(images_per_row * asp_ratio)):
            subimg = img.crop((int(simg_w*row), int(simg_h*col), 
                int(simg_w*(row+1)), int(simg_h*(col+1))))
            block_averages[row].append(get_avg_RGB(subimg))
    return block_averages

"""
from a list of photo paths, opens photos, then processes them into mini-photos (i.e. photos to replace blocks)
"""
def make_mini_photos_grayscale(photos_paths, subimage_width):
    num_photos = len(photos_paths)
    counter = 0
    printProgressBar(0, num_photos)

    mini_photos = []
    scaled_width = subimage_width
    scaled_height = scaled_width 

    for photo_path in photos_paths:
        raw_photo = convert_img_to_grayscale(Image.open(photo_path))
        raw_photo_mini = rescale(raw_photo, scaled_width, scaled_height)
        mini_photos.append(raw_photo_mini)
        counter = counter + 1
        printProgressBar(counter, num_photos)

    return mini_photos

#todo: get rid of width/height args and also for progressBarPrints get rid of prefix/suffix/length args
def make_mini_photos_colored(photos_paths, subimage_width, split=0):
    num_photos = len(photos_paths)
    counter = 0
    printProgressBar(0, num_photos)

    mini_photos = []
    scaled_width = subimage_width
    scaled_height = scaled_width 
    

    for photo_path in photos_paths:
        raw_photo = Image.open(photo_path).convert("RGB")
        raw_photo_mini = rescale(raw_photo, scaled_width, scaled_height)
        mini_photos.append(raw_photo_mini)

        if(split != 0):
            width, height = raw_photo.size
            # also append 4 sub photos
            rp_1 = raw_photo.crop((0, 0, int(width/2), int(height/2)))
            rp_1_mini = rescale(rp_1, scaled_width, scaled_height)
            mini_photos.append(rp_1_mini)

            rp_2 = raw_photo.crop((0, int(height/2), int(width/2), height))
            rp_2_mini = rescale(rp_2, scaled_width, scaled_height)
            mini_photos.append(rp_2_mini)

            rp_3 = raw_photo.crop((int(width/2), 0, width, int(height/2)))
            rp_3_mini = rescale(rp_3, scaled_width, scaled_height)
            mini_photos.append(rp_3_mini)

            rp_4 = raw_photo.crop((int(width/2), int(height/2), width, height))
            rp_4_mini = rescale(rp_4, scaled_width, scaled_height)
            mini_photos.append(rp_4_mini)
        
        counter = counter + 1
        printProgressBar(counter, num_photos)

    return mini_photos

"""
creates low res (8x8) photos
returns a list of tuples which are (low_res_photo, orig_photo)
"""
def create_low_res_photos(photos):
    print("creating low res photos")
    num_photos = len(photos)
    counter = 0
    printProgressBar(counter, num_photos)

    low_res_photos = []
    res = 8
    for photo in photos:
        low_res_photo = rescale(photo, res, res) # rescale doesn't modify original
        low_res_photos.append((low_res_photo, photo))

        counter = counter + 1
        printProgressBar(counter, num_photos)

    return low_res_photos

"""
returns an array with size block_averages.size
which contains scaled and colored versions of img
such that each scaled and colors version has  
subimage_width pixels in each row
and it's average color is approximately the corresponding
value in block_averages
"""
def create_averaged_imagesV1(img, block_averages, subimage_width):
    orig_height = img.size[1]
    orig_color = get_avg_grayscale(img)

    scaled_width = subimage_width
    scaled_height = int(orig_height * (float(subimage_width) / orig_height))

    # modify img to the scaled size    
    img.thumbnail((scaled_width, scaled_height))

    averaged_images = []
    for row in range(len(block_averages)):
        averaged_images.append([])
        for col in range(len(block_averages[0])):
            scaled_img_cpy = img.copy()
            scaled_colored_img = raise_img_by_val(scaled_img_cpy, block_averages[row][col] - orig_color)
            averaged_images[row].append(scaled_colored_img)   
        printProgressBar(row, len(block_averages)-1)
    return averaged_images

"""
given block averages of original image which we're trying to recursify
as well as a list of photos which will compose the sub-images of the original image
we stitch create a 2d array of images where the (i,j)th image comes from rated_sorted_photos
and has a similar grayscale value t block_averages
"""
def create_averaged_imagesV2(block_averages, rated_sorted_photos):
    num_rows = len(block_averages)
    printProgressBar(0, num_rows)
    images = []
    rsp_copy = copy.copy(rated_sorted_photos)
    total_photos = len(rated_sorted_photos)
    for row in range(len(block_averages)):
        images.append([])
        for col in range(len(block_averages[0])):
            # find closest image in rsp_copy to block_averages[row][col], without replacement (to avoid having many adjacent photos be identical).
            closest_index = bisect.bisect_right(KeyList(rsp_copy, key=lambda x: x[0]), block_averages[row][col])
            if(closest_index >= len(rsp_copy)): # deal w/ case that block_avgs[row][col] > all rsp_copy grayscale scores
                closest_index = len(rsp_copy) - 1
            img_rating, img = rsp_copy.pop(closest_index)
            
            # lifting
            img_cpy = img.copy()
            scaled_colored_img = raise_img_by_val(img_cpy, block_averages[row][col] - img_rating)
            images[row].append(scaled_colored_img)

            # thresholding
            if(abs(img_rating - block_averages[row][col]) > 30 or len(rsp_copy) < len(rated_sorted_photos)/2): # not sure if 30 is the best threshold but it works! 
                rsp_copy = copy.copy(rated_sorted_photos)

        printProgressBar(row+1, num_rows)
            
    return images

"""
computes RGB distance between p1 and p2 which are RGB tuples
"""
def RGB_dist(p1, p2):
    # dist = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2] - p2[2])
    dist = math.sqrt(((p1[0]-p2[0])*.3)**2 + ((p1[1]-p2[1])*.59)**2 + ((p1[2]-p2[2])*.11)**2)
    return dist

"""
library is an arbitrary (unsorted) list of (RGB, photo) tuples
returns index of photo in library closest to target_RGB
"""
def find_closest_RGB(target_RGB, library):
    min_dist = float("inf")
    min_index = -1
    for i in range(len(library)):
        photo_RGB = library[i][0]
        dist = RGB_dist(target_RGB, photo_RGB)
        if(dist < min_dist):
            min_dist = dist
            min_index = i
    return min_index, min_dist

"""
colored version
rated_photos is not sorted
"""
def create_averaged_imagesV3(block_averages, rated_photos):
    num_rows = len(block_averages)
    printProgressBar(0, num_rows)
    images = []
    rs_copy = copy.copy(rated_photos)
    for row in range(len(block_averages)):
        images.append([])
        for col in range(len(block_averages[0])):
            # find closest image in rs_copy to block_averages[row][col], with replacement
            closest_index, dist = find_closest_RGB(block_averages[row][col], rs_copy)
            if(closest_index >= len(rs_copy)): # deal w/ case that block_avgs[row][col] > all rs_copy color scores
                closest_index = len(rs_copy) - 1
            img_rating, img = rs_copy.pop(closest_index)
            images[row].append(img)

            # thresholding
            threshold = 30
            if(dist > threshold or len(rs_copy) < len(rated_photos)/2): # the lower threshold is, the closer colors match, but the more chance there is for repeated images.
                rs_copy = copy.copy(rated_photos)
        printProgressBar(row+1, num_rows) 
    return images

"""
does not modify rated_and_sorted_photos
creates a 2D array of images which match target_images

does this by doing a low res comparison of target_image and rated_sorted_photo

target_images - a 2D array of 8x8 target images
rated_sorted_photos - a list of (grayscale val, (low_res_image, orig_image)) tuples

returns:
2D array of orig_images which match target_images
"""
def create_averaged_imagesV4(target_images, rated_sorted_photos):
    print("creating averaged images")
    num_rows = len(target_images)
    printProgressBar(0, num_rows)
    rsp_cpy = copy.copy(rated_sorted_photos)

    res = 8
    images = []
    for row in range(len(target_images)):
        images.append([])
        for col in range(len(target_images[0])):
            # refill rp_copy occasionally (simpler, hackier way to threshold)
            if(len(rsp_cpy) <= float(len(rated_sorted_photos))*19.0/20.0):
                rsp_cpy = copy.copy(rated_sorted_photos)
            N = len(rsp_cpy)/10 + 1 # needs tweaking. dividing by larger = less color match, more pattern match. 
            close_low_res_photos = get_closest_grayscale(target_images[row][col], rsp_cpy, N)
            match = low_res_match_grayscale(target_images[row][col], close_low_res_photos, res)
            images[row].append(match[0][1]) # match is ((low_res, orig), index)
            rsp_cpy.pop(match[1]) # chose w/o replacement to avoid monotony. Comment out if you do want replacement. Note: this is the only thing you need to comment out if you want replacement.
        printProgressBar(row+1, num_rows)
            
    return images

"""
**colored version of create_averaged_imagesV4()

does not modify rated_and_sorted_photos
creates a 2D array of images which match target_images

does this by doing a low res comparison of target_image and rated_sorted_photo

target_images - a 2D array of 8x8 target images
rated_photos - a list of (RGB val, (low_res_image, orig_image)) tuples

returns:
2D array of orig_images which match target_images
"""
def create_averaged_imagesV5(target_images, rated_photos):
    print("creating averaged images")
    num_rows = len(target_images)
    printProgressBar(0, num_rows)
    rp_cpy = copy.copy(rated_photos)

    res = 8
    images = []
    for row in range(len(target_images)):
        images.append([])
        for col in range(len(target_images[0])):
            close_low_res_photos = get_close_images_by_avg_color(target_images[row][col], rp_cpy)
            match, dist = low_res_match_colored(target_images[row][col], close_low_res_photos, res)
            images[row].append(match[0][1]) # match is ((low_res, orig), index)
            rp_cpy.pop(match[1]) # chose w/o replacement to avoid monotony. Comment out if you do want replacement. Note: this is the only thing you need to comment out if you want replacement.

            # refill match not close enough, or if rp_cpy is getting too empty.
            if(dist > 2000 or len(rp_cpy) < len(rated_photos)/2): # not sure if 2000 is the right threshold, but it works decently.
                rp_cpy = copy.copy(rated_photos)

        printProgressBar(row+1, num_rows)
            
    return images

"""
stitches 2d array of sub-images images into a larger image
"""
def stitch_images(images):
    subimg_width, subimg_height = images[0][0].size
    width = subimg_width * len(images)
    height = subimg_height * len(images[0])
    result = Image.new(images[0][0].mode, (width, height))
    for row in range(len(images)):
        for col in range(len(images[0])):
            result.paste(images[row][col], (row*subimg_width, col*subimg_height))
    return result
