from random import randint
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from numba import jit
# import scipy.ndimage
# import PIL
# from PIL import Image
# from PIL import ImageDraw
# import pickle
# import glob
import copy
from skimage.morphology import convex_hull_image


def most_dist_points(xs, ys):
    x0 = np.mean(xs)
    y0 = np.mean(ys)
    largest_dists = {}
    dists = []
    biggest_dist = 0
    for i in range(len(xs)):
        dist = calc_dist([x0, y0], [xs[i], ys[i]])
        if dist > biggest_dist:
            biggest_dist = dist
            pt = [xs[i], ys[i]]

    biggest_dist = 0
    for i in range(len(xs)):
        dist = calc_dist(pt, [xs[i], ys[i]])
        if dist > biggest_dist:
            biggest_dist = dist
            pt_2 = [xs[i], ys[i]]

    return pt, pt_2, dist

# @jit(nopython=True,parallel=True,fastmath=True)
def calc_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def find_positions(frame, min_blob_size, max_blob_size,show_regions=False):
    num_regions, regions, stats, centroids = cv2.connectedComponentsWithStats(frame.astype(np.uint8))
    if show_regions:
      plt.imshow(regions)
      plt.show()
    result = []
    inds = []
    for region_index in np.arange(1, num_regions):
        # region_stats = stats[region_index]

        # remove too big or too small blobs
        # region_width, region_height = region_stats[2], region_stats[3]
        # if region_width < min_blob_size or region_width > max_blob_size \
        #         or region_height < min_blob_size or region_height > max_blob_size:
        #     continue
        if np.shape(np.where(regions==region_index))[1] < min_blob_size or np.shape(np.where(regions==region_index))[1] > max_blob_size :
          regions[regions==region_index]=0
          continue


        # get blob properties : main axis and centroid
        ind = _find_main_axis(regions, region_index)
        x, y = centroids[region_index][0], centroids[region_index][1]

        # find object type (biggest occurence in the region)
        # unique_values, count = np.unique(frame[regions == region_index], return_counts=True)
        # typ = unique_values[np.argmax(count)]

        result.append([x, y])
        inds.append(ind)


    return result,regions,inds

# @jit(fastmath=True)
def _find_main_axis(regions, region_index,return_ax=False):
    xs, ys = np.where(regions == region_index)

    inds = np.concatenate((ys[:, np.newaxis], xs[:, np.newaxis]), axis=1)

    if return_ax:
        m = np.concatenate([-ys[:, np.newaxis], xs[:, np.newaxis]], axis=1)
        _, _, v = np.linalg.svd(m - np.mean(m, axis=0), full_matrices=False)

        return [np.arctan2(v[0, 0], v[0, 1]),np.arctan2(v[-1,0],v[-1,1])],inds
    return inds

# @jit(fastmath=True,parallel=True)
def add_outlines(inds_regions,orig_frame,output_frame=0,add_nucleus_outline=True,color=(255,255,255),add_cell_outline=False,cell_outline_color=(255,255,255)):
  # orig_frame_copy = copy.copy(orig_frame)
  # if not output_frame.any():
  if not output_frame:
    # print('no output frame')
    output_img = np.zeros_like(orig_frame)
  else:
    output_img = copy.copy(output_frame)
    # print('output frame shape: ',np.shape(output_img))
  for i in range(len(inds_regions)):
    # print('i: ',i)
    mask = np.zeros_like(orig_frame)
    # print('did mask')
    mask[inds_regions[i][:,1],inds_regions[i][:,0]]=1
    # print('ones the mask')
    cell_seg = orig_frame*mask
    if add_cell_outline:
      cell_seg_2 = copy.copy(cell_seg)
      cell_seg_2[cell_seg_2>0]=1
      cell_seg_2[cell_seg_2<1]=0
      cell_seg_2 = convex_hull_image(cell_seg_2)
      kernel = np.ones((3, 3), np.uint8)
      cell_eroded = cv2.morphologyEx(cell_seg_2.astype('uint8'), cv2.MORPH_ERODE, kernel,iterations = 1)
      cell_outline = cell_seg_2-cell_eroded
      # cell_outline =  convex_hull_image(cell_outline)
      cell_outline_rgb = creating_color_image(cell_outline_color,cell_outline)
      output_img[cell_outline_rgb>0]=cell_outline_rgb[cell_outline_rgb>0]
    # print('multiplied with mask')
    if add_nucleus_outline:
      unique_labels = np.unique(cell_seg)
      # print('orig unique labels')
      unique_labels=unique_labels[1:] #ommit zero
      # print('ommitted zero')
      intl_guess = [0,np.min(unique_labels),np.mean(unique_labels),np.max(unique_labels)]
      # print('intl guess: ',intl_guess)
      ts = finding_threshold_bt(cell_seg,show=False,num_ths=2,initial_guesses=intl_guess)
      # print('ts: ',ts)
      t = np.ceil(ts[-2])
      # print('t: ',t)
      nucleus = copy.copy(cell_seg)
      nucleus[nucleus<t]=0
      nucleus[nucleus>0]=1
      nucleus = convex_hull_image(nucleus)
      kernel = np.ones((3, 3), np.uint8)
      nuc_eroded = cv2.morphologyEx(nucleus.astype('uint8'), cv2.MORPH_ERODE, kernel,iterations = 1)
      nuc_outline_2d = nucleus - nuc_eroded
      # nuc_outline_2d = convex_hull_image(nuc_outline_2d)
      nuc_outline_rgb = creating_color_image(color,nuc_outline_2d)
      # print('shape output img: ',np.shape(output_img))
      # print('shape outline rg: ',np.shape(nuc_outline_rgb))
      output_img[nuc_outline_rgb>0]=nuc_outline_rgb[nuc_outline_rgb>0]
  return output_img

# @jit(nopython=True,parallel=True,fastmath=True)
def creating_color_image(color,img):
  r = color[0]
  r_img = r*img
  g = color[1]
  g_img = g*img
  b = color[2]
  b_img = b*img
  rgb_color = np.concatenate((r_img[:,:,np.newaxis],g_img[:,:,np.newaxis],b_img[:,:,np.newaxis]),axis=2)
  # rgb_color = Image.fromarray(rgb_color.astype('uint8'),'RGB')
  # rgb_color = rgb_color.convert('RGBA')
  return rgb_color

# @jit(nopython=True,fastmath=True)
def finding_threshold_bt(cls_1,show=False,num_ths=1,initial_guesses=None,max_its=None):
  #we're expecting to get an array of a bunch of numbers from cls_1 and cls_2
  combined_array = cls_1
  percentile = 100*(1/(num_ths+1))
  percentiles = [int(np.ceil(percentile*(i))) for i in range(num_ths+2)]
  old_threshs = [np.percentile(combined_array,i) for i in percentiles]
  if initial_guesses:
    old_threshs = initial_guesses
  # print('old thresh: ',old_thresh)
  my_eps = 0.001
  eps=200
  i=0
  if not max_its:
    max_its = 100000
  while eps>my_eps and i<max_its:
    i+=1
    inds = [np.where(np.logical_and(combined_array>old_threshs[i],combined_array<old_threshs[i+1])) for i in range(num_ths+1)]
    sums = [np.shape(inds[i])[1] for i in range(len(inds))]
    avgs = [np.mean(combined_array[inds[i]]) for i in range(len(inds))]
    try:
      new_threshs = [np.average([avgs[i],avgs[i+1]],weights=[sums[i],sums[i+1]]) for i in range(num_ths)]
    except ZeroDivisionError:
      return old_threshs
    new_threshs.insert(0,np.min(combined_array))
    new_threshs.append(np.max(combined_array))
    eps = np.mean(abs(np.array(new_threshs)-np.array(old_threshs)))
    old_threshs = new_threshs
  if show:
    plt.imshow(segment_pic(cls_1,new_thresh))
    plt.show()
  return new_threshs

def my_custom_random(exclude):
  # exclude=[2,5,7]
  randInt = randint(0,10000)
  return my_custom_random(exclude) if randInt in exclude else randInt

# @jit(nopython=True,parallel=True,fastmath=True)
def create_opaque_color(color,transparency):
  # green = (0,204,102)
  # blue = (0,128,255)
  # gray = (128,128,128)
  #transparecty->[0,1]
  opacity = int(255*transparency)
  return color+(opacity,)


def save_image(img,img_dir):
  img.save(img_dir, quality=100)

# @jit(nopython=True,parallel=True,fastmath=True)
def find_crop_loc(C,frame_shape):
    fin_loc = []
    for i in range(len(C)):
        loc = C[i]
        if loc < 75:
            f_l0 = 0
            f_l1 = 150
        elif loc > frame_shape[i] - 75:
            f_l1 = frame_shape[i]
            f_l0 = f_l1 - 150
        else:
            f_l0 = int(loc - 75)
            f_l1 = f_l0 + 150
        fin_loc.append([f_l0,f_l1])
    return fin_loc