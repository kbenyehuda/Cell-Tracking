from PIL import Image
from PIL import ImageDraw
import glob
from PIL import ImageFont
import os
# import numpy as np
# import cv2
# from numba import jit
# import cv2
# import copy
# from skimage.morphology import convex_hull_image
# from random import randint
# import pickle
# import glob
# import matplotlib.pyplot as plt
# import math
# import scipy.ndimage
# import PIL
from cell_tracking_ex_funcs import *
from preproccessing import *

# COLORS
green = (0, 255, 0)
blue = (0, 0, 255)
gray = (128, 128, 128)
red = (255, 0, 0)
yellow = (255, 255, 0)
purple = (255, 0, 255)
black = (0, 0, 0)
white = (255, 255, 255)

def track_cells(frames_dir, frames_inds=(0, 1000), add_nuc_out=False, add_head_out=False, add_major_ax=False,
                 add_trail=False, add_bb=True,
                 min_blob=200, max_blob=1000, former_frames_centroids=None, former_frames=None,
                 beg_frame_idx=0,save_drawn_frames=False,saving_frames_dir = None,saving_cells = False):

    ''' STARTING UP NEW VARIABLES OR USING OLD ONES IN INPUT '''
    # new_cell = 0
    dist_th = 50
    frames_th = 2
    font_b = ImageFont.truetype(font='times_font.ttf', size=30, index=0, encoding='')
    if not former_frames_centroids:
        frames_centroids = {}
        # all_centers = {}
        cell_ids = []
        # drawn_frames = []

    else:
        frames_centroids = former_cell_centroids
        cell_ids = list(frames_centroids.keys())
        # all_centers = former_centers
        # draw_frames = former_frames

    '''STARTING TO RUN THE FUNCTION FOR EVERY FRAME'''
    for i in range(frames_inds[0], frames_inds[1]):
        adds_dict_orig_keys = {}
        adds_dict_new_keys = {}
        # if i % 50 == 0:
        #     print(i)
        frame_path = glob.glob(frames_dir+'frame_'+str(i+1)+'.bmp')
        frame = cv2.imread(frame_path[0])
        frame = preproccess(np.squeeze(frame[:,:,0]))
        orig_frame = copy.copy(frame)
        frame_shape = orig_frame.shape
        bin_frame = np.ones_like(frame)
        t_max = 100
        t_min = 0
        bin_frame[np.logical_and(frame < t_max, frame >= t_min)] = 0
        frame_stats, regions, inds_region = find_positions(bin_frame, min_blob, max_blob,
                                                           show_regions=False)  # was 8,1000
        # orig_frame = 255*(orig_frame)
        if save_drawn_frames:
            rgb_orig = np.concatenate(
                (orig_frame[:, :, np.newaxis], orig_frame[:, :, np.newaxis], orig_frame[:, :, np.newaxis]), axis=2)
            rgb_orig = add_outlines(inds_region, frame, output_frame=rgb_orig, add_nucleus_outline=add_nuc_out,
                                    color=purple, add_cell_outline=add_head_out, cell_outline_color=green)
            pil_bin_frame = Image.fromarray(rgb_orig.astype('uint8'), mode='RGB')
            pil_bin_frame = pil_bin_frame.convert("RGBA")
            draw = ImageDraw.Draw(pil_bin_frame)
        region_names = np.unique(regions)
        if i != 0 and not former_frames:
            old_regions = new_regions
        new_regions = {}
        for j in range(len(frame_stats)):
            # CENTROID
            center = [(frame_stats[j][0], frame_stats[j][1])]
            # other_center = [frame_stats[j][0], frame_stats[j][1]]
            new_regions[region_names[j]] = center
            # SQUARE

            hsz = 30
            # LINES
            if add_major_ax:
                pt1, pt2, dist = most_dist_points(inds_region[j][:, 0], inds_region[j][:, 1])
                draw.line([(pt1[0], pt1[1]), (pt2[0], pt2[1])], fill=create_opaque_color(blue, 0.9))
            # draw.line(line[0][1],fill='blue')

        if i != 0 and not former_frames:
            # old_keys = list(frames_centroids.keys())
            new_keys = list(new_regions.keys())
            for new_key in new_keys:
                old_keys = list(frames_centroids.keys())
                try:
                    dists = [((
                                      (frames_centroids[old_key][0][-1][0] - new_regions[new_key][0][0]) ** 2
                                      + (frames_centroids[old_key][0][-1][1] - new_regions[new_key][0][1]) ** 2) ** 0.5)
                             / ((beg_frame_idx + i) - frames_centroids[old_key][1][1])
                             for old_key in old_keys]

                except:
                    print('unable to calculate dists')
                    break

                no_key_found = 1

                while no_key_found:
                    if len(dists) > 0:
                        key = old_keys[np.argmin(dists)]
                        if np.min(dists) < dist_th and (beg_frame_idx + i) - frames_centroids[key][1][1] < frames_th:
                            if key not in adds_dict_orig_keys:
                                adds_dict_orig_keys[key] = new_regions[new_key][0]
                                no_key_found = 0

                            else:

                                del adds_dict_orig_keys[key]
                                no_key_found = 0

                        else:
                            old_keys.remove(key)
                            dists.remove(np.min(dists))
                    else:
                        new_cy, new_cx = new_regions[new_key][0]
                        no_key_found = 0
                        if new_cx < 50 or new_cx > frame.shape[1] - 50 or new_cy < 50 or new_cy > frame.shape[0] - 50 or\
                            (beg_frame_idx + i) < 10:
                            # new_cell += 1
                            cell_id = my_custom_random(cell_ids)
                            new_key_name = 'C_' + str(cell_id)
                            cell_ids.append(cell_id)
                            adds_dict_new_keys[new_key_name] = [new_regions[new_key][0]]

            old_keys = list(frames_centroids.keys())
            for key in old_keys:
                if key not in adds_dict_orig_keys and (beg_frame_idx + i) - frames_centroids[key][1][1] > frames_th and \
                        len(frames_centroids[key][0]) < 5:
                    del frames_centroids[key]
                    cell_ids.remove(int(key[2:]))



        else:
            new_keys = list(new_regions.keys())

            for g in range(len(list(new_regions.keys()))):
                # new_cell += 1
                cell_id = my_custom_random(cell_ids)
                key_name = 'C_' + str(cell_id)
                cell_ids.append(cell_id)
                adds_dict_new_keys[key_name] = [new_regions[new_keys[g]][0]]

        if adds_dict_new_keys:
            for key in list(adds_dict_new_keys.keys()):
                c = adds_dict_new_keys[key]
                if saving_cells:
                    [loc_x,loc_y] = find_crop_loc(c[0],frame_shape)
                    cropped_cell = orig_frame[loc_y[0] :loc_y[1], loc_x[0]:loc_x[1]]
                    frames_centroids[key] = [adds_dict_new_keys[key], [(beg_frame_idx + i), (beg_frame_idx + i)],
                                         [cropped_cell]]
                else:
                    frames_centroids[key] = [adds_dict_new_keys[key], [(beg_frame_idx + i), (beg_frame_idx + i)],
                                             [i]]
                if add_trail:
                    draw.line(frames_centroids[key][0], width=1, fill=create_opaque_color(yellow, 0.1))
                txt_x = int(round(adds_dict_new_keys[key][0][0])) + 15  # was 15
                txt_y = int(round(adds_dict_new_keys[key][0][1])) + 15  # was 15
                if save_drawn_frames:
                    draw.text((txt_x, txt_y), key, font=font_b, fill=create_opaque_color(white, 0.9))
                # all_centers[key] = [[adds_dict_new_keys[key][0][0], adds_dict_new_keys[key][0][1]]]

                if add_bb:
                    square = [(adds_dict_new_keys[key][0][0] - hsz, adds_dict_new_keys[key][0][1] + hsz),
                              (adds_dict_new_keys[key][0][0] + hsz, adds_dict_new_keys[key][0][1] + hsz),
                              (adds_dict_new_keys[key][0][0] + hsz, adds_dict_new_keys[key][0][1] - hsz),
                              (adds_dict_new_keys[key][0][0] - hsz, adds_dict_new_keys[key][0][1] - hsz)]
                    if key == 'C_3':
                        draw.polygon(square, outline=create_opaque_color(green, 0.3))
                    else:
                        draw.polygon(square, outline=create_opaque_color(red, 0.3))

        if adds_dict_orig_keys:
            for key in list(adds_dict_orig_keys.keys()):
                frames_centroids[key][0].append(adds_dict_orig_keys[key])
                frames_centroids[key][1][1] = beg_frame_idx + i
                if saving_cells:
                    c = adds_dict_orig_keys[key]
                    [loc_x,loc_y] = find_crop_loc(c,frame_shape)
                    frames_centroids[key][2].append(orig_frame[loc_y[0] :loc_y[1], loc_x[0]:loc_x[1]])
                else:
                    frames_centroids[key][2].append(i)
                if add_trail:
                    draw.line(frames_centroids[key][0], width=1, fill=create_opaque_color(yellow, 0.1))
                txt_x = int(round(adds_dict_orig_keys[key][0])) + 15
                txt_y = int(round(adds_dict_orig_keys[key][1])) + 15
                if save_drawn_frames:
                    draw.text((txt_x, txt_y), key, font=font_b, fill=create_opaque_color(white, 0.9))

                if add_bb:
                    square = [(adds_dict_orig_keys[key][0] - hsz, adds_dict_orig_keys[key][1] + hsz),
                              (adds_dict_orig_keys[key][0] + hsz, adds_dict_orig_keys[key][1] + hsz),
                              (adds_dict_orig_keys[key][0] + hsz, adds_dict_orig_keys[key][1] - hsz),
                              (adds_dict_orig_keys[key][0] - hsz, adds_dict_orig_keys[key][1] - hsz)]
                    if key == 'C_3':
                        draw.polygon(square, outline=create_opaque_color(green, 0.3))
                    else:
                        draw.polygon(square, outline=create_opaque_color(red, 0.3))

                # all_centers[key].append([adds_dict_orig_keys[key][0], adds_dict_orig_keys[key][1]])
        if save_drawn_frames:
            if not os.path.isdir(saving_frames_dir):
                os.mkdir(saving_frames_dir)
            save_image(pil_bin_frame, saving_frames_dir+'/frame_' + str(beg_frame_idx + i) + '.png')
            # drawn_frames.append(pil_bin_frame)
        if i % 500 == 0:
            print('len(frames_centroids): ', len(frames_centroids))

    return frames_centroids

