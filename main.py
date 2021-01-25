from glob import glob
from numba import prange
import os
import time
from cell_tracking_main_func import track_cells
import pickle

base_dir = 'D:/Users/Keren/Documents/university/Year 2/DNA Fragmentation/Acridine Orange/Motility/new_videos/final'

donor_folders = glob(base_dir+'/*')

for d_i in prange(len(donor_folders)):
    donor_folder = donor_folders[d_i]
    if os.path.isdir(donor_folder):
        video_folders = glob(donor_folder+'/*')
        for v_i in prange(len(video_folders)):
            video_folder = video_folders[v_i]+'/'   #vid_dir from pre_main
            if not os.path.isfile(video_folder+'frames_centroids.pickle'):
                #save_video_dir = ...
                num_photos = len(glob(video_folder+'*.bmp'))
                st = time.time()
                print('starting to track ',video_folder)
                frames_centroids = track_cells(video_folder, frames_inds=(0, num_photos), add_nuc_out=False, add_head_out=False,
                                               add_major_ax=False, add_trail=False, add_bb=False, min_blob=50,
                                               max_blob=2000, save_drawn_frames=False)
                print('took ', time.time() - st, ' sec for running track_cells')

                with open(video_folder+'frames_centroids.pickle', 'wb') as f:
                  pickle.dump(frames_centroids, f)
                print('saved pickle file')



