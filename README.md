# Cell-Tracking
This is the cell tracking algorithm used in the project.

It assumes a specific string for saved video files for reading the videos and then saving the video's time-space array per cell in a pickled dictionary with the appropriate string

The video path assumed: base_dir + 'donor x/video_y/frame_z.png'
  
  In every video directory, the algorithm saves a pickle file with the name 'frames_centroids.pickle'
  
  The pickle file has a dictionary whose keys are random IDs given for each tracked cell.
  
  The value for each key is a list containing 3 sub-lists.
    
    The first sub-list is the (x,y) corrdinates of each frame the cell was tracked.
    
    The second sub-list is the first and last frame the cell was tacked.
    
    The third sub-list is a list of all the frames where the cell was tracked.

The directory needs to be changed to the relevant directory where the videos are stored.
