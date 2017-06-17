"""
parameters.py defines all parameters used for vehicle detection and tracking
"""

train_model = False # True--model needs to be trained, False--otherwise
reduce_sample = False  # reduce data sample size when using limited memory resources
n_sample = 10  # reduced sample size if option selected

# choose parameter set for sliding-window and bounsing box definition
# accuracy = 1--96.4%, 2--99.39%, 3--98.25%, 4--
param_set = 2

x_start_stop = [150, 1280]
y_start_stop = [None, None]
xy_window = (64, 64)
xy_overlap = (0.75, 0.75)

# select system parameters
### TODO: Tweak these parameters and see how the results change.
if param_set==1: # 96.4% accuracy
    color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 0  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    filename_svc_model = 'trained_svc_model_01.pkl'
    filename_xscaler = 'x_scaler_01.pkl'
    model_accuracy = 96.4


elif param_set==2:  # 98.87 % accuracy ; 99.47% with mirrored images
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (64, 64)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    filename_svc_model = 'trained_svc_model_02.pkl'
    filename_xscaler = 'x_scaler_02.pkl'
    # model_accuracy = 98.87
    model_accuracy = 99.47 # with mirror images


elif param_set == 3:    # 98.25% accuracy
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = False  # Spatial features on or off
    hist_feat = False  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    filename_svc_model = 'trained_svc_model_03.pkl'
    filename_xscaler = 'x_scaler_03.pkl'
    model_accuracy = 98.25


elif param_set == 4: # 98.87%
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    filename_svc_model = 'trained_svc_model_04.pkl'
    filename_xscaler = 'x_scaler_04.pkl'
    model_accuracy = 98.87

elif param_set == 5:  # 98.87 % accuracy ; 99.39 with mirrored images
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    filename_svc_model = 'trained_svc_model_05.pkl'
    filename_xscaler = 'x_scaler_05.pkl'
    # model_accuracy = 98.87
    model_accuracy = 96.47  # with mirror images

elif param_set==6:  #
    color_space = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (64, 64)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    filename_svc_model = 'trained_svc_model_06.pkl'
    filename_xscaler = 'x_scaler_06.pkl'
    # model_accuracy = 98.87
    model_accuracy = 99.24 # with mirror images

else:
    color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 0  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    filename_svc_model = 'trained_svc_model.pkl'
    filename_xscaler = 'x_scaler.pkl'
    model_accuracy = 'NA'
