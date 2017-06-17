"""
main.py is the main function used for the purpose of vehicle detection and tracking,
as well as lane identification and marking in a video stream from camera mounted on the
vehicle.
"""

# import necessary modules
import sys
import numpy as np
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from collections import deque
import cv2

from parameters import *
from train_car_model import *
from vehicle_detection import *

from calib_cam import *
from img_transform import *
from find_lane import *

###############################################################################
# vehicle detection common section
boxes = deque()
def label_vehicles(img):
    # vehicle detection
    # find cars in an image
    # scales = [0.75, 1.0, 1.5]
    # ystart_all = [360, 360, 360]
    # ystop_all = [500, 620, 620]
    scales = [1.0, 1.5, 2.0]
    ystart_all = [400, 400, 400]
    ystop_all = [700, 700, 700]
    # use sliding window search for vehicle identification on 3 scales
    for i in range(0, len(scales)):
        scale = scales[i]
        ystart = ystart_all[i]
        ystop = ystop_all[i]
        box, box_image = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient,
                                   pix_per_cell, cell_per_block, spatial_size, hist_bins)
        # gather all boxes
        boxes.append(box)
    # remove some boxes
    if len(boxes) > 9: ####### change and test
        boxes.popleft()
        boxes.popleft()
        boxes.popleft()
    combo_box = []
    for box_list in boxes:
        combo_box += box_list
    # Add heat to each box in box list
    heat = add_heat(np.zeros_like(img[:, :, 0]), combo_box)
    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heat, 4)  # last=3
    # # Visualize the heatmap when displaying
    # heatmap = np.clip(heatmap, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    image_with_boxes, labeled_boxes = draw_labeled_bboxes(img, labels)

    # plt.figure()
    # plt.imshow(image_with_boxes)
    # plt.show()

    return image_with_boxes, labeled_boxes

###############################################################################
# image processing function for vehicle detection

def detect_vehicle(veh_image):
    # find cars in an image
    image_with_boxes, labeled_boxes = label_vehicles(veh_image)

    # plt.figure()
    # plt.imshow(image_with_boxes)
    # plt.title("image with boxes")
    # plt.show()

    return image_with_boxes

###############################################################################
# image processing function for lane-finding
def find_lanes(img):
    # load undistortion matrix from camera
    with open('./camera_calib_dist_pickle.p', 'rb') as pick:
        dist_pickle = pickle.load(pick)

    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']

    # undistort image (needed for video editing)
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)

    # apply thresholding
    thresh_img = combined_thresh(undist_img)
    # compare_images(image, thresh_img)

    # define window parameters
    offset = 0.20 * img.shape[1]  # offset for dst points
    win_bottom = 0.76  # .76
    win_top = .08
    win_height = .62
    y_bottom = .94

    # apply perspective transform to region of interest
    warped_img = perspective_transform(thresh_img, offset, win_bottom, win_top,
                                       win_height, y_bottom)
    # compare_images(image, warped_img)

    # detect lane and compute curvature
    left_fitx, lefty, right_fitx, righty, ploty, lane_info = detect_lane(warped_img)

    # draw lane on image
    lane_marked_image = draw_lanes(img, warped_img, left_fitx, right_fitx, ploty, offset,
                                   win_bottom, win_top, win_height, y_bottom)
    # plt.imshow(lane_marked_image)
    # plt.show()

    return lane_marked_image

###############################################################################
# process image for vehicle detection & lane finding in a given image
def process_image(img):
    image_feed = np.copy(img)
    # detect vehicles
    image_with_boxes, labeled_boxes = label_vehicles(img)

    # lane identification
    image_lane_marked = find_lanes(image_feed)

    # Draw the box on the image
    for bbox in labeled_boxes:
        cv2.rectangle(image_lane_marked, bbox[0], bbox[1], (255, 0, 0), 6)

    processed_image = image_lane_marked

    # processed_image = image_with_boxes

    return processed_image


##################################
#### main script #################
##################################
if __name__== '__main__':
    data_type = 'video'  # 'image' or 'video', to test on respective file
    # train and save model parameters per cars and notcars data set
    if train_model: # train model and save files
        model_training()
        print("closing application....exit")
        sys.exit()

    else:  # load previously trained model for sliding window search & classification
        # load previously trained model
        svc, X_scaler = load_trained_model()
        print("\n Loaded previous trained model, with parameter set-", str(param_set),
              ", Model accuracy = ", str(model_accuracy), "% \n")

    # defined data type to test on
    if data_type=='image':  # test algorithm on image
        image_file = './test_images/test1.jpg'
        image = mpimg.imread(image_file)

        # image processing steps, e.g: window-search, bbox definition, etc.
        print('Testing on image file')
        # new_image = detect_vehicle(image)
        new_image = process_image(image)

        plt.figure()
        plt.imshow(new_image)
        plt.show()

    elif data_type=='video': # test on video
        print("Starting video feed")

        # proj_output = "./output_test_video.mp4"
        # clip1 = VideoFileClip("./test_video.mp4")
        # output_clip = clip1.fl_image(
        #     detect_vehicle)  # NOTE: this function expects color images!!
        # output_clip.write_videofile(proj_output, audio=False)

        # proj_output = "./output_project_video.mp4"
        # clip1 = VideoFileClip("./project_video.mp4")
        # output_clip = clip1.fl_image(
        #     detect_vehicle)  # NOTE: this function expects color images!!
        # output_clip.write_videofile(proj_output, audio=False)

        # proj_output = "./output_project_video.mp4"
        # clip1 = VideoFileClip("./project_video.mp4").subclip(0.0, 10.0)
        # output_clip = clip1.fl_image(detect_vehicle)
        # output_clip.write_videofile(proj_output, audio=False)

        # proj_output = "./output_project_vehicle_detect.mp4"
        # clip1 = VideoFileClip("./project_video.mp4").cutout(0, 30.0)

        # proj_output = "./output_project_vehicle_detect_NEW_01.mp4"
        # clip1 = VideoFileClip("./project_video.mp4")
        # output_clip = clip1.fl_image(detect_vehicle)
        # output_clip.write_videofile(proj_output, audio=False)

        proj_output = "./output_project_lane_vehicle_detect_NEW_01.mp4"
        clip1 = VideoFileClip("./project_video.mp4")
        output_clip = clip1.fl_image(process_image)
        output_clip.write_videofile(proj_output, audio=False)

    print("\n End-of-main.py \n")