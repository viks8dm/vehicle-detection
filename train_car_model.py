"""
main script for training a model to differentiate between car and non-car images
"""

import glob
import time
import sys
import cv2
from random import shuffle
import numpy as np
import pickle
import matplotlib.image as mpimg
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split

from parameters import *
from vehicle_detection import *

################################################################
# read data file function
def read_data():
    # read all file names from respective folders
    car_files = glob.glob('./data/vehicles/**/*.png')
    notcar_files = glob.glob('./data/non-vehicles/**/*.png')

    # read reduced sample size is specified
    # helpful when available memory is limited
    if reduce_sample:
        car_files = car_files[0:n_sample]
        notcar_files = notcar_files[0:n_sample]

    print("Total vehicles images = ", len(car_files))
    print("Total non-vehicles images = ", len(notcar_files))

    # extract car and non-car images
    print("Extracting car and non-car images ...")
    cars = np.array([(mpimg.imread(file)*255).astype('uint8') for file in car_files])
    notcars = np.array([(mpimg.imread(file)*255).astype('uint8') for file in
                        notcar_files])
    print("... ... DONE")

    return car_files, notcar_files, cars, notcars

################################################################
# flip images horizontally
def flip_horz(images):
    mirror_image = []
    for image in images:
        # add mirror of alternate images only
        # if n%2 == 0:
        mirror_image.append(cv2.flip(image,1))

    return mirror_image

################################################################
# image augmentation scripts from Traffic-Sign-Calssifier
def augment_brightness_camera_images(image):
    # code fomr vivek yadav
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    # code fomr vivek yadav
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    shear_M = cv2.getAffineTransform(pts1,pts2)
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img

# augment image as needed
def image_augmentation(img):
    # define flipped images
    # flipped_images = flip_horz(img)

    # # define random augmented images
    aug_images = []
    for image in img:
        new_image = transform_image(image, 20, 15, 5, brightness=1)
        aug_images.append(new_image)

    # append all augmented images
    # with_flipped_images = np.append(img, flipped_images, axis=0)
    # all_images = np.append(with_flipped_images, aug_images, axis=0)

    all_images = np.append(img, aug_images, axis=0)

    return all_images


################################################################
def model_training():
    # flash message when training the model
    print("===========================================================")
    print(" Training model for CAR and NON-CAR classification")
    print(" Using Parameter Set - ", str(param_set), " with following values")
    print("-----------------------------------------------------------")
    print("color_space = ", color_space, " # Can be RGB, HSV, LUV, HLS, YUV, YCrCb")
    print("orient = ", str(orient), " # HOG orientations")
    print("pix_per_cell = ", str(pix_per_cell), " # HOG pixels per cell")
    print("cell_per_block = ", str(cell_per_block), " # HOG cells per block")
    print("hog_channel = ", str(hog_channel), " # Can be 0, 1, 2, or 'ALL'")
    print("spatial_size = ", str(spatial_size), " # Spatial binning dimensions")
    print("hist_bins = ", str(hist_bins)," # Number of histogram bins")
    print("spatial_feat = ", str(spatial_feat)," # Spatial features on or off")
    print("hist_feat = ", str(hist_feat)," # Histogram features on or off")
    print("hog_feat = ", str(hog_feat)," # HOG features on or off")
    print(" ")
    print("filename_svc_model = ", filename_svc_model)
    print("filename_xscaler = ", filename_xscaler)
    print("===========================================================")

    # read car and non-car images for training
    car_files, notcar_files, car_images, notcar_images = read_data()

    # add augmented images to training set
    if param_set==6:
        print("Initial car-image set size: ", len(car_images))
        print("Initial NOT-car-image set size: ", len(notcar_images))
        flipped_car_images = flip_horz(car_images)
        car_images = np.append(car_images, flipped_car_images, axis=0)

        flipped_notcar_images = flip_horz(notcar_images)
        notcar_images = np.append(notcar_images, flipped_notcar_images, axis=0)
        print("Final car-image set size: ", len(car_images))
        print("Final NOT-car-image set size: ", len(notcar_images))

    if param_set == 5:
        print("Initial car-image set size: ", len(car_images))
        print("Initial NOT-car-image set size: ", len(notcar_images))
        car_images = image_augmentation(car_images)
        notcar_images = image_augmentation(notcar_images)
        print("Final car-image set size: ", len(car_images))
        print("Final NOT-car-image set size: ", len(notcar_images))

    # extract car and non-car features
    car_features = extract_features(car_images, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcar_images, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(80, 100)
    print("Using random-state = ", str(rand_state))
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()

    # save trained model
    saved_svc_model = open(filename_svc_model, 'wb')
    pickle.dump(svc, saved_svc_model)
    save_scaler = open(filename_xscaler, 'wb')
    pickle.dump(X_scaler, save_scaler)

    print("Model training complete"
          "Files saved")

    return


################################################################
# load trained model
def load_trained_model():
    load_svc_model = open(filename_svc_model, 'rb')
    load_scalar = open(filename_xscaler, 'rb')

    svc = pickle.load(load_svc_model)
    X_scalar = pickle.load(load_scalar)

    return svc, X_scalar

##################################
#### main script #################
##################################
if __name__== "__main__":
    # define parameters and train model
    if train_model: # train model and save files
        model_training()
        print("closing application....exit")
        sys.exit()

    else:
        print("No valid action specified......closing application")


