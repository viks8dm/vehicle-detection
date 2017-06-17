#**Self-Driving Car Engineer Nanodegree** 

##Project-5: Vehicle Detection and Tracking

[image1]: ./output_images/sample_images.jpg "Car and non-car samples"
[image2]: ./output_images/hog_car_image_32.jpg "HOG of car image (orientation=32)"
[image3]: ./output_images/hog_car_image_08.jpg "HOG of car image  (orientation=8)"
[image4]: ./output_images/hog_noncar_image_32.jpg "HOG of non-car image  (orientation=32)"
[image5]: ./output_images/hog_noncar_image_08.jpg "HOG of non-car image  (orientation=8)"
[image6]: ./output_images/training_sample.jpg "Sample training run"
[image7]: ./output_images/parameter_set_2.jpg "Parameter Set 2 details"
[image8]: ./output_images/all_sliding_window.jpg "Map of all sliding window"
[image9]: ./output_images/sample_bounding_box_image-1.jpg "Sample image with bounding boxes (test image 1)"
[image10]: ./output_images/sample_bounding_box_image-6.jpg "Sample image with bounding boxes (test image 6)"
[image11]: ./output_images/heatmap_image1.jpg "Parameter Set 2 details"
[image12]: ./output_images/heatmap_thresholding_image1.jpg "Parameter Set 2 details"
[image13]: ./output_images/vehicle_detected_image.jpg "Final image with vehicles marked"
[image14]: ./output_images/vehicle_and_lane_image.jpg "Final image with vehicles marked & lanes identified"


###Project Goals

The goals / steps of this project are as follows:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, one can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* For first two steps normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the vehicle-detection pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

### Files submitted (with this report)

* `main.py` - main script used for calling model-training, image-testing or video-streaming for the project. It contains sub-scripts for vehicle-detection & lane-finding (from advanced-lane-finding project). Several parameters for estimating a vehicle bounding box are also defined here.
* `parameters.py` - defines all parameters that are being used or that were used at some point for model training and actual implementation of vehicle-detection method.
* `train_car_model.py` - This script is used for model training purposes. It has functions to read data, add augmented images to the original set, perform model training using linear-SVC and finally save trained model data for future use for vehicle-detection.
* `vehicle_detection.py` - This script has multiple functions for feature extraction, hog implementation, color conversion,  heat-thresholding, drawing bounding box, etc.
* `adv_lane_finding.py` - main script from previous project on `advanced-lane-finding`.
* `calib_cam.py` - main script used by `adv_lane_finding.py` for camera calibration and distortion correction.
* `find_lane.py` - script for finding lane lines on the road image using polynomial fitting on region of interest and radius of curvature computation.
* `img_transform.py` - This script uses color transforms, gradients, etc., to create a thresholded binary image that is used for lane detection and marking.
* `trained_svc_model_02.pkl` - Linear SVM classifier model, saved after training run.
* `x_scalar_02.pkl` - X-scalar fature set saved after training run.
* `camera_calib_dist_pickle.p` - pickle file that saves camera calibration and distortion-correction matrices for future use.
* `output_project_vehicle_detection_set2.mp4` - output video stream with vehicle detection.
* `output_project_lane_vehicle_detection_set2.mp4` - output video stream with vehicle detection and lane identification.

---

###Histogram of Oriented Gradients (HOG) & Model Training

In order to train the model, I tried various combinations of parameters and methods and tested different cases over a couple of weeks. To start with, I read car and non-car images and plotted a few with different color spaces to familiarize myself with the training data set I was dealing with.

A sample image from car and non-car set is shown in [image1]

![alt text][image1]

I used HOG visualization see impact of various parameters like orientation, pixels per cell, and numbers of cells per block to see how changing a parameter changes the output (visually). An example is shown in the images below for orientation values of 32 and 8. It appears that if one uses very high value of orientation, the system tries overfit in certain areas where there are curves of slanting edges. 

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

As far as change in training model accuracy is concerned, these parameter changes did not result in significant change in the training accuracy, hence I selected a parameter-set (parameter-set 2 in my implementation) which gave slightly better accuracy than other sets and used it for final implementation.

In addition to HOG classification, I used a histogram of colors with color_hist() and a feature vector of the colors with bin_spatial() for training the SVC. A sample training run looked like that shown [image6] below.

--
![alt text][image6]
--

My desire was to achieve training accuracy of 99% or better. In an effort to do this, I experimented with color-spaces, hog-channels and added image augmentation, similar to that was dor for `traffic-sign-classification project`. I used  [Vivek Yadav's script](https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3) and tried to randomly augment both car as well as non-car images via rotation, stretching/translation and changing brighness. However, this resulted in reduction in accuracy to in 96% range. This was discouraging and I could see several false positives in my video stream as a result. But, I kept trying and finally only image augmentation that I used was addition of mirror images for each image in the car and non-car image set. This gave `99.47%` accuracy for the training set that I used finally (with parameter-set 2 displayed in [image7]).

--
![alt text][image7]
--

The final training pipeline used about 9K of original car and non-car images respectively and added 1 mirror image for each. Hence total 18K car-images and 18K non-car images were used. Further, as was done in examples shown in lectures, the features were standardized using X scalar mean of 0 and equal variance. Image labels were h-stacked, data shuffled and split into 80:20 ratio of training & test sets.

NOTES:

* All parameters are defined in `parameters.py`.
* Most training related functions are part of `train_car_model.py`.
* `extract_features()` which is used by training script is part of `vehicle_detection.py`, since it is shared by model-training and vehicle-detection scripts.


---


###Sliding Window Search


Once the model is trained a sliding window search is run on the bottom half of the image to find cars in the image (`find_cars()` under `vehicle_detection.py` is the main script for this. It call others modules like `extract_features()`, `slide_window()`, etc. which are all part of `vehicle_detection.py`).

I did find suggessions online and on the forum that recommended using only bottom right quarter of the image, however, I wanted to develop a methodology for a general driving case and not limit to the particular case shown in the video where the vehicles that are driving in the same direction appear only on the bottom right section of each video-frame. This posed some difficuty since my implementation sees some vehicles that are coming from the other side of the divider, however, I did not try to exclude that portion of the image.

I experimented with multiple grid sizes and finally opted for 3-scales with values of 1.0, 1.5 and 2.0. A typical search grid is as shown in [image8] below. These values were finally selected on the basis of multiple experiments over a few days (These values are defined in `main.py` under `label_vehicles()` which then calls `find_cars()` from `vehicle_detection.py`).

![alt text][image8]

When a sliding window search is run using the grid shown above, the hog features for that part are taken out, and any cars that are found are bound with a box. Two sample images with all bounding boxes are shown in [image9] and [image10]. As was mentioned previously, please note that in [image9] the car of the other side (left) of the road is also detected and marked for one of the selected scales. In other cases it is not marked because the car is only partially visible; some portion is hidden behind guard rails.

![alt text][image9]
![alt text][image10]

Once the bounding boxes are identified, a blank image is taken and an incremental heatmap is applied that increments the pixel value of each location in each bounding box by 1. This gives the frequency of how often each pixel appeared in a bounding box, that is visualized in [image11] below for (test img-1) shown above (`add_heat()` from `vehicle_detectin.py` is used for this)

![alt text][image11]

To get rid of any false positives, a thresholding on this heatmap frequency is applied. Here I use a value of 4 for cutoff (`apply_threshold()` from `vehicle_detectin.py` is used for this).

![alt text][image12]

Thereafter `draw_labeled_bboxes()` from `vehicle_detection.py` takes these separate labels, and applies a single bounding box to each one by finding the min and max of x & y points for the region of interest. This combines duplicate bounding boxes, and returns images with a single bounding box for each car, which looks similar to [image13] below.

![alt text][image13]

I combined my vehicle detection code with the advanced lane finding project to detect vehicles and mark lanes at the same time. For this, I run vehicle-detection script on each image (or video-frame) to find bounding boxes, then use the original image for lane-finding. Once I have lane and bouding box information, I use `process_image()` in `main.py` with cv2.rectangle function to combine all these into one image and the output is displayed below.

![alt text][image14]


---


### Video Implementation

My video implementation follows the methodology described above for each frame, where each image or frame is passed through `process_image()`. In order to efficiently track time-series bounding box information and to get rid of old bounding-boxes beyond 9 most recent bounding boxes, I use a deque function (`from collections import deque` - boxes = deque() is defined on top of `main.py`). These 9 most recent bounding boxes are used for heatmap definition, thresholding and final bounding box identification.

I am attaching videos for two difference set of scales I experimented with and which come very close in their performance:

* `scales = [0.75, 1.0, 1.5]`: 
	* Here's a [link to video with vehicles-detected](./output_videos/output_project_vehicle_detect_set2_scales1.mp4)
	* Here's a [link to video with vehicles-detected & lanes-marked](./output_videos/output_project_lane_vehicle_detect_set2_scales1.mp4)

* `scales = [1.0, 1.5, 2.0]`: 
	* Here's a [link to my video with vehicles-detected](./output_videos/output_project_vehicle_detect_set2_scales2.mp4)
	* Here's a [link to video with vehicles-detected & lanes-marked](./output_videos/output_project_lane_vehicle_detect_set2_scales2.mp4)

---

###Discussion

* Developing a robust vehicle-detection method seems like a balancing act, where endless tuning of so many different parameters is needed to arrive at a failry ok schele.

* Calculating hog features once for the entire image and then subsampling over sliding windows is a useful trick, as it saves time and resource that would have been necessary if we were to gradients at every pixel on the fly.

* Using different color spectrums together might be useful (instead of just using one, like YCrCb, as I have done). However, this might make the computation slow, so there is a trade-off that needs to be addressed in decision making.

* The pipeline is likely to fail in poor lighting conditions, dim-lighting, night-time driving, fog, rain, etc. Even with the current set of images, we get a lot of false positives if a parameter is changed just slightly. Combining camera images with some other sensors, like radars might be beneficial.



