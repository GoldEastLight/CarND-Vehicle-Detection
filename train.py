from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
import os.path
import time
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from p5 import *


train_parameters = {
    'color_space': 'HSV',
    'orient': 9,
    'pix_per_cell': 8,
    'cell_per_block': 2,
    'hog_channel': 'ALL',
    'spatial_size': (16, 16),
    'hist_bins': 32,
    'spatial_feat': True,
    'hist_feat': True,
    'hog_feat': True,
}

model = {
    'svc': None,
    'X_scaler': None,
}


def train(parmeters = train_parameters):
    vehicles_files = glob.glob('./vehicles/**/*.png', recursive=True)
    non_vehicles_files = glob.glob('./non-vehicles/**/*.png', recursive=True)

    sample_size = -1
    cars = vehicles_files[0:sample_size]
    notcars = non_vehicles_files[0:sample_size]

    color_space = parmeters['color_space']  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = parmeters['orient']  # HOG orientations
    pix_per_cell = parmeters['pix_per_cell']  # HOG pixels per cell
    cell_per_block = parmeters['cell_per_block']  # HOG cells per block
    hog_channel = parmeters['hog_channel']  # Can be 0, 1, 2, or "ALL"
    spatial_size = parmeters['spatial_size']  # Spatial binning dimensions
    hist_bins = parmeters['hist_bins']  # Number of histogram bins
    spatial_feat = parmeters['spatial_feat']  # Spatial features on or off
    hist_feat = parmeters['hist_feat']  # Histogram features on or off
    hog_feat = parmeters['hog_feat']  # HOG features on or off

    t = time.time()
    car_features = extract_features(cars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
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
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1, 1]}
    # svr = SVC()
    # clf = GridSearchCV(svr, parameters)
    # clf.fit(X_train, y_train)
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC

    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return svc, X_scaler


def load_data():
    pickle_file = 'model.p'
    global model
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            param = pickle.load(f)
            model.update(param)
    else:
        svc, X_scaler = train()
        model['X_scaler'] = X_scaler
        model['svc'] = svc

        with open(pickle_file, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def detect_car(image, parameters=train_parameters, model=model):
    color_space = train_parameters['color_space']  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = train_parameters['orient']  # HOG orientations
    pix_per_cell = train_parameters['pix_per_cell']  # HOG pixels per cell
    cell_per_block = train_parameters['cell_per_block']  # HOG cells per block
    spatial_size = train_parameters['spatial_size']  # Spatial binning dimensions
    hist_bins = train_parameters['hist_bins']  # Number of histogram bins
    X_scaler = model['X_scaler']
    svc = model['svc']

    scale_configs = ((400, 500, 0.5), (400, 550, 1), (400, 650, 1.5), (400, 700, 2))

    box_lists = []
    t = time.time()
    for config in scale_configs:
        ystart = config[0]
        ystop = config[1]
        scale = config[2]
        box_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        box_lists.extend(box_list)
    label_draw_img, heatmap = draw_labeled_boxes(image, box_lists, 8)
    box_draw_img = draw_boxes(image, box_lists)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to prediction...')
    return label_draw_img, box_draw_img, heatmap


def test():
    image = mpimg.imread('./test_images/test4.jpg')
    draw_image = np.copy(image)
    print(image.shape)
    y_start_stop = [np.int(image.shape[0]*(2/3)), image.shape[0]]
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255
    # Check the prediction time for a single sample

    # windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #                     xy_window=(96, 96), xy_overlap=(0.5, 0.5))
    #
    # hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
    #                         spatial_size=spatial_size, hist_bins=hist_bins,
    #                         orient=orient, pix_per_cell=pix_per_cell,
    #                         cell_per_block=cell_per_block,
    #                         hog_channel=hog_channel, spatial_feat=spatial_feat,
    #                         hist_feat=hist_feat, hog_feat=hog_feat)
    #
    # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    #
    #
    #
    # plt.imshow(window_img)
    # plt.show()

    ystart = 400
    ystop = 656
    scale = 1.5

    t = time.time()
    box_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    draw_img, heatmap = draw_labeled_boxes(image, box_list, 4)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to prediction...')
    # plt.imshow(out_img)
    # plt.show()

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    load_data()

    color_space = train_parameters['color_space']  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = train_parameters['orient']  # HOG orientations
    pix_per_cell = train_parameters['pix_per_cell']  # HOG pixels per cell
    cell_per_block = train_parameters['cell_per_block']  # HOG cells per block
    hog_channel = train_parameters['hog_channel']  # Can be 0, 1, 2, or "ALL"
    spatial_size = train_parameters['spatial_size']  # Spatial binning dimensions
    hist_bins = train_parameters['hist_bins']  # Number of histogram bins
    spatial_feat = train_parameters['spatial_feat']  # Spatial features on or off
    hist_feat = train_parameters['hist_feat']  # Histogram features on or off
    hog_feat = train_parameters['hog_feat']  # HOG features on or off
    X_scaler = model['X_scaler']
    svc = model['svc']
    y_start_stop = [None, None]  # Min and max in y to search in slide_window()

    test_files = glob.glob('./test_images/*.jpg')

    for test_file in test_files:
        image = mpimg.imread(test_file)
        label_draw_img, box_draw_img, heatmap = detect_car(image)

        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(label_draw_img)
        plt.title('Car Positions')
        plt.subplot(132)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        plt.subplot(133)
        plt.imshow(box_draw_img)
        plt.title('box')
        fig.tight_layout()
        plt.show()