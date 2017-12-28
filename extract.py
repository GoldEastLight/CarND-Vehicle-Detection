import numpy as np
import cv2
from skimage.feature import hog


class FeatureExtractor():
    def __init__(self, parameters, model):
        self.parameters = parameters
        self.model = model
        self.c_color_space = parameters['c_color_space']  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.h_color_space = parameters['h_color_space']
        self.orient = parameters['orient']  # HOG orientations
        self.pix_per_cell = parameters['pix_per_cell']  # HOG pixels per cell
        self.cell_per_block = parameters['cell_per_block']  # HOG cells per block
        self.spatial_size = parameters['spatial_size']  # Spatial binning dimensions
        self.hist_bins = parameters['hist_bins']  # Number of histogram bins
        self.spatial_feat = parameters['spatial_feat']  # Spatial features on or off
        self.hist_feat = parameters['hist_feat']  # Histogram features on or off
        self.hog_feat = parameters['hog_feat']  # HOG features on or off
        self.hog_channel = parameters['hog_channel']
        self.X_scaler = model['X_scaler']
        self.svc = model['svc']
        self.hog_features = None

    def get_hog_features(self, img, vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block),
                                      block_norm='L2-Hys',
                                      transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=self.orient,
                           pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cell_per_block, self.cell_per_block),
                           block_norm='L2-Hys',
                           transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

    # Define a function to compute binned color features
    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    # # Define a function to compute color histogram features
    # # NEED TO CHANGE bins_range if reading .png files with mpimg!
    # def color_hist(self, img, nbins=32, bins_range=(0, 255)):
    #     # print("color_hist", np.max(img))
    #     # Compute the histogram of the color channels separately
    #     channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    #     channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    #     channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    #     # Concatenate the histograms into a single feature vector
    #     hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    #     # hist_features = hist_features/np.max(hist_features)
    #     # Return the individual histograms, bin_centers and feature vector
    #     return hist_features

    # Define a function to compute color histogram features
    # NEED TO CHANGE bins_range if reading .png files with mpimg!
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # print("color_hist", np.max(img))
        # Compute the histogram of the color channels separately
        channel1_hist = cv2.calcHist([img], [0], None, [nbins], bins_range).ravel()
        channel2_hist = cv2.calcHist([img], [1], None, [nbins], bins_range).ravel()
        channel3_hist = cv2.calcHist([img], [2], None, [nbins], bins_range).ravel()

        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist, channel2_hist, channel3_hist))
        # hist_features = hist_features/np.max(hist_features)
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def convert_color(self, img, conv='YCrCb'):
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if conv == 'YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        elif conv == 'LUV':
            return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif conv == 'HSV':
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif conv == 'HLS':
            return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif conv == 'YUV':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif conv == 'RGB':
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif conv == 'LAB':
            return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        else:
            feature_image = np.copy(img)
            print("Covert_color: No Color")
            return feature_image


    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_color_features(self, feature_image):
        spatial_size = self.spatial_size
        hist_bins = self.hist_bins
        spatial_feat = self.spatial_feat
        hist_feat = self.hist_feat

        if spatial_feat:
            self.spatial_features = self.bin_spatial(feature_image, size=spatial_size)
        else:
            self.spatial_features = None

        if hist_feat:
            self.hist_features = self.color_hist(feature_image, nbins=hist_bins)
        else:
            self.hist_features = None

        return self.spatial_features, self.hist_features

    def extract_hog_features(self, feature_image):
        hog_channel = self.hog_channel
        hog_feat = self.hog_feat

        if hog_feat:
            if hog_channel == 6:
                hog1 = self.get_hog_features(feature_image[:, :, 0], feature_vec=False)
                hog2 = self.get_hog_features(feature_image[:, :, 1], feature_vec=False)
                hog3 = self.get_hog_features(feature_image[:, :, 2], feature_vec=False)
                self.hog_features = np.dstack((hog1, hog2, hog3))
            elif hog_channel == 3:
                hog1 = self.get_hog_features(feature_image[:, :, 0], feature_vec=False)
                hog2 = self.get_hog_features(feature_image[:, :, 1], feature_vec=False)
                self.hog_features = np.dstack((hog1, hog2))
            elif hog_channel == 4:
                hog1 = self.get_hog_features(feature_image[:, :, 0], feature_vec=False)
                hog3 = self.get_hog_features(feature_image[:, :, 2], feature_vec=False)
                self.hog_features = np.dstack((hog1, hog3))
            elif hog_channel == 5:
                hog2 = self.get_hog_features(feature_image[:, :, 1], feature_vec=False)
                hog3 = self.get_hog_features(feature_image[:, :, 2], feature_vec=False)
                self.hog_features = np.dstack((hog2, hog3))
            else:
                hog = self.get_hog_features(feature_image[:, :, hog_channel], feature_vec=False)
                self.hog_features = hog
                # hog_feature.append(hog3)
            # print(self.hog_features.shape)
        else:
            self.hog_features = None

        return self.hog_features

    def get_feature_image(self, origin_img, color_space, ystart=None, ystop=None):
        if ystart is None:
            ystart = 0
        if ystop is None:
            ystop = origin_img.shape[0]
        img_toextract = origin_img[ystart:ystop, :, :]
        feature_image = self.convert_color(img_toextract, color_space)
        return feature_image

    def flat_features(self, spatial_features, hist_features, hog_features):
        features = []
        if spatial_features is not None:
            features.extend(spatial_features)
        if hist_features is not None:
            features.extend(hist_features)
        if hog_features is not None:
            for ch in range(hog_features.shape[2]):
                hog_channel = hog_features[:, :, ch]
                hog_feature = hog_channel.ravel()
            # for ch in range(len(hog_features)):
            #     hog_feature = hog_features[ch].ravel()
                features.extend(hog_feature)
        return features

    def extract_features(self, image_files):
        features = []
        for filename in image_files:
            img = cv2.imread(filename)
            features_img = self.get_feature_image(img, self.c_color_space)
            self.extract_color_features(features_img)
            if self.h_color_space != self.c_color_space:
                features_img = self.get_feature_image(img, self.h_color_space)
            self.extract_hog_features(features_img)
            features.append(self.flat_features(self.spatial_features, self.hist_features, self.hog_features))
        return np.array(features)

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, ystart, ystop, scale, step):
        ctrans_tosearch = self.get_feature_image(img, self.c_color_space, ystart, ystop)
        if self.h_color_space == self.c_color_space:
            htrans_tosearch = ctrans_tosearch
        else:
            htrans_tosearch = self.get_feature_image(img, self.h_color_space, ystart, ystop)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
            if self.h_color_space == self.c_color_space:
                htrans_tosearch = ctrans_tosearch
            else:
                htrans_tosearch = cv2.resize(htrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
        # Define blocks and steps as above
        nxblocks = (ctrans_tosearch.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ctrans_tosearch.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = step  # parameters['cells_per_step']  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        hog_features = self.extract_hog_features(htrans_tosearch)

        bbox_list = []
        slide_boxes = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                img_features = []
                # Get color features
                if self.spatial_feat:
                    spatial_features = self.bin_spatial(subimg, size=self.spatial_size)
                    img_features.append(spatial_features)
                if self.hist_feat:
                    hist_features = self.color_hist(subimg, nbins=self.hist_bins)
                    img_features.append(hist_features)

                if self.hog_feat:
                    for ch in range(hog_features.shape[2]):
                        hog_channel = hog_features[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window, ch]
                        hog_feature = hog_channel.ravel()
                    # for ch in range(len(hog_features)):
                    #     hog = hog_features[ch]
                    #     hog_feature = hog[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        img_features.append(hog_feature)

                img_features = np.concatenate(img_features).reshape(1, -1)
                # Scale features and make a prediction
                test_features = self.X_scaler.transform(img_features)
                test_prediction = self.svc.predict(test_features)
                test_confidence = self.svc.decision_function(test_features)

                # if test_prediction == 1:
                #     # print(np.max(test_features), np.min(test_features), np.mean(test_features))
                #     print(test_prediction, test_confidence)

                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                box = ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart))
                slide_boxes.append(box)
                if test_prediction == 1 and test_confidence > 0.2:
                    # print(test_prediction, test_confidence)
                    bbox_list.append(box)
                    if test_confidence > 0.8:
                        bbox_list.append(box)
                    if test_confidence > 1.2:
                        bbox_list.append(box)
                    if test_confidence > 1.4:
                        bbox_list.append(box)

        return np.array(bbox_list), np.array(slide_boxes)


