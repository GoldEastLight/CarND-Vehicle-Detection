from p5 import *
from train import *

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


f1 = 'project_video.mp4'
f2 = 'test_video.mp4'

# Change Input File Here

input_file = f1

w_name = input_file
cv2.namedWindow(w_name, cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(input_file)


def progress_bar_cb(x):
    cap.set(cv2.CAP_PROP_POS_FRAMES, x)


cv2.createTrackbar('Frame', w_name, 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), progress_bar_cb)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
prefix = input_file.split('.')[0]
out_debug = cv2.VideoWriter(prefix + '_debug.avi', fourcc, 20.0, (1152, 864))
out_project = cv2.VideoWriter(prefix + '_project.avi', fourcc, 20.0, (1280, 720))
delay = 1
while (cap.isOpened()):
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break

    ret, frame = cap.read()
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos('Frame', w_name, frame_idx)


    print("--------current frame:", frame_idx)

    label_draw_img, box_draw_img, heatmap = detect_car(frame)
    print(label_draw_img.shape, heatmap.shape)
    print(np.max(label_draw_img), np.max(heatmap))
    scale = 0.5
    resize_shape = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    draw_img_resized = resize_image(label_draw_img, resize_shape, "box")
    heatmap_resized = resize_image(heatmap, resize_shape, 'heatmap')
    # print(final_img.shape)
    # img = np.hstack((draw_img_resized, heatmap_resized))
    cv2.imshow(w_name, draw_img_resized)
    cv2.imshow('2', box_draw_img)
    # out_debug.write(draw_img)
    # out_project.write(heatmap)

cap.release()
cv2.destroyAllWindows()