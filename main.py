from tracker import load_data, Tracker
from utils import resize_image
import cv2
import numpy as np


load_data()
tracker = Tracker()


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

    if not ret:
        break
    print("--------current frame:", frame_idx)

    layer1_input_boxes_img, layer1_output_img, layer2_input_boxes_img, layer2_output_img = tracker.detect_car(frame)
    # print(label_draw_img.shape, heatmap.shape)
    # print(np.max(label_draw_img), np.max(heatmap))
    scale = 0.4
    resize_shape = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    layer1_input_boxes_img_resized = resize_image(layer1_input_boxes_img, resize_shape, "layer1_input_boxes_img")
    layer1_output_img_resized = resize_image(layer1_output_img, resize_shape, "layer1_output_img")
    layer2_input_boxes_img_resized = resize_image(layer2_input_boxes_img, resize_shape, 'layer2_input_boxes_img')
    layer2_output_img_resized = resize_image(layer2_output_img, resize_shape, 'layer2_output_img')


    # heatmap_resized = resize_image(heatmap, resize_shape, 'heatmap')

    # print(np.max(labeled_img_l1_resized), np.max(heatmap_resized))
    img_h1 = np.hstack((layer1_input_boxes_img_resized, layer1_output_img_resized))
    img_h2 = np.hstack((layer2_input_boxes_img_resized, layer2_output_img_resized))
    img = np.vstack((img_h1, img_h2))
    cv2.imshow(w_name, img)
    # cv2.imshow('2', heatmap_resized)
    # out_debug.write(draw_img)
    # out_project.write(heatmap)

cap.release()
cv2.destroyAllWindows()