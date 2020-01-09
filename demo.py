import random
import time

import numpy as np
import torch
import cv2
import imutils
import streamlit as st
import matplotlib.pyplot as plt

from models import Darknet, load_darknet_weights
from utils.parse_config import parse_data_cfg
from utils.utils import load_classes, non_max_suppression, plot_one_box, scale_coords
from utils.datasets import letterbox

archs = {
    "YOLOv3-tiny-3l-ultralytics": {
        "cfg": "my_cfg/yolov3-tiny_3l-corn.cfg",
        "weights": "backup/best.pt",
        "default_img_size": 608,
    },
    "YOLOv3-tiny-416": {
        "cfg": "my_cfg/yolov3-tiny-corn-416.cfg",
        "weights": "backup/yolov3-tiny-corn-416_best.weights",
        "default_img_size": 416,
    },
    "YOLOv3-tiny-608": {
        "cfg": "my_cfg/yolov3-tiny-corn-608.cfg",
        "weights": "backup/yolov3-tiny-corn-608_best.weights",
        "default_img_size": 608,
    },
    "YOLOv3-tiny-3l-608": {
        "cfg": "my_cfg/yolov3-tiny_3l-corn.cfg",
        "weights": "backup/yolov3-tiny_3l-corn_best.weights",
        "default_img_size": 608,
    }
}

def get_colors(n):
    # colors = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors + \
        # plt.get_cmap("tab20c").colors
    # while len(colors) < n:
    #     colors *= 2
    # colors = colors[:n]
    # colors = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
    #           for c in colors]

    colors = [
        (102, 255, 51),
        (153, 0, 204),
        (153, 102, 0),
        (255, 102, 0),
        (0, 204, 255),
    ]
    colors = [c[::-1] for c in colors]  # RGB to BGR
    return colors[:n]


@st.cache(allow_output_mutation=True)
def get_model(cfg_path, weights_path, img_size, device):
    model = Darknet(cfg_path, img_size)
    # Load weights
    if weights_path.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(
            weights_path, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights_path)
    # Eval mode
    model.to(device).eval()
    return model


def load_image(img_size):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)

    # Padded resize
    resized_img = letterbox(original_img, new_shape=img_size)[0]
    # Normalize RGB
    resized_img = resized_img[:, :, ::-1]  # BGR to RGB
    resized_img = np.ascontiguousarray(
        resized_img, dtype=np.float32)  # uint8 to fp16/fp32
    resized_img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return original_img, resized_img


def main():
    # Initialize model
    device = "cpu"
    cfg_path = archs[arch]["cfg"]
    weights_path = archs[arch]["weights"]

    model = get_model(cfg_path, weights_path, img_size, device)

    # Get classes and colors
    classes = load_classes(parse_data_cfg("data/corn.data")['names'])
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(classes))]
    colors = get_colors(len(classes))
    # Run inference
    x = torch.from_numpy(resized_img).to(device)
    x = x.permute(2, 0, 1)
    if x.ndimension() == 3:
        x = x.unsqueeze(0)

    start = time.time()
    pred = model(x)[0]
    time_forward = time.time() - start

    start = time.time()
    # Apply NMS
    pred = non_max_suppression(pred, conf_thresh, nms_thresh)
    # Process detections
    res_img = original_img.copy()
    res_height = 900
    res_txt = "* **Detections:**\n"
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            # print(resized_img.shape)
            # print(original_img.shape)
            print(det.shape)
            det[:, :4] = scale_coords(
                resized_img.shape, det[:, :4], original_img.shape).round()
            print(det.shape)

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                res_txt += f"    * {classes[int(c)]}: {n}\n"

            # Write results
            for *xyxy, conf, cls in det:
                label = f"{classes[int(cls)]} {conf:.2f}"
                # print(xyxy)
                plot_one_box(xyxy, res_img, label=label,
                             color=colors[int(cls)],
                             font_scale=0.2)
    time_postprocess = time.time() - start

    st.markdown(res_txt)
    st.markdown(f"""
    * **Total time taken:** {time_forward + time_postprocess:.2f}s
        * Forward pass: {time_forward:.2f}s
        * Postprocessing: {time_postprocess:.2f}s
    """)

    if res_img.shape[0] > res_height:
        res_img = imutils.resize(res_img, height=res_height)
    if res_img.shape[1] > res_height:
        res_img = imutils.resize(res_img, width=res_height)
    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    st.image(res_img)
    print(f"Done {time_forward + time_postprocess:.2f}s")



st.title("Corn quality assessment")

uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
st.sidebar.title("Model")
arch = st.sidebar.selectbox("architecture", sorted(list(archs.keys())))

img_size = st.sidebar.slider(
    "image input size", min_value=320, max_value=832, step=32,
    value=archs[arch]["default_img_size"])
conf_thresh = st.sidebar.slider("confidence threshold", min_value=0.,
                                max_value=1., step=0.05, value=0.25)
nms_thresh = st.sidebar.slider(
    "nms threshold", min_value=0., max_value=1., step=0.05, value=0.4)


if uploaded_file is not None:
    original_img, resized_img = load_image(img_size)
    main()
# # Load image
# original_img, img = load_image("data/test/IMG_0256.jpg", img_size)
