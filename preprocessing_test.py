import streamlit as st
from PIL import Image
import cv2
import math
import numpy as np
from typing import Tuple
from deskew import determine_skew

BILATERAL_FILTER_DIAMETER = 5
BILATERAL_FILTER_SIGMA_COLOR = 55
BILATERAL_FILTER_SIGMA_SPACE = 60

ADAPTIVE_THRESH_BLOCK_SIZE = 21
ADAPTIVE_THRESH_C = 4


def deskew_img(image: np.ndarray) -> np.ndarray:
    try:
        angle = determine_skew(image)
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(
            np.cos(angle_radian) * old_width
        )
        height = abs(np.sin(angle_radian) * old_width) + abs(
            np.cos(angle_radian) * old_height
        )

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(
            image,
            rot_mat,
            (int(round(height)), int(round(width))),
            borderValue=(0, 0, 0),
        )
    except:
        return image


def convert_img(img: "Image") -> np.ndarray:
    img = np.array(img)
    return img


def normalize_img(img: np.ndarray) -> np.ndarray:
    return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


def denoise_img(img: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img, BILATERAL_FILTER_DIAMETER, BILATERAL_FILTER_SIGMA_COLOR, BILATERAL_FILTER_SIGMA_SPACE)


def grayscale_img(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def threshold_img(img: np.ndarray, threshold_val: int) -> np.ndarray:
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C)
    return thresh

def preprocess(img):
  a, col0, b = st.columns([1, 20, 1])
  col1, col2, col3 = st.columns([1, 1, 1])
  col4, col5, col6 = st.columns([1, 1, 1])


  img = convert_img(img)
  with col2.container(border=True):
      st.image(img, output_format="auto", caption="original image")

  img = normalize_img(img)
  with col4.container(border=True):
      st.image(img, output_format="auto", caption="normalized image")

  img = grayscale_img(img)
  with col5.container(border=True):
      st.image(img, output_format="auto", caption="grayscale image")

  img = denoise_img(img)

  img = deskew_img(img)
  with col6.container(border=True):
      st.image(img, output_format="auto", caption="deskew image")

  img = threshold_img(img, threshold_val=40)
  with col3.container(border=True):
      st.image(img, output_format="auto", caption="threshold image")

  img = Image.fromarray(img)
  return img