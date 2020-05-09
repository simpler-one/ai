import numpy as np
import cv2
import math

# IMG_PATH = "./data/termination/data/0.png"
IMG_PATH = "./data/1.png"
FEAT_DETECTOR = cv2.FastFeatureDetector_create(nonmaxSuppression=True)


def main():
    org_img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(org_img, -1, 255, cv2.THRESH_OTSU)
    img = img.astype(np.int16)

    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = to_skeleton(img, ellipse, iteration=4)
    img = np.clip(img, 0, 255)

    skeleton = img.astype(np.uint8)

    img = cv2.filter2D(img, -1, neighbor_filter())
    neighbor_img = np.clip(img, 0, 255).astype(np.uint8)

    fil = get_filter7x7()
    img = cv2.filter2D(img, -1, fil)
    print(np.max(img), np.min(img))
    img = np.clip(img, 0, 255).astype(np.uint8)
    # img = img * (255 / max(np.max(img), 1))

    img = np.clip(img, 0, 255).astype(np.uint8)
    # _, img = cv2.threshold(img, -1, 255, cv2.THRESH_OTSU)
    print(np.max(img), np.min(img))

    cv2.namedWindow("ORG IMAGE", cv2.WINDOW_NORMAL)
    cv2.imshow("ORG IMAGE", org_img)

    cv2.namedWindow("SKELETON IMAGE", cv2.WINDOW_NORMAL)
    cv2.imshow("SKELETON IMAGE", skeleton)

    cv2.namedWindow("NEIGHBOR IMAGE", cv2.WINDOW_NORMAL)
    cv2.imshow("NEIGHBOR IMAGE", neighbor_img)

    cv2.namedWindow("IMAGE", cv2.WINDOW_NORMAL)
    cv2.imshow("IMAGE", img)
    cv2.waitKey()


def to_skeleton(image, kernel, iteration=1):
    img = image

    for _ in range(iteration):
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        tophat = image - opened
        img = eroded | tophat

    return img


def get_filter7x7():
    base = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 1, 0, 0],
        [0, 0, 2, 5, 2, 0, 0],
        [0, 0, 1, 2, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])

    distance = np.array([
        # [4.2, 3.6, 3.2, 3.0, 3.2, 3.6, 4.2],
        # [3.6, 2.8, 2.2, 2.0, 2.2, 2.8, 3.6],
        # [3.2, 2.2, 1.4, 1.0, 1.4, 2.2, 3.2],
        # [3.0, 2.0, 1.0,   0, 1.0, 2.0, 3.0],
        # [3.2, 2.2, 1.4, 1.0, 1.4, 2.2, 3.2],
        # [3.6, 2.8, 2.2, 2.0, 2.2, 2.8, 3.6],
        # [4.2, 3.6, 3.2, 3.0, 3.2, 3.6, 4.2],
        [0.0, 0.0, 3.2, 3.0, 3.2, 0.0, 0.0],
        [0.0, 2.8, 2.2, 2.0, 2.2, 2.8, 0.0],
        [3.2, 2.2, 1.4, 1.0, 1.4, 2.2, 3.2],
        [3.0, 2.0, 1.0,   0, 1.0, 2.0, 3.0],
        [3.2, 2.2, 1.4, 1.0, 1.4, 2.2, 3.2],
        [0.0, 2.8, 2.2, 2.0, 2.2, 2.8, 0.0],
        [0.0, 0.0, 3.2, 3.0, 3.2, 0.0, 0.0],
    ])

    return (base * 3 ** 2 - 2 * distance ** 2) / 7 ** 2


def get_filter9x9():
    distance = np.array([
        [5.7, 5.0, 4.5, 4.1, 4.0, 4.1, 4.5, 5.0, 5.7],
        [5.0, 4.2, 3.6, 3.2, 3.0, 3.2, 3.6, 4.2, 5.0],
        [4.5, 3.6, 2.8, 2.2, 2.0, 2.2, 2.8, 3.6, 4.5],
        [4.1, 3.2, 2.2, 1.4, 1.0, 1.4, 2.2, 3.2, 4.1],
        [4.0, 3.0, 2.0, 1.0,   0, 1.0, 2.0, 3.0, 4.0],
        [4.1, 3.2, 2.2, 1.4, 1.0, 1.4, 2.2, 3.2, 4.1],
        [4.5, 3.6, 2.8, 2.2, 2.0, 2.2, 2.8, 3.6, 4.5],
        [5.0, 4.2, 3.6, 3.2, 3.0, 3.2, 3.6, 4.2, 5.0],
        [5.7, 5.0, 4.5, 4.1, 4.0, 4.1, 4.5, 5.0, 5.7],
    ])

    base = 1.5 * 4 ** 2
    f = 1.0 / 9 ** 2
    return (base - distance ** 3) * f


def get_filter11x11():
    distance = np.array([
        [7.1, 6.4, 5.8, 5.4, 5.1, 5.0, 5.1, 5.4, 5.8, 6.4, 7.1],
        [6.4, 5.7, 5.0, 4.5, 4.1, 4.0, 4.1, 4.5, 5.0, 5.7, 6.4],
        [5.8, 5.0, 4.2, 3.6, 3.2, 3.0, 3.2, 3.6, 4.2, 5.0, 5.8],
        [5.4, 4.5, 3.6, 2.8, 2.2, 2.0, 2.2, 2.8, 3.6, 4.5, 5.4],
        [5.1, 4.1, 3.2, 2.2, 1.4, 1.0, 1.4, 2.2, 3.2, 4.1, 5.1],
        [5.0, 4.0, 3.0, 2.0, 1.0,   0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [5.1, 4.1, 3.2, 2.2, 1.4, 1.0, 1.4, 2.2, 3.2, 4.1, 5.1],
        [5.4, 4.5, 3.6, 2.8, 2.2, 2.0, 2.2, 2.8, 3.6, 4.5, 5.4],
        [5.8, 5.0, 4.2, 3.6, 3.2, 3.0, 3.2, 3.6, 4.2, 5.0, 5.8],
        [6.4, 5.7, 5.0, 4.5, 4.1, 4.0, 4.1, 4.5, 5.0, 5.7, 6.4],
        [7.1, 5.7, 5.0, 4.5, 4.1, 4.0, 4.1, 4.5, 5.0, 5.7, 7.1],
    ])

    base = 3 * 4 ** 2
    f = 1.0 / 9 ** 2
    return (base - distance ** 3) * f


# -----
main()
