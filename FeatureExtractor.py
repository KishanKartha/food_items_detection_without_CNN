from numpy import pi, exp, sqrt
import numpy as np
import cv2
import pandas as pd


# import os


def get_features(cropped_imgs: list) -> pd.DataFrame:
    """
    :rtype: pandas.Dataframe
    """
    # Gaussian kernel for giving more importance to center portion
    s, k = 50, 100  # generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
    probs = [exp(-z * z / (2 * s * s)) / sqrt(2 * pi * s * s) for z in range(-k, k)]
    gaussian_kernel = np.outer(probs, probs)

    circle = np.zeros((200, 200), dtype="uint8")
    cv2.circle(circle, (100, 100), 50, 255, -1)
    # cv2.imshow("mask", circle)

    # feature extraction from cropped images
    df = pd.DataFrame()
    num = 1
    for img_original in cropped_imgs:

        img_resized = img_original

        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

        centerimg = img_gray * gaussian_kernel
        centerimg = cv2.normalize(src=centerimg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8U)
        imagecenterforhist = cv2.bitwise_and(img_resized, img_resized, mask=circle)

        filtered_img_vector = []

        # Generate Gabor features

        # kernels = []  # Create empty list to hold all kernels that we will generate in a loop
        for theta in range(3):  # number of thetas.
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  # Sigma with values of 1 and 3
                for lamda in [np.pi / 8, np.pi / 4, np.pi / 2, np.pi]:  # Range of wavelengths
                    for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5

                        ksize = 9
                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                        # kernels.append(kernel)
                        # Now filter the image and add values to a new column
                        fimg = cv2.filter2D(centerimg, cv2.CV_8UC3, kernel)
                        filtered_img_vector.append(fimg.mean())
                        filtered_img_vector.append(fimg.std())

        # color data
        (B, G, R) = cv2.split(imagecenterforhist)
        filtered_img_vector.append(B.mean())
        filtered_img_vector.append(B.std())
        filtered_img_vector.append(G.mean())
        filtered_img_vector.append(G.std())
        filtered_img_vector.append(R.mean())
        filtered_img_vector.append(R.std())

        # insert label

        df[num] = filtered_img_vector
        num = num + 1
    df = df.T
    return df

# df.to_csv()
