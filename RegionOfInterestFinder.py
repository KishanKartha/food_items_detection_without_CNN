from sklearn.cluster import KMeans
import numpy as np
import cv2
import os

# %matplotlib inline
# from google.colab.patches import cv2_imshow

import sys
import random as rand
import GraphOperator as go


# ---------------------------------------------


def get_roi(input_img_path):
    """
    :rtype: list
    """

    def generate_image(ufset, width, height):
        random_color = lambda: (int(rand.random() * 255), int(rand.random() * 255), int(rand.random() * 255))
        color = [random_color() for i in range(width * height)]
        # print(color)

        save_img = np.zeros((height, width, 3), np.uint8)
        for y in range(height):
            for x in range(width):
                color_idx = ufset.find(y * width + x)
                # print(color_idx)
                # print("#######")
                save_img[y, x] = color[color_idx]
        return save_img

    def segmentation(sigma, k, min_size, input_image_file, output_image_file):
        # sigma = float(sys.argv[1])
        # k = float(sys.argv[2])
        # min_size = float(sys.argv[3])

        img = cv2.imread(input_image_file)
        float_img = np.asarray(img, dtype=float)

        gaussian_img = cv2.GaussianBlur(float_img, (5, 5), sigma)
        b, g, r = cv2.split(gaussian_img)
        smooth_img = (r, g, b)

        height, width, channel = img.shape
        graph = go.build_graph(smooth_img, width, height)

        weight = lambda edge: edge[2]
        sorted_graph = sorted(graph, key=weight)

        ufset = go.segment_graph(sorted_graph, width * height, k)
        ufset = go.remove_small_component(ufset, sorted_graph, min_size)

        save_img = generate_image(ufset, width, height)
        # cv2.imwrite(output_image_file, save_img)
        # print(type(save_img))
        return (save_img)

    save_img = segmentation(0.5, 500, 50, input_img_path, r"/content/010102001.jpg")
    # sigma k min_size input_image_file output_image_file

    # Step 2: Kmeans on the segmented image, to find the color cluster

    modified_image = save_img.reshape(save_img.shape[0] * save_img.shape[1], 3)

    clf = KMeans(n_clusters=10)  # number of clusters taken arbitarily
    labels = clf.fit_predict(modified_image)

    new_labels = np.reshape(labels, (save_img.shape[0], save_img.shape[1]))

    new_labels = cv2.normalize(src=new_labels, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Step 3: extract postions from the image belonging to a particular label(predicted by Kmeans)

    # # Read color image
    orginal_img = cv2.imread(input_img_path)

    masked_img = []

    # # Iterate all colors in mask
    for color in np.unique(new_labels):
        m = np.uint8(new_labels == color)
        k = cv2.bitwise_and(orginal_img, orginal_img, mask=m)
        # median = cv2.medianBlur(k,7)
        median = cv2.medianBlur(k, 17)
        # median = cv2.medianBlur(median,11)
        # median = cv2.medianBlur(median,7)
        masked_img.append(median)

    # Step 4: find the contours from each extracted image(from previous step)
    # Step 5: draw bounding boxes on the orginal image based on the contours.
    # Step 6: crop the regions of interest

    out = orginal_img.copy()

    cropped_imgs = []

    for img in masked_img:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # dictinary with keys as lenght of the contours and values as contours
        # then the dictionar is sorted in descending order
        # first four largest contours are selected.

        d = {}

        for y in contours:
            #   d[len(y)] = y
            # g = list(d)

            # g.sort(reverse=True)

            iw, ih, ic = save_img.shape

            # for x in (0,1,2,3,4):
            x, y, w, h = cv2.boundingRect(y)

            if w > iw * 0.8 or h > ih * 0.8:
                continue
            if w < iw * 0.15 or h < ih * 0.15:
                continue

            out = cv2.rectangle(out, (x, y), (x + w, y + h), (0, 235, 0), 2)
            x1 = x - w / 2
            y1 = y - h / 2

            # Show image with bounding box
            # cv2_imshow(out)

            cropped_imgs.append(cv2.resize(orginal_img[int(y):int(y + h), int(x):int(x + w)], (200, 200)))
    cv2.imshow(" image with bounding boxes", out)
    cv2.waitKey(0)
    # closing all open windows
    cv2.destroyAllWindows()

    return (cropped_imgs)
