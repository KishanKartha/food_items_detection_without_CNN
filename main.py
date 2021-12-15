import RegionOfInterestFinder as roi
import FeatureExtractor as fe
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import cv2


# show contents in cropped_imgs
def showimages(cropped_imgs,y_probs = []):  # datatype is list of images

    fig = plt.figure(figsize=(20., 20.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(5, 5),
                     axes_pad=0.4,  # pad between axes in inch.
                     )
    if len(y_probs) == 0:
        for ax, im in zip(grid, cropped_imgs):
            # Iterating over the grid returns the Axes.
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    else:
        for ax, im, p in zip(grid, cropped_imgs, y_probs):
            if (np.max(p) * 100) < 40:
                continue

            title = food[int(np.where(p == np.max(p))[0])] + " " + str(round((np.max(p) * 100), 2))
            # Iterating over the grid returns the Axes.
            ax.set_title(title, fontdict=None, loc='center', color="k")
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    plt.show()


cropped_imgs: list = roi.get_roi(r"pictures/3.jpg")

showimages(cropped_imgs)

df = fe.get_features(cropped_imgs)

test_features = np.array(df)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
test_features = scaler.fit_transform(test_features)

#classification

from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

import json

with open("models/model1_131221_fulldata.json",'r') as f:
    model_json = json.load(f)

kish = model_from_json(model_json)
kish.load_weights("models/model1_131221_fulldata.h5")

food = ['Meat', 'Noodles-Pasta', 'Rice', 'Soup']

y_probs = kish.predict(test_features, verbose=0)

showimages(cropped_imgs,y_probs)