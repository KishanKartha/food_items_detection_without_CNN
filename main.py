import RegionOfInterestFinder as roi
import FeatureExtractor as fe
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json


# to extract features from the images and create a pandas dataframe
def convert_training_data_to_csv(directory = "training_imagedata/"):

    df_train = pd.DataFrame()
    num = 1
    image_list = []
    for filename in os.listdir(directory):
        for image in os.listdir(directory + filename):
            if image.endswith(".jpg"):

                img_original = cv2.imread(directory + filename + "/" + image)
                img_resized = cv2.resize(img_original, (200, 200))
                image_list.append(img_resized)


    df_train = fe.get_features(image_list)
    df.to_csv("featuredataset/Gabor.csv")



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
            if (np.max(p) * 100) < 40:  #only those with propability greater than 40% are displayed
                continue

            title = food[int(np.where(p == np.max(p))[0])] + " " + str(round((np.max(p) * 100), 2))
            # Iterating over the grid returns the Axes.
            ax.set_title(title, fontdict=None, loc='center', color="k")
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    plt.show()



#get regions of interest
cropped_imgs: list = roi.get_roi(r"input_pictures/4.jpg")  #provide image here for testing, then run the main.py

#show regions of interests(ROI)
showimages(cropped_imgs)

#extract features from ROI
df = fe.get_features(cropped_imgs)

#convert to numpy array
test_features = np.array(df)

#preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
test_features = scaler.fit_transform(test_features)

#classification
#load trained models
import json
with open("models/model1_161221_fulldata.json",'r') as f:
    model_json = json.load(f)

kish = model_from_json(model_json)
kish.load_weights("models/model1_161221_fulldata.h5")

#classification labels
food = ['Meat', 'Noodles-Pasta', 'Rice', 'Soup']


#prediction
y_probs = kish.predict(test_features, verbose=0)

#show result
showimages(cropped_imgs,y_probs)