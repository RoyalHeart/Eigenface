import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import cv2 as cv
import os


##Helper functions. Use when needed.
def show_orignal_images(pixels):
    # Displaying Orignal Images
    fig, axes = plt.subplots(6,
                             10,
                             figsize=(12, 7),
                             subplot_kw={
                                 "xticks": [],
                                 "yticks": []
                             })
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(64, 64), cmap="gray")
    plt.show()


def show_eigenfaces(pca):
    # Displaying Eigenfaces
    fig, axes = plt.subplots(2,
                             5,
                             figsize=(9, 4),
                             subplot_kw={
                                 "xticks": [],
                                 "yticks": []
                             })
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(64, 64), cmap="gray")
        ax.set_title("PC " + str(i + 1))
    plt.show()


def save_eigenfaces(pca):
    # Displaying Eigenfaces
    fig, axes = plt.subplots(2,
                             5,
                             figsize=(9, 4),
                             subplot_kw={
                                 "xticks": [],
                                 "yticks": []
                             })
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(64, 64), cmap="gray")
        # save figure for eigenface
        plt.savefig('eigenface_realtime.jpg')
        ax.set_title("PC " + str(i + 1))
    plt.show()


## Step 1: Read dataset and visualize it
df = pd.read_csv("6_ppl.csv")
targets = df["target"]
pixels = df.drop(["target"], axis=1)
print(np.array(pixels).shape)

x_train, x_test, y_train, y_test = train_test_split(pixels,
                                                    targets,
                                                    train_size=0.9)

# show_orignal_images(pixels)
# ## Step 3: Perform PCA.
# pca = PCA(n_components=10, whiten=True).fit(x_train)
# face_path = "ProgramPython/Eigenface/Eigenface_picture"
# show_eigenfaces(pca)

# ## Step 4: Project Training data to PCA
# print("Projecting the input data on the eigenfaces orthonormal basis")
# X_train_pca = pca.transform(x_train)
##############

# ## Step 5: Initialize Classifer and fit training data
# clf = MLPClassifier(hidden_layer_sizes=(1024, ),
#                     batch_size=256,
#                     verbose=True,
#                     early_stopping=True).fit(X_train_pca, y_train)

print(cv.__version__)
capture = cv.VideoCapture(0)
capture.set(16, 1920)
capture.set(9, 1080)

# create empty matrix to store face list
list_face_matrix = np.empty((0, 4096), float)

numFrame = 0
while True:
    success, frame = capture.read()
    # cv.imshow("live video from webcam", frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray frame", gray)
    haar_cascade = cv.CascadeClassifier(
        'ProgramPython/Eigenface/haar_face.xml')
    faces_rect = haar_cascade.detectMultiScale(gray,
                                               scaleFactor=1.1,
                                               minNeighbors=5)

    # if there are faces in the frame capture its and store to a matrix to compute eigenface
    if len(faces_rect) != 0:
        for (x, y, w, h) in faces_rect:
            # draw rectangle around the face
            cv.rectangle(frame, (x, y), (x + w, y + h), (128, 101, 231),
                         thickness=1,
                         lineType=0)
            # show how many frames are captured
            cv.putText(frame,
                       str(numFrame), (int(x + (w / 2)), y),
                       cv.FONT_HERSHEY_PLAIN,
                       1, (128, 101, 231),
                       thickness=1)
            crop_img = gray[y:y + h, x:x + w]
            crop_img_gray = cv.resize(crop_img, (64, 64),
                                      interpolation=cv.INTER_AREA)
            # store value of crop gray img to values
            values = (crop_img_gray.flatten('C') / 255)
            values = values.reshape(1, 4096)
            list_face_matrix = np.append(list_face_matrix, values, axis=0)
            print(f"{numFrame} append {values}")
            numFrame += 1
        # compute eigenface
        if numFrame == 54:
            list_face_matrix_pd = pd.DataFrame(list_face_matrix,
                                               columns=list(x_test.columns),
                                               index=[x for x in range(54)])
            pca_realtime = PCA(n_components=10).fit(list_face_matrix_pd)
            show_eigenfaces(pca_realtime)
            save_eigenfaces(pca_realtime)
            list_face_matrix = np.empty((0, 4096))
            numFrame = 0

    cv.imshow("Detected faces", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
