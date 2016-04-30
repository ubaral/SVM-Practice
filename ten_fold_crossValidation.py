import scipy.io
import numpy as np
from sklearn import svm
import random

# load in the data files
mat = scipy.io.loadmat("data/digit-dataset/train.mat")
train_images = mat["train_images"]
train_labels = mat["train_labels"]
numTrainImages = np.size(train_images, 2)

# we flatten the labels and images of 10,000 random images from our set of 60,000.
# These 10,000 images will be partitioned and used to perform the 10-cross validation later.
indexSet = set([i for i in range(numTrainImages)])

k = 10
partitionSize = int(10000 / k)
# each k integer value 0,1,2,...,9 will map to the kth partitioning of the 10000 images.
# Each partition will be a list of a list of images and a list of corresponding labels,
# for training/validation purposes.
partitionDict = dict()
for i in range(k):
    partitionDict[i] = [[], []]

# Here we select 10000 random images from the full set of 60,000 images we have.
# In the loop we will mod by k and send that image to the appropriate "bucket" partition that it belongs to.
lstOfImages = []
for _ in range(10000):
    partitionKey = _ % k
    i = random.sample(indexSet, 1)[0]
    indexSet.remove(i)
    partitionDict[partitionKey][0].append([item for sublist in train_images[:, :, i] for item in sublist])
    partitionDict[partitionKey][1].append(int(train_labels[i][0]))

# run k iterations and validate for kth partition and train on the rest.

C_Values = [2e-6, 1e-6, (.5) * (1e-6), 1e-7]
for c_val in C_Values:
    accuracyTotalSum = 0
    for kk in range(k):
        clf = svm.SVC(kernel='linear', C=c_val)
        flattened_images = []
        flattened_labels = []
        for j in range(k):
            if j != kk:
                flattened_images += partitionDict[j][0]
                flattened_labels += partitionDict[j][1]
        # train the classifier with all images except those in partition number kk
        flattened_images = np.array(flattened_images)
        flattened_labels = np.array(flattened_labels)
        # TRAIN!
        clf.fit(flattened_images, flattened_labels)
        # calculate the running total of accuracy's which will be used to calculate the average
        correctGuess = 0
        predictedLabels = clf.predict(partitionDict[kk][0])
        actualLabels = partitionDict[kk][1]
        for i in range(partitionSize):
            if predictedLabels[i] == actualLabels[i]:
                correctGuess += 1
        accuracyTotalSum += correctGuess / partitionSize

    averageAccuracy = accuracyTotalSum / k
    print("average accuracy for C = " + str(c_val) + " was " + str(averageAccuracy))
