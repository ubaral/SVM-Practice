import scipy.io
import numpy as np
from sklearn import svm
import random

# load in the data files
mat = scipy.io.loadmat("data/spam-dataset/spam_data.mat")
training_data = mat["training_data"]
training_labels = mat["training_labels"]
testing_data = mat["test_data"]

y = training_labels[0]
X = training_data
k = 12
totalSamps = np.size(training_labels)
partitionSize = int(totalSamps / k)
partitionDict = dict()
for i in range(k):
    partitionDict[i] = [[], []]

for i in range(totalSamps):
    partitionKey = i % k
    partitionDict[partitionKey][0].append(training_data[i, :])
    partitionDict[partitionKey][1].append(training_labels[0][i])
# run k iterations and validate for kth partition and train on the rest.
C_Values = [10]
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
