import scipy.io
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import random

# montage_images.m, converted, function from Piazza Post
def montage_images(images):
    num_images = min(1000, np.size(images, 2))
    numrows = math.floor(math.sqrt(num_images))
    numcols = math.ceil(num_images / numrows)
    img = np.zeros((numrows * 28, numcols * 28))
    for k in range(num_images):
        r = k % numrows
        c = k // numrows
        img[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = images[:, :, k]
    return img


sampleNums = [100, 200, 500, 1000, 2000, 5000, 10000]
errRates = []

mat = scipy.io.loadmat("data/digit-dataset/train.mat")
train_images = mat["train_images"]
train_labels = mat["train_labels"]
numTrainImages = np.size(train_images, 2)
confusionMatrices = dict()

for numSamples in sampleNums:
    confusionMatrix = list()
    for i in range(10):
        row = list()
        for j in range(10):
            row.append(0)
        confusionMatrix.append(row)

    flattened_images = list()
    flattened_labels = list()
    indexSet = set([i for i in range(numTrainImages)])  # set of indices to sample from
    # choose samples we will use to train the classifier
    for _ in range(numSamples):
        i = random.sample(indexSet, 1)[0]
        indexSet.remove(i)
        flattened_images.append([item for sublist in train_images[:, :, i] for item in sublist])
        flattened_labels.append(train_labels[i][0])

    # use library to fit the actual samples to relevant classes
    clf = svm.SVC(kernel='linear')
    clf.fit(flattened_images, flattened_labels)

    # validation phase to calculate error
    numValidationImages = 10000
    correctGuess = 0
    for __ in range(numValidationImages):
        i = random.sample(indexSet, 1)[0]
        indexSet.remove(i)
        digPredicted = clf.predict([[item for sublist in train_images[:, :, i] for item in sublist]])[0]
        digActual = train_labels[i][0]
        confusionMatrix[digActual][digPredicted] += 1
        if digPredicted == digActual:
            correctGuess += 1

    errRate = correctGuess / numValidationImages
    errRates.append(errRate)

    confusionMatrices[numSamples] = confusionMatrix

print("x Coords = " + str(sampleNums))
print("y Coords = " + str(errRates))
print("\n\nconfusionMatrix : ")

for matr in confusionMatrices:
    print(str(matr) + ":")
    print(str(np.trace(confusionMatrices[matr])) + "<--TRACE")
    for row in confusionMatrices[matr]:
        print('[%s]' % (' '.join('%04s' % i for i in row)))

plt.plot(sampleNums, errRates, '--bo')
plt.ylabel('Accuracy Rate')
plt.xlabel('Number of Samples')
plt.ylim((0, 1))
if len(sampleNums) != 0:
    plt.xlim((sampleNums[0] - sampleNums[0] / 20, sampleNums[-1] + sampleNums[-1] / 20))
    plt.show()
