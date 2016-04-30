import scipy.io
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import random


def plot_confusion_matrix(cm, y, title='Confusion Matrix (10000 Training Samples)', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(y))
    plt.xticks(tick_marks, y, rotation=45)
    plt.yticks(tick_marks, y)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def custom_feature_extractor(imagetrue):
    featureslist = [item for sublist in imagetrue for item in sublist]
    h = np.size(imagetrue, 0)
    w = np.size(imagetrue, 1)
    featureslist += [np.average(imagetrue[h / 2:, w / 2:]), np.average(imagetrue[h / 2:, :w / 2]),
                     np.average(imagetrue[:h / 2, w / 2:]), np.average(imagetrue[:h / 2, :w / 2])]
    topclosed = False
    prevrow = (False, False, [], [], [])
    numholes = 0
    edgecount = 0
    for row in np.floor(imagetrue / 35.0):
        for i in range(np.size(row) - 1):
            if row[i] != 0 and row[i + 1] == 0:
                edgecount += 1
            elif row[i] == 0 and row[i + 1] != 0:
                edgecount += 1

        nonzero_elems = np.nonzero(row)[0]
        if len(nonzero_elems) > 0:
            separation = np.ediff1d(nonzero_elems) - 1
            nonzerodiffs = np.nonzero(separation)[0]
            startnonzero = nonzero_elems[0]
            endnonzero = nonzero_elems[-1]
            if len(nonzerodiffs) > 0:  # has gaps
                gaps_in_row = []
                for block in nonzerodiffs:
                    gap_start_index = nonzero_elems[block]
                    gap_end_index = nonzero_elems[block] + separation[block] + 1
                    gaps_in_row.append((gap_start_index, gap_end_index))

                # do stuff here, that has gaps of 0pixels, figure out how to detect holes
                if prevrow[0]:
                    if prevrow[1]:  # prevrow has Gaps, this row has gaps
                        for prevGap in prevrow[2]:
                            leakfromtop = shortcircuit = True
                            for ii in range(len(gaps_in_row)):
                                for p in range(gaps_in_row[ii][0], gaps_in_row[ii][1] + 1):
                                    shortcircuit = shortcircuit and prevrow[4][p] != 0
                                    if not shortcircuit:
                                        break
                                if shortcircuit:
                                    topclosed = True

                                if prevGap[0] >= startnonzero and prevGap[1] <= endnonzero:
                                    leakfromtop = False
                                    break
                            if leakfromtop:
                                topclosed = False
                                leakfromtop = False
                    else:  # prevrow no gaps, this row has gaps
                        for gap in gaps_in_row:
                            if not prevrow[3][0] <= gap[0] + 1 or not prevrow[3][1] >= gap[1] - 1:
                                topclosed = False

                else:  # zero pixels filled row
                    topclosed = False
                prevrow = (True, True, gaps_in_row, (startnonzero, endnonzero), row)
            else:  # only stuff here has no zero gaps in it
                if not prevrow[0]:  # prev row was all zeros and this row is all solid
                    topclosed = True
                else:  # prev row not all zeros
                    if prevrow[1]:  # prevrow has gaps, this row solid
                        for gap in prevrow[2]:
                            if startnonzero - 1 <= gap[0] and gap[1] <= endnonzero + 1:
                                if topclosed:
                                    numholes += 1
                    else:  # prevrow solid, this row solid
                        topclosed = True
                prevrow = (True, False, [], (startnonzero, endnonzero), row)
        else:  # row is all zeros
            prevrow = (False, False, [], (), row)

    binary = [0, 0, 0]
    if numholes > 2:
        numholes = 2
    binary[numholes] = 1

    return featureslist + binary + [edgecount]


mat = scipy.io.loadmat("data/digit-dataset/train.mat")
testMat = scipy.io.loadmat("data/digit-dataset/test.mat")
train_images = np.array(mat["train_images"])
train_labels = np.array(mat["train_labels"])
test_images = np.array(testMat["test_images"])
test_images = np.transpose(test_images)

numTrainImages = np.size(train_images, 2)
print("numTrainSamples = " + str(numTrainImages))
numSamples = numTrainImages
flattened_images = []
flattened_labels = []
indexSet = set([i for i in range(numTrainImages)])  # set of indices to sample from
# choose samples we will use to train the classifier
for _ in range(numSamples):
    i = random.sample(indexSet, 1)[0]
    indexSet.remove(i)
    flattened_images.append(custom_feature_extractor(train_images[:, :, i]))
    flattened_labels.append(train_labels[i][0])

# use library to fit the actual samples to relevant classes using our best calculated C value
clf = svm.SVC(kernel='linear', C=1e-6)
clf.fit(np.array(flattened_images), np.array(flattened_labels))

imagesToPredict = []
for i in range(np.size(test_images, 2)):
    imagesToPredict.append(custom_feature_extractor(test_images[:, :, i]))

imagesToPredict = np.array(imagesToPredict)
predicted = clf.predict(imagesToPredict)
f = open("output.csv", 'w')
f.write("Id,Category\n")
for i in range(np.size(predicted)):
    f.write(str(i+1) + "," + str(predicted[i]) + "\n")
print("DONE!")
# Validation will remove for actual training, and train on all 60000 images!
# numValidationImages = 10000
# confusionMatrix = np.zeros((10, 10))
#
# correctGuess = 0
# imagesToPredict = [[], []]
# for __ in range(numValidationImages):
#     i = random.sample(indexSet, 1)[0]
#     indexSet.remove(i)
#     imagesToPredict[1].append(train_labels[i][0])
#     imagesToPredict[0].append(custom_feature_extractor(test_images[:, :, i]))
#
# predicted = clf.predict(np.array(imagesToPredict[0]))
# i = 0
# for digPredicted in predicted:
#     digActual = imagesToPredict[1][i]
#     confusionMatrix[digActual][digPredicted] += 1
#     if digPredicted == digActual:
#         correctGuess += 1
#     i += 1
#
# accreate = correctGuess / numValidationImages
# print("accuracy rate is " + str(accreate))

# plot_confusion_matrix(confusionMatrix, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# plt.show()
