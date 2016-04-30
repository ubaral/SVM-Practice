import scipy.io
import numpy as np
from sklearn import svm
import random

mat = scipy.io.loadmat("data/spam-dataset/spam_data.mat")
training_data = mat["training_data"]
training_labels = mat["training_labels"][0]
testing_data = mat["test_data"]

# calculated C value
clf = svm.SVC(kernel='linear', C=10)
clf.fit(training_data, training_labels)

predicted = clf.predict(testing_data)

f = open("spamOutput.csv", 'w')
f.write("Id,Category\n")
for i in range(np.size(predicted)):
    f.write(str(i + 1) + "," + str(predicted[i]) + "\n")
print("DONE!")
