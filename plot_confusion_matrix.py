import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['pgf.texsystem'] = 'pdflatex'

# sublime text macros for the win.
confusionMat_100 = np.array([[825, 0, 19, 0, 6, 18, 23, 4, 57, 4],
                             [0, 1011, 21, 33, 9, 1, 2, 15, 46, 10],
                             [28, 42, 633, 101, 48, 6, 42, 25, 14, 2],
                             [36, 40, 51, 744, 6, 33, 38, 18, 9, 81],
                             [7, 31, 14, 4, 643, 0, 17, 34, 3, 238],
                             [35, 41, 18, 127, 23, 419, 89, 9, 10, 135],
                             [28, 52, 37, 2, 42, 21, 820, 0, 11, 0],
                             [17, 42, 5, 35, 22, 0, 1, 732, 0, 202],
                             [5, 112, 50, 284, 17, 33, 31, 53, 289, 87],
                             [9, 22, 21, 33, 83, 1, 4, 69, 5, 725]])

confusionMat_200 = np.array([[923, 0, 5, 20, 2, 8, 31, 4, 4, 8],
                             [1, 1042, 3, 21, 0, 4, 1, 4, 12, 7],
                             [53, 74, 688, 56, 44, 4, 51, 14, 24, 13],
                             [19, 34, 31, 851, 2, 62, 6, 11, 20, 13],
                             [5, 31, 1, 1, 673, 2, 5, 9, 7, 198],
                             [35, 78, 7, 161, 39, 456, 15, 8, 72, 36],
                             [32, 33, 14, 9, 45, 11, 800, 0, 9, 2],
                             [30, 93, 16, 16, 43, 3, 1, 820, 3, 36],
                             [13, 91, 50, 165, 25, 34, 27, 19, 563, 30],
                             [10, 39, 1, 26, 105, 5, 0, 73, 17, 682]])

confusionMat_500 = np.array([[870, 1, 9, 4, 3, 27, 16, 12, 13, 1],
                             [0, 1068, 10, 8, 1, 4, 1, 0, 12, 6],
                             [12, 34, 740, 20, 19, 6, 26, 51, 11, 7],
                             [6, 32, 44, 778, 2, 66, 7, 20, 55, 38],
                             [6, 7, 4, 3, 827, 6, 24, 5, 4, 61],
                             [9, 42, 9, 80, 13, 651, 15, 7, 19, 22],
                             [17, 20, 18, 0, 18, 45, 882, 0, 9, 0],
                             [13, 49, 12, 4, 30, 7, 3, 904, 4, 68],
                             [7, 52, 41, 39, 15, 68, 15, 27, 676, 67],
                             [16, 20, 13, 12, 80, 5, 1, 71, 4, 814]])

confusionMat_1000 = np.array([[916, 1, 10, 5, 1, 21, 11, 7, 7, 10],
                              [0, 1143, 2, 1, 1, 4, 1, 5, 6, 5],
                              [23, 39, 785, 13, 15, 12, 28, 10, 25, 6],
                              [28, 17, 46, 834, 1, 56, 14, 21, 29, 23],
                              [7, 12, 8, 1, 892, 6, 28, 7, 2, 55],
                              [23, 17, 16, 85, 10, 670, 20, 6, 16, 21],
                              [26, 14, 37, 2, 14, 23, 829, 1, 2, 0],
                              [2, 25, 34, 5, 18, 10, 0, 856, 2, 85],
                              [15, 53, 20, 59, 16, 74, 17, 10, 643, 29],
                              [17, 10, 12, 14, 103, 14, 2, 59, 7, 757]])

confusionMat_2000 = np.array([[947, 0, 6, 5, 6, 19, 9, 0, 12, 1],
                              [0, 1075, 8, 0, 1, 5, 0, 5, 8, 1],
                              [19, 27, 818, 18, 30, 16, 34, 13, 25, 7],
                              [17, 23, 22, 880, 3, 54, 0, 12, 27, 8],
                              [7, 9, 7, 1, 861, 11, 8, 5, 2, 52],
                              [28, 13, 9, 74, 14, 696, 13, 4, 31, 16],
                              [13, 5, 16, 1, 29, 32, 834, 0, 10, 0],
                              [9, 15, 21, 12, 18, 7, 0, 890, 6, 60],
                              [7, 52, 21, 53, 14, 47, 14, 3, 719, 21],
                              [19, 13, 6, 24, 75, 6, 1, 72, 13, 820]])

confusionMat_5000 = np.array([[921, 0, 4, 2, 4, 12, 12, 1, 10, 2],
                              [2, 1102, 11, 8, 2, 4, 2, 1, 15, 0],
                              [17, 18, 830, 26, 17, 5, 29, 23, 28, 3],
                              [16, 18, 34, 841, 2, 49, 3, 14, 54, 14],
                              [7, 6, 11, 0, 851, 1, 11, 11, 3, 46],
                              [20, 20, 17, 58, 22, 719, 26, 4, 20, 11],
                              [13, 9, 24, 2, 19, 10, 885, 0, 1, 0],
                              [7, 13, 18, 12, 14, 1, 0, 898, 5, 56],
                              [9, 38, 35, 71, 12, 45, 12, 15, 718, 17],
                              [11, 17, 5, 18, 62, 7, 0, 53, 10, 838]])

confusionMat_10000 = np.array([[924, 0, 10, 4, 6, 24, 3, 1, 4, 1],
                               [2, 1086, 3, 1, 2, 3, 2, 1, 8, 2],
                               [17, 20, 874, 28, 18, 5, 20, 17, 22, 1],
                               [15, 13, 37, 849, 4, 40, 5, 24, 20, 8],
                               [4, 6, 19, 3, 885, 1, 11, 7, 2, 48],
                               [23, 13, 8, 87, 20, 684, 9, 6, 38, 2],
                               [17, 4, 24, 1, 12, 17, 872, 2, 1, 0],
                               [2, 9, 27, 9, 27, 2, 2, 879, 2, 83],
                               [15, 40, 36, 50, 8, 45, 14, 17, 746, 14],
                               [5, 4, 14, 9, 84, 8, 0, 58, 12, 829]])


# This function was found online from the scikit-learn examples of confusion matrix website here is a link to the page:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
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

# plot_confusion_matrix(confusionMat_100, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# plot_confusion_matrix(confusionMat_10000, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# plot_confusion_matrix(confusionMat_500, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# plot_confusion_matrix(confusionMat_1000, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# plot_confusion_matrix(confusionMat_2000, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# plot_confusion_matrix(confusionMat_5000, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# plot_confusion_matrix(confusionMat_10000, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# plt.savefig('LaTex Writeup/graphics/confusionMat_10000.pgf')
# plt.show()
