from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import random
matplotlib.rcParams.update({'font.size': 15})
#matplotlib.rcParams.update({'font.weight': 'heavy'})

def main():
    true_labels = [random.randint(1, 10) for i in range(100)]
    predicted_labels = [random.randint(1, 10) for i in range(100)]
    plot = getConfusionMatrixPlot(true_labels, predicted_labels)
    plot.show()

def getConfusionMatrixPlot(true_labels, predicted_labels, alphabet, cm =None):
    # Compute confusion matrix
    if cm==None:
        cm = confusion_matrix(true_labels, predicted_labels)

    # create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(cm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=100)

    # add color bar
    # plt.colorbar(res)

    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))

    # add genres as ticks
    #alphabet = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    plt.xticks(range(width), alphabet[:width], rotation=30)
    plt.yticks(range(height), alphabet[:height])
    #plt.tight_layout()
    return plt

def getFontColor(value):
    if value > 90:
        return "white"
    else:
        return "black"

if __name__ == "__main__":
    main()
