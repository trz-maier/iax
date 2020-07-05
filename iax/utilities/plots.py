import numpy as np
import matplotlib.pyplot as plt


def plot_prediction(image, model, labels: dict, title=''):

    if title:
        title += '\n'

    prediction = list(model.predict(np.array([image]))[0])
    index = prediction.index(max(prediction))

    plt.imshow(image)
    plt.title("%s%s:%s (%s)" % (title, index, labels[index], "{:.0%}".format(prediction[index])))
    plt.axis('off')

    return plt
