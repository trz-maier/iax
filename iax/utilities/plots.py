import numpy as np
import matplotlib.pyplot as plt


def plot_prediction(image, fun, labels: dict = None, title: str = ''):

    """
    Function plots an image with its prediction from the provided model
    :param image: input image
    :param fun: model predict function, i.e. model.predict
    :param labels: dictionary of labels
    :param title: additional title to be put on top of the prediction
    """

    if title:
        title += '\n'

    prediction = tuple(fun(np.array([image]))[0])
    index = prediction.index(max(prediction))

    if labels:
        label = labels[index]
    else:
        label = 'LABEL:%s' % index

    plt.imshow(image)
    plt.title("%s%s:%s (%s)" % (title, index, label, "{:.0%}".format(prediction[index])))
    plt.axis('off')

