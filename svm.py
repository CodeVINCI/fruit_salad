import numpy as np



class SVM:
    labels = ['bananas', 'mangoes', 'strawberries', 'pineapples', 'watermelons']
    def __init__(self, dir):
        self.dir = dir
        width = int(input("Enter the width of the image in data: "))
        height = int(input("Enter the height of the image in data: "))
        color_channel = int(input("Enter the number of color channels in data: "))
        self.W = np.random.rand(5,width*height*color_channel)

    def L_i_vectorized(self, x, y, W):
        scores = W.dot(x)
        margins = np.maximum(0, scores - scores[y]+1)
        margins[y] = 0
        loss_i = np.sum(margins)
        return loss_i