import cv2
import os


class Nearestneighbour:
    labels = ['bananas','mangoes','strawberries','pineapples','watermelons']
    def __init__(self,dir):
        self.trainig_set = self.train(training_dir_loc=dir)
        pass

    def predict(self,img_url):
        img = cv2.imread(img_url,1)
        min_sum = img.sum()
        label_output = ""
        for label in self.trainig_set:
            for i in self.trainig_set[label]:
                diff = img - i
                scalar = diff.sum()
                if scalar < min_sum:
                    min_sum = scalar
                    label_output = label

        return label_output

    def train(self, training_dir_loc):
        #load the training data
        data_urls = [(i,os.path.join(training_dir_loc,i)) for i in self.labels]
        data = {}
        for url in data_urls:
            class_data = []
            files = os.listdir(url[1])
            for file in files[:10]:
                class_data.append(cv2.imread(os.path.join(url[1],file),1))
            data[url[0]] = class_data
        return data



