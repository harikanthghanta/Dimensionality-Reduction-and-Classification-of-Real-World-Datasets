from collections import Counter, defaultdict
from evaluators import *
#import load_data as ld
import numpy as np
import pandas as pd

class NaiveBayes():
    """
    Naive Bayes Classifier:
    It is trained with a 2D-array X (dimensions m,n) and a 1D array Y (dimension 1,n).
    X should have one column per feature (total m) and one row per training example (total n).
    After training a dictionary is filled with the class probabilities per feature.
    """
    def get_data(self):
        datafile = './adult/adult.data'
        file_test = './adult/adult.test'
        df = pd.read_csv(datafile, header=None)
        Y_train = df[14].values
        print df
        del df[14]
        del df[2]
        X_train = df.values

        df_test = pd.read_csv(file_test, header=None)
        Y_test = df_test[14].values
        del df_test[14]
        del df_test[2]
        X_test = df_test.values
        return X_train, Y_train, X_test, Y_test

    def train(self, X, Y):
        self.labels = np.unique(Y)
        no_rows, no_cols = np.shape(X)
        self.initialize_nb_dict()
        self.class_probabilities = self.calculate_relative_occurences(Y)
        #fill self.nb_dict with the feature values per class
        for label in self.labels:
            row_indices = np.where(Y == label)[0]
            X_ = X[row_indices, :]
            no_rows_, no_cols_ = np.shape(X_)
            for jj in range(0,no_cols_):
                self.nb_dict[label][jj] += list(X_[:,jj])
        #transform the dict which contains lists with all feature values 
        #to a dict with relative feature value occurences per class
        for label in self.labels:
            for jj in range(0,no_cols):
                self.nb_dict[label][jj] = self.calculate_relative_occurences(self.nb_dict[label][jj])

    def classify_single_elem(self, X_elem):
        Y_dict = {}
        for label in self.labels:
            class_probability = self.class_probabilities[label]
            for ii in range(0,len(X_elem)):
              relative_feature_values = self.nb_dict[label][ii]
              if X_elem[ii] in relative_feature_values.keys():
                class_probability *= relative_feature_values[X_elem[ii]]
              else:
                class_probability *= 0
            Y_dict[label] = class_probability
        return self.get_max_value_key(Y_dict)
                    
    def classify(self, X):
        self.predicted_Y_values = []
        no_rows, no_cols = np.shape(X)
        for ii in range(0,no_rows):
            X_elem = X[ii,:]
            prediction = self.classify_single_elem(X_elem)
            self.predicted_Y_values.append(prediction)
        return self.predicted_Y_values

    def calculate_relative_occurences(self, data):
        no_examples = len(data)
        ro_dict = dict(Counter(data))
        for key in ro_dict.keys():
            ro_dict[key] = ro_dict[key] / float(no_examples)
        return ro_dict

    def get_max_value_key(self, data):
        values = data.values()
        keys = data.keys()
        max_value_index = values.index(max(values))
        max_key = keys[max_value_index]
        return max_key
        
    def initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = defaultdict(list)



print("training naive bayes")
nbc = NaiveBayes()
X_train, Y_train, X_test, Y_test = nbc.get_data()
nbc.train(X_train, Y_train)
print("trained")
predicted_Y = nbc.classify(X_test)
y_labels = np.unique(Y_test)
print y_labels
for y_label in y_labels:
    f1 = f1_score(predicted_Y, Y_test, y_label)
    a = getAccuracy(predicted_Y, Y_test)
    print("Accuracy is: %s" % a)
    print("F1-score on the test-set for class %s is: %s" % (y_label, f1))