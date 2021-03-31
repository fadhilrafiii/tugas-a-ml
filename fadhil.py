from Graph import Graph, Vertex, Edge
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

def accuration(y_true, y_predict):
    if (len(y_true) != len(y_predict)):
        print("actual list and prediction list don't have the same length!")
        return
    else:
        total = len(y_predict)
        count_true = 0
        for actual, predict in zip(y_true, y_predict):
            if (actual == predict):
                count_true += 1
        
        return count_true/total

def recall(tp, fn):
    if (tp == 0):
        return 0
    return tp/(tp + fn)

def precision(tp, fp):
    if (tp == 0):
        return 0
    return tp/(tp + fp)

def f1(precision, recall):
    if (precision == 0 && recall == 0):
        return 0
    return 2 * (precision * recall) / (precision + recall)

def confusion_matrix(y_true, y_predict, true_class):
    if (len(y_true) != len(y_predict)):
        print("actual list and prediction list don't have the same length!")
        return
    else:
        # [tp, fn, fp, tn]
        matrix = [0, 0, 0, 0]
        for actual, predict in zip(y_true, y_predict):
            print(actual, predict)
            if (actual == predict):
                if (predict == true_class):
                    matrix[0] += 1
                else:
                    matrix[3] += 1
            else:
                if (predict == true_class):
                    matrix[2] += 1
                else:
                    matrix[1] += 1
    
         return matrix
