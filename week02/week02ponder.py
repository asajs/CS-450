from sklearn import datasets
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sys
import collections


class HardCodedModel:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def predict(self, data_test, k):
        prediction = []

        for item in data_test:
            prediction.append(self.predict_one_manual(item, k))

        return prediction

    def predict_one_manual(self, item, k):
        distance = 0
        distances_from_item = {}
        for i in range(len(self.data)):
            for j in range(len(item)):
                distance += (item[j] - self.data[i][j])**2
            distances_from_item[distance] = self.targets[i]
            distance = 0
        ordered_distances = collections.OrderedDict(sorted(distances_from_item.items()))
        neighbors = list(ordered_distances.values())[:k]
        result = collections.Counter(neighbors)
        return result.most_common(1)[0][0]


class HardCodedClassifier:
    @staticmethod
    def fit(self, data, targets):
        return HardCodedModel(data, targets)


def get_list(split):
    # The ecoli dataset.
    data = read_info()
    data, target = parse_ecoli(data)
    return model_selection.train_test_split(data, target, test_size=split)

    # The Iris dataset
    # iris = datasets.load_iris()
    # return model_selection.train_test_split(iris.data, iris.target, test_size=split)


def parse_ecoli(passed_in_data):
    tmp_data = []
    inner_data = []
    for item in passed_in_data.split('\n'):
        for data_point in item.split():
            try:
                inner_data.append(float(data_point))
            except ValueError:
                inner_data.append(data_point)
        tmp_data.append(inner_data)
        inner_data = []
    data_data = []
    target_data = []
    for i in range(len(tmp_data)):
        data_data.append(tmp_data[i][1:7])
        target_data.append(tmp_data[i][-1:])
    data_data = np.array(data_data)
    target_data = np.array([item for sublist in target_data for item in sublist])
    return data_data, target_data


def percentage_correct(predicted, test):
    right = sum([1 for x in range(len(predicted)) if predicted[x] == test[x]])
    return str(int(right / (len(predicted)) * 1000) / 10.0)


def hard_coded_classifier(data_train, data_test, target_train, target_test, k):
    classifier = HardCodedClassifier()
    iris_model = classifier.fit(data_train, target_train)

    targets_predicted = iris_model.predict(data_test, k)

    print("Percentage correct for hard coded classifier: " + percentage_correct(targets_predicted, target_test) + "%")


def k_nearest_neighbors(data_train, data_test, target_train, target_test, k):
    classifier = KNeighborsClassifier(n_neighbors=k)
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test)

    print("Percentage correct for library classifier: " + percentage_correct(targets_predicted, target_test) + "%")


def read_info():
    filename = "ecoli.txt"
    try:
        with open(filename, 'r') as f:
            dataset = f.read()
    except IOError:
        print("Error reading file")
        sys.exit()
    return dataset


if __name__ == "__main__":
    test_split = 0.3
    n = 3

    main_data_train, main_data_test, main_target_train, main_target_test = get_list(test_split)

    hard_coded_classifier(main_data_train, main_data_test, main_target_train, main_target_test, n)
    k_nearest_neighbors(main_data_train, main_data_test, main_target_train, main_target_test, n)
