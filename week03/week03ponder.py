from sklearn import datasets
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import collections
import sys
import pandas as pd


class kNNModel:
    def __init__(self, data, targets):
        scale = StandardScaler()
        self.data = scale.fit_transform(X=data, y=None)
        self.targets = targets

    def predict(self, data_test, k):
        prediction = []
        scale = StandardScaler()
        for item in scale.fit_transform(X=data_test, y=None):
            prediction.append(self.predict_one_manual(item, k))

        return prediction

    def predict_one_manual(self, item, k):
        distance = 0
        distances_from_item = []
        for i in range(len(self.data)):
            for j in range(len(item)):
                distance += (item[j] - self.data[i][j])**2
            distances_from_item.append((distance, self.targets[i]))
            distance = 0
        distances_from_item = sorted(distances_from_item)
        common = [item[1] for item in distances_from_item[:k]]
        common = collections.Counter(common)
        return common.most_common(1)[0][0]


class kNNClassifier:
    def fit(self, data, targets):
        return kNNModel(data, targets)


def get_list():
    # The ecoli dataset
    # data = read_info("ecoli.txt")
    # return parse_ecoli(data)

    # The cars dataset
    data = read_info("cars.txt")
    return parse_cars(data)

    # The Iris dataset
    # iris = datasets.load_iris()
    # return iris.data, iris.target


def parse_cars(passed_in_data):
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    tmp_data = []
    inner_data = []
    for item in passed_in_data.split('\n'):
        for data_point in item.split(','):
            inner_data.append(data_point)
        tmp_data.append(inner_data)
        inner_data = []
    data = pd.DataFrame(np.array(tmp_data), columns=headers)
    replace_doors_persons = {"doors": {"5more": 5, "4": 4, "3": 3, "2": 2},
                     "persons": {"more": 6, "4": 4, "2": 2}}
    data.replace(replace_doors_persons, inplace=True)
    # TODO: explode columns
    print(data.dtypes)


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
    return right / len(predicted) * 100


def knn_classifier(data_train, data_test, target_train, target_test, k, n):
    classifier = kNNClassifier()
    iris_model = classifier.fit(data_train, target_train)

    targets_predicted = iris_model.predict(data_test, k)

    return percentage_correct(targets_predicted, target_test)


def k_nearest_neighbors(data_train, data_test, target_train, target_test, k, n):
    classifier = KNeighborsClassifier(n_neighbors=k)
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test)

    return percentage_correct(targets_predicted, target_test)


def n_folder(n, data, target, k):
    kf = model_selection.KFold(n_splits=n, shuffle=True)
    average_own = []
    average_builtin = []
    for train_index, test_index in kf.split(data):
        average_own.append(knn_classifier(data[train_index], data[test_index], target[train_index], target[test_index], k, n))
        average_builtin.append(k_nearest_neighbors(data[train_index], data[test_index], target[train_index], target[test_index], k, n))
    print("Average score for built in kNN classifier: {:0.1f}%".format(sum(average_builtin) / n))
    print("Average score for custom kNN classifier: {:0.1f}%".format(sum(average_own) / n))


def read_info(filename):
    try:
        with open(filename, 'r') as f:
            dataset = f.read()
    except IOError:
        print("Error reading file")
        sys.exit()
    return dataset


if __name__ == "__main__":
    test_split = 0.3
    k = 9
    n = 2

    data, target = get_list()

    n_folder(n, data, target, k)


