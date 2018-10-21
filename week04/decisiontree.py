from sklearn import datasets
from sklearn import model_selection
from collections import defaultdict
from collections import Counter
import math
import numpy as np
import pandas as pd


class DecisionTreeClass():
    def fit(self, data, target, headers):
        tree = self.build_tree(data, target, headers)
        return DecisionTreeModel(tree)

    def entropy(self, data, target):
        tally = defaultdict(list)
        for i, item in enumerate(data):
            tally[item].append(i)

        entropies = []
        for item in tally:
            unique, counts = np.unique(target[tally[item]], return_counts=True)
            total = sum(counts)
            e = 0.0
            for i in counts:
                e -= i/total*math.log2(i/total)
            e = len(tally[item])/len(data)*e
            entropies.append(e)
        return sum(entropies)

    def build_tree(self, data, target, headers):
        labels = np.unique(data)
        pick_target = np.unique(target)
        if len(pick_target) == 1:
            return pick_target[0]
        elif len(headers) == 1:
            list_of_count = Counter(target)
            return list_of_count.most_common(1)[0][0]
        else:
            # Super high entropy so it gets overwritten
            min_entropy = 999
            index_of_data = 0
            print("data: " + str(data.shape[1]))
            for column in range(data.shape[1]):
                e = self.entropy(data[:, column], target)
                if e < min_entropy:
                    min_entropy = e
                    index_of_data = column
            print("header: " + str(len(headers)))
            print("index: " + str(index_of_data))
            class_name = headers[index_of_data]

            tree = {class_name: {}}
            # Column we picked to split on
            new_headers = headers
            del new_headers[index_of_data]
            search_column = data[:, index_of_data]
            for label in labels:
                indexes = []
                # Collect each occurrence of the item
                for i, item in enumerate(search_column):
                    if item == label:
                        indexes.append(i)
                new_data = []
                new_target = []
                for i in indexes:
                    new_data.append(data[i])
                    new_target.append(target[i])
                new_data = np.array(new_data)
                temp_one = np.array(new_data[:, :index_of_data])
                temp_two = np.array(new_data[:, index_of_data+1:])
                new_data = np.concatenate((temp_one, temp_two))
                new_target = np.array(new_target)

                sub = self.build_tree(new_data, new_target, new_headers)
                # print("second: " + str(class_name) + " " + str(label))
                tree[class_name][label] = sub

            return tree


class DecisionTreeModel():
    def __init__(self, tree):
        self.tree = tree

    def predict(self, data):
        # Predict tree
        return 0


if __name__ == "__main__":
    headers = ["infants", "water", "budget", "physician", "salvador", "religious", "satellite",
               "nicaraguan", "missile", "immigration", "corporation", "education", "superfund", "crime",
               "exports", "africa"]

    data = pd.read_csv("house-votes-84.csv", names=headers, na_values="?")
    data = data.fillna('Missing')
    data = np.array(data)
    tree = DecisionTreeClass()
    model = tree.fit(data[:, 1:], data[:, 0], headers)

