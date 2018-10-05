from sklearn.preprocessing import StandardScaler
import collections


class kNNModel:
    def __init__(self, data, targets):
        scale = StandardScaler()
        self.data = scale.fit_transform(X=data, y=None)
        self.targets = targets

    # Predict every item
    def predict_class(self, data_test, k):
        prediction = []
        scale = StandardScaler()
        for item in scale.fit_transform(X=data_test, y=None):
            prediction.append(self.predict_one_class(item, k))

        return prediction

    # Get the most common neighbor
    def predict_one_class(self, item, k):
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

    # Use regression. The only difference is what it calls
    def predict_regression(self, data_test, k):
        prediction = []
        scale = StandardScaler()
        for item in scale.fit_transform(X=data_test, y=None):
            prediction.append(self.predict_one_regression(item, k))

        return prediction

    # Predict one item.
    def predict_one_regression(self, item, k):
        distance = 0
        distances_from_item = []
        for i in range(len(self.data)):
            for j in range(len(item)):
                distance += (item[j] - self.data[i][j])**2
            distances_from_item.append((distance, self.targets[i]))
            distance = 0
        distances_from_item = sorted(distances_from_item)
        common = [item[1] for item in distances_from_item[:k]]
        # Get the average of the neighbors
        return sum(common) / k


class kNNClassifier:
    def fit(self, data, targets):
        return kNNModel(data, targets)
