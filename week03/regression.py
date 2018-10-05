from sklearn.neighbors import KNeighborsRegressor
from sklearn import model_selection
import knn


# Use delta to predict accuracy
def percentage_correct_regression(predicted, test, delta):
    right = sum([1 for x in range(len(predicted)) if (abs(predicted[x] - test[x]) <= delta)])
    return right / len(predicted) * 100


# Make the model and call prediction
def knn_regression(data_train, data_test, target_train, target_test, k, delta):
    classifier = knn.kNNClassifier()
    model = classifier.fit(data_train, target_train)

    targets_predicted = model.predict_regression(data_test, k)

    return percentage_correct_regression(targets_predicted, target_test, delta)


# The built in library call
def k_nearest_neighbors_regression(data_train, data_test, target_train, target_test, k, delta):
    classifier = KNeighborsRegressor(n_neighbors=k)
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test)

    return percentage_correct_regression(targets_predicted, target_test, delta)


# Use n-folding to get an overall accuracy
def n_folder_regress(n, data, target, k, delta):
    kf = model_selection.KFold(n_splits=n, shuffle=True)
    average_own = []
    average_builtin = []
    for train_index, test_index in kf.split(data):
        average_own.append(knn_regression(data[train_index], data[test_index], target[train_index], target[test_index], k, delta))
        average_builtin.append(k_nearest_neighbors_regression(data[train_index], data[test_index], target[train_index], target[test_index], k, delta))
    print("Average score for library kNN classifier: {:0.1f}%".format(sum(average_builtin) / n))
    print("Average score for custom kNN classifier: {:0.1f}%".format(sum(average_own) / n))
