from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
import knn


# Use delta to predict accuracy
def percentage_correct_class(predicted, test):
    right = sum([1 for x in range(len(predicted)) if predicted[x] == test[x]])
    return right / len(predicted) * 100


# Make the model and call prediction
def knn_classifier(data_train, data_test, target_train, target_test, k):
    classifier = knn.kNNClassifier()
    model = classifier.fit(data_train, target_train)

    targets_predicted = model.predict_class(data_test, k)

    return percentage_correct_class(targets_predicted, target_test)


# The built in library call
def k_nearest_neighbors_class(data_train, data_test, target_train, target_test, k):
    classifier = KNeighborsClassifier(n_neighbors=k)
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test)

    return percentage_correct_class(targets_predicted, target_test)


# Use n-folding to get an overall accuracy
def n_folder_class(n, data, target, k):
    kf = model_selection.KFold(n_splits=n, shuffle=True)
    average_own = []
    average_builtin = []
    for train_index, test_index in kf.split(data):
        average_own.append(knn_classifier(data[train_index], data[test_index], target[train_index], target[test_index], k))
        average_builtin.append(k_nearest_neighbors_class(data[train_index], data[test_index], target[train_index], target[test_index], k))
    print("Average score for built in kNN classifier: {:0.1f}%".format(sum(average_builtin) / n))
    print("Average score for custom kNN classifier: {:0.1f}%".format(sum(average_own) / n))
