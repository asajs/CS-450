import pandas as pd
import numpy as np
from sklearn import datasets
import sys


# Open files
def read_info(filename):
    try:
        with open(filename, 'r') as f:
            dataset = f.read()
    except IOError:
        print("Error reading file")
        sys.exit()
    return dataset


def get_adult_list():
    # The adult/census income dataset
    headers = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
               "relationship", "race", "sex", "capital-gain", "capital-loss", "hours", "country", "income"]
    filename = 'adult.csv'
    data = pd.read_csv(filename, names=headers, na_values="?", skipinitialspace=True)
    return parse_adult(data)

def get_ecoli_list():
    # The ecoli dataset
    data = read_info("ecoli.txt")
    return parse_ecoli(data)


def get_cars_list():
    # The cars dataset
    data = read_info("cars.txt")
    return parse_cars(data)


def get_autism_list():
    # The Autism dataset
    headers = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "age", "gender", "ethnicity", "jaundice",
               "autism", "country", "used", "result", "age_desc", "relation", "class"]
    filename = 'Autism-Adult-Data.csv'
    data = pd.read_csv(filename, names=headers, na_values="?")
    return parse_autism(data)


def get_mpg_list():
    # The mpg dataset
    data = read_info("mpg.txt")
    return parse_mpg(data)


def get_iris_list():
    # The Iris dataset
    iris = datasets.load_iris()
    return iris.data, iris.target


def parse_adult(p_data):
    data = p_data.drop(["education", "fnlwgt"], axis=1)
    data.dropna(how="any", inplace=True)
    replace = {"sex": {"Female": 0, "Male": 1},
               "income": {"<=50K": 0, ">50K": 1}}
    data.replace(replace, inplace=True)
    data = pd.get_dummies(data, columns=["workclass", "marital-status", "occupation",
                                         "relationship", "race", "country"])
    target = data["income"]
    data = data.iloc[:, :-1]
    data = data.astype(float)
    target = target.astype(float)
    return np.array(data), np.array(target)


def parse_autism(passed_in_data):
    # These shouldn't/don't have any effect on the outcome, or they are all the same
    data = passed_in_data.drop(['used', 'age_desc', 'relation', 'ethnicity', 'country'], axis=1)
    # instead of exploding the columns, just replace them with one or zero
    replace = {"gender": {"f": 0, "m": 1},
               "jaundice": {"no": 0, "yes": 1},
               "autism": {"no": 0, "yes": 1}}
    data.replace(replace, inplace=True)
    target = data['class']
    data = data.iloc[:, 0:-1]
    data.fillna('0', inplace=True)
    target.fillna('NO', inplace=True)
    data = data.astype(float, copy=True)
    return np.array(data), np.array(target)


def parse_cars(passed_in_data):
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    tmp_data = []
    inner_data = []
    # Manually parse the csv
    for item in passed_in_data.split('\n'):
        for data_point in item.split(','):
            inner_data.append(data_point)
        tmp_data.append(inner_data)
        inner_data = []
    data = pd.DataFrame(np.array(tmp_data), columns=headers)
    # Make some assumptions about the data
    replace = {"doors": {"5more": 5.0, "4": 4.0, "3": 3.0, "2": 2.0},
               "persons": {"more": 6.0, "4": 4.0, "2": 2.0},
               "buying": {"vhigh": 4.0, "high": 3.0, "med": 2.0, "low": 1.0},
               "maint": {"vhigh": 4.0, "high": 3.0, "med": 2.0, "low": 1.0},
               "lug_boot": {"big": 3.0, "med": 2.0, "small": 1.0},
               "safety": {"high": 3.0, "med": 2.0, "low": 1.0},
               "class": {"unacc": 0.0, "acc": 1.0, "good": 2.0, "vgood": 3.0}}
    data.replace(replace, inplace=True)
    target = data['class']
    data = data.iloc[:, 0:-1]
    return np.array(pd.get_dummies(data)), np.array(target)


def parse_ecoli(passed_in_data):
    tmp_data = []
    inner_data = []
    for item in passed_in_data.split('\n'):
        for data_point in item.split():
            # Attempt cast to float in a safe and fast way
            try:
                inner_data.append(float(data_point))
            except ValueError:
                inner_data.append(data_point)
        tmp_data.append(inner_data)
        inner_data = []
    data = []
    target = []
    for i in range(len(tmp_data)):
        data.append(tmp_data[i][1:7])
        target.append(tmp_data[i][-1:])
    data = np.array(data)
    target = np.array([item for sublist in target for item in sublist])
    return data, target


def parse_mpg(passed_in_data):
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
    data = []
    target = []
    for i in range(len(tmp_data)):
        data.append(tmp_data[i][1:7])
        target.append(tmp_data[i][0])
    data = pd.DataFrame(data)
    data.replace("?", np.nan, inplace=True)
    # Column 2 is the only one that has some missing values. Replace those with the average of column 2
    data.fillna(data[2].mean(), inplace=True)
    return np.array(data), np.array(target)
