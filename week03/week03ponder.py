import parse
import classifier
import regression


if __name__ == "__main__":
    test_split = 0.3
    k = 9
    n = 2
    delta = 4.0

    print("\nEcoli dataset")
    ecoli_data, ecoli_target = parse.get_ecoli_list()
    classifier.n_folder_class(n, ecoli_data, ecoli_target, k)

    print("\nAutism dataset")
    autism_data, autism_target = parse.get_autism_list()
    classifier.n_folder_class(n, autism_data, autism_target, k)

    print("\nCar dataset")
    car_data, car_target = parse.get_cars_list()
    classifier.n_folder_class(n, car_data, car_target, k)

    print("\nIris dataset")
    iris_data, iris_target = parse.get_iris_list()
    classifier.n_folder_class(n, iris_data, iris_target, k)

    print("\nmpg dataset")
    mpg_data, mpg_target = parse.get_mpg_list()
    regression.n_folder_regress(n, mpg_data, mpg_target, k, delta)
