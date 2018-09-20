import random as rand
import numpy as np


def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    random = rand.Random()

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 100

        # Lets have the views range from 100-10000
        views = random.randint(100, 10000)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array


class Movie:
    """
    Class to hold movie title, year released, and runtime in minutes
    :return:
    """
    def __repr__(self):
        return str(self.title) + " (" + str(self.year) + ") - " + str(self.runtime) + " mins."

    def __init__(self, title="", year=0, runtime=0):
        if runtime < 0:
            runtime = 0
        self.runtime = runtime
        self.title = title
        self.year = year

    def get_run_time(self):
        return (self.runtime // 60), self.runtime % 60


def create_movie_list():
    movies = [Movie("Star Wars", 2017, 160), Movie("Princess Bride", 1910, 100), Movie("Incredibles", 2018, 140),
              Movie("Jason Bourne", 2015, 160)]
    return movies


if __name__ == "__main__":
    m = create_movie_list()
    d = [movie for movie in m if movie.runtime < 150]
    dict_movies = {}
    for movie in m:
        dict_movies[movie.title] = rand.uniform(1, 5)

    for d in dict_movies:
        print(d + " " + str(dict_movies[d]))

    movie_array = get_movie_data()

    print(movie_array.shape[0])
    print(movie_array.shape[1])

    print(movie_array[:2])
    print(movie_array[:, -2:])
    print(movie_array[:, 1])
