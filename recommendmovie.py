from __future__ import print_function, division
from argparse import ArgumentParser
from collections import namedtuple
from math import sqrt, log
from random import random
import csv


def init_parser():
    parser = ArgumentParser(description="Given a User ID from the MovieLens database, "
                                        "predict the user's scores for specified movies.")
    parser.add_argument('user_id', metavar='user-id', nargs='?',
                        help="The User ID from ratings.csv to predict ratings for.")
    parser.add_argument('ids', metavar='movies', nargs='*',
                        help="A list of movies (IMDb IDs) to predict the ratings of.")
    # This is ignored for the cross-validation routine, it would probably take a while
    parser.add_argument('-f', '--full', action='store_true',
                        help="Use the full dataset rather than the small dataset.")
    parser.add_argument('-m', '--movielens', action='store_true',
                        help="Specify that the IDs are MovieLens IDs rather than IMDb IDs.")
    parser.add_argument('-t', '--tmdb', action='store_true',
                        help="Specify that IDs are TMDb IDs rather than IMDb IDs.")
    parser.add_argument('-g', '--genres', action='store_true',
                        help="Use the content-based filtering method to calculate predicted rating using genres.")
    parser.add_argument('-r', '--rmse', metavar="percent", nargs='?', type=int, const=10,
                        help="Run the cross-validation test routine to calculate RMSE for each distance measure. "
                             "If a percent is specified, only that percent of users in the dataset will be used "
                             "for the routine, selected randomly (default: 10).")
    distance = parser.add_mutually_exclusive_group()
    # This is a placebo argument since it defaults to pearson anyway
    distance.add_argument('-p', '--pearson', action='store_true',
                          help="Use pearson correlation to calculate distances for collaborative filtering (default).")
    distance.add_argument('-c', '--cosine', action='store_true',
                          help="Use cosine similarity instead of Pearson correlation to "
                               "calculate distances for collaborative filtering.")
    distance.add_argument('-e', '--euclidean', action='store_true',
                          help="Use euclidean distance instead of Pearson correlation to "
                               "calculate distances for collaborative filtering.")
    # distance.add_argument('-b', '--manhattan', action='store_true',
    #                       help="Use manhattan distance instead of Pearson correlation to "
    #                            "calculate distances for collaborative filtering.")
    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()
    if args.rmse:
        if args.rmse > 100 or args.rmse < 0:
            parser.error("cross-validation percent needs to be between 0 and 100.")
        movies = get_movies_from_ids(None, True, args.full)
        print(" Cosine RMSE:        Pearson RMSE:       Euclidean RMSE:     Manhattan RMSE:")
        print(calculate_rmse_for_each_distance_measure(args.rmse, movies, args.genres))
    else:
        if args.movielens:
            movie_ids = args.ids
        elif args.tmdb:
            movie_ids = get_movie_ids_from_webdb_ids(args.ids, args.full, tmdb=True)
        else:
            movie_ids = get_movie_ids_from_webdb_ids(args.ids, args.full)
        if not args.user_id:
            parser.error("user-id required when not performing cross-validation routine.")
        elif not args.ids:
            parser.error("movies required when not performing cross-validation routine.")
        for movie, rating in get_predicted_ratings(args.user_id, movie_ids, args.genres, args.full,
                                                   args.cosine, args.euclidean, False):
            print(movie, "| Predicted rating:", round_stars(rating), "stars")


def calculate_rmse_for_each_distance_measure(percent, movies, genres):
    """Leave-out-1 cross validation to calculate RMSE for each of the separate distance measures.
    percent defines how much of the data set to use in the cross validation. e.g. percent=80 skips 20% of the uids."""
    # This is pretty gross, hacky, reused code
    # It also takes about 10 minutes to run on my machine on the small dataset
    # The full dataset probably takes days, I didn't try it....

    # Create a set of mid - rating mappings for each userid in the table
    uid_to_ratings = {}
    mid_to_uids = {}
    with open('ml-latest-small/ratings.csv') as file:
        next(file)
        for uid, mid, rating, _ in csv.reader(file):
            try:
                uid_to_ratings[uid][mid] = float(rating)
            except KeyError:
                uid_to_ratings[uid] = {mid: float(rating)}
            try:
                mid_to_uids[mid].add(uid)
            except KeyError:
                mid_to_uids[mid] = set([uid])
    # For some (or all) users in the table, split their ratings into k equal subsets
    # run get_rating on k-1 subsets of the movies that they've rated, witholding the k'th subset
    # compare each predicted rating to each actual rating of the movie and average the difference over all trials
    cosine_dif = 0
    pearson_dif = 0
    euclidean_dif = 0
    manhattan_dif = 0
    cosine_length = 0
    pearson_length = 0
    euclidean_length = 0
    manhattan_length = 0
    for uid, current_user_ratings in uid_to_ratings.items():
        if len(current_user_ratings) <= 1 or random() >= (percent / 100):
            continue
        print(uid)
        cosine_length += len(current_user_ratings)
        pearson_length += len(current_user_ratings)
        euclidean_length += len(current_user_ratings)
        manhattan_length += len(current_user_ratings)
        for test_mid in current_user_ratings:
            all_other_user_ratings = [uid_to_ratings[uid] for uid in mid_to_uids[test_mid]]

            real_rating = current_user_ratings[test_mid]
            test_ratings = {mid: rating for mid, rating in current_user_ratings.items() if mid != test_mid}
            rating_sum = sum(rating for _, rating in test_ratings.items())
            our_avg = rating_sum / len(test_ratings)
            try:
                dif = real_rating
                dif -= get_rating(test_mid, genres, movies, test_ratings,
                                  all_other_user_ratings, our_avg)
                dif **= 2
                pearson_dif += dif
            except ZeroDivisionError:
                pearson_length -= 1
            try:
                dif = real_rating
                dif -= get_rating(test_mid, genres, movies, test_ratings,
                                  all_other_user_ratings, our_avg, cosine=True)
                dif **= 2
                cosine_dif += dif
            except ZeroDivisionError:
                cosine_length -= 1
            try:
                dif = real_rating
                dif -= get_rating(test_mid, genres, movies, test_ratings,
                                  all_other_user_ratings, our_avg, euclidean=True)
                dif **= 2
                euclidean_dif += dif
            except ZeroDivisionError:
                euclidean_length -= 1
            try:
                dif = real_rating
                dif -= get_rating(test_mid, genres, movies, test_ratings,
                                  all_other_user_ratings, our_avg, manhattan=True)
                dif **= 2
                manhattan_dif += dif
            except ZeroDivisionError:
                manhattan_length -= 1
    return (sqrt(cosine_dif / cosine_length), sqrt(pearson_dif / pearson_length),
            sqrt(euclidean_dif / euclidean_length), sqrt(manhattan_dif / manhattan_length))


def get_predicted_ratings(user_id, movie_ids, genres, full, cosine, euclidean, manhattan):
    """For a given User ID, predict ratings for each movie in the list of MovieLens IDs using collaborative filtering.
    "full" specifies that that full database should be used, and "cosine" specifies that 
    the cosine similarity distance function should be used instead of Pearson correlation"""
    our_user_ratings, all_other_user_ratings = get_relevant_user_ratings(user_id, movie_ids, full)
    movies = get_movies_from_ids(movie_ids, genres, full)
    rating_sum = sum(rating for _, rating in our_user_ratings.items())
    our_avg = rating_sum / len(our_user_ratings)
    for mid in movie_ids:
        yield movies[mid], get_rating(mid, genres, movies, our_user_ratings, all_other_user_ratings, our_avg, cosine)


def pearson_correlation(our_user_ratings, other_user_ratings, our_avg, other_avg):
    """Compute the Pearson correlation between two users, given a list of
    their ratings and their precomputed average rating"""
    numer = 0
    for mid, our_rating in our_user_ratings.items():
        if mid in other_user_ratings:
            numer += (our_rating - our_avg) * (other_user_ratings[mid] - other_avg)
    denom = sum((rating - our_avg) ** 2 for _, rating in our_user_ratings.items())
    denom *= sum((rating - other_avg) ** 2 for _, rating in other_user_ratings.items())
    # denom can sometimes be 0 if a user who rated the movie gave all movies the same rating
    # i.e.the difference between the rating for every movie of theirs and the average is 0
    # because the average == the score of every movie that they rated
    # this would also cause a problem if OUR user id gave all movies the same rating
    #
    # not sure what to do about this, just filter out users who rated all movies the same if attempting
    # to compute pearson correlation?
    #
    # Right now we just set the correlation to 0 to indicate the data isn't useful
    #
    # e.g. for movielens id 57, uid 149211, full dataset
    # recommendmovie.py 2 0113321 -f
    return 0 if denom == 0 else numer / sqrt(denom)


def cosine_similarity(our_user_ratings, other_user_ratings):
    """Computer the cosine similarity between two users, given a list of
    their ratings and their precomputed average rating"""
    numer = 0
    for mid, our_rating in our_user_ratings.items():
        if mid in other_user_ratings:
            numer += our_rating * other_user_ratings[mid]
    denom = sqrt(sum(rating ** 2 for _, rating in our_user_ratings.items()))
    denom *= sqrt(sum(rating ** 2 for _, rating in other_user_ratings.items()))
    return numer / denom


def euclidean_distance(our_user_ratings, other_user_ratings):
    return sqrt(sum((our_rating - other_user_ratings[mid]) ** 2
                for mid, our_rating in our_user_ratings.items() if mid in other_user_ratings))


def square_euclidean_distance(our_user_ratings, other_user_ratings):
    return sum((our_rating - other_user_ratings[mid]) ** 2
               for mid, our_rating in our_user_ratings.items() if mid in other_user_ratings)


def manhattan_distance(our_user_ratings, other_user_ratings):
    return sum(abs(our_rating - other_user_ratings[mid])
               for mid, our_rating in our_user_ratings.items() if mid in other_user_ratings)


def bray_curtis_distance(our_user_ratings, other_user_ratings):
    numer = 0
    denom = 0
    for mid, our_rating in our_user_ratings.items():
        if mid in other_user_ratings:
            numer += abs(our_rating - other_user_ratings[mid])
            denom += abs(our_rating + other_user_ratings[mid])
    return numer / denom


def canberra_distance(our_user_ratings, other_user_ratings):
    return sum(abs(our_rating - other_user_ratings[mid]) / (abs(our_rating) + abs(other_user_ratings[mid]))
               for mid, our_rating in our_user_ratings.items() if mid in other_user_ratings)


def get_rating(mid, genres, movies, our_user_ratings, all_other_user_ratings, our_avg,
               cosine=False, euclidean=False, manhattan=False):
    """Given a Movie ID and a list of our user's ratings, return the predicted rating for each movie
    using collaborative filtering with a distance function of either cosine similarity or Pearson correlation"""
    numer = 0
    denom = 0
    if genres:
        corpus = {}
        # we need a mapping from genres to a number of documents they appear in
        for other_user_rating in all_other_user_ratings:
            for mid, _ in other_user_rating.items():
                for genre in movies[mid].genres:
                    try:
                        corpus[genre] += 1
                    except KeyError:
                        corpus[genre] = 1
        N = len(all_other_user_ratings)
    for other_user_ratings in all_other_user_ratings:
        if mid not in other_user_ratings:
            continue
        other_avg = sum(rating for _, rating in other_user_ratings.items()) / len(other_user_ratings)
        diff = (other_user_ratings[mid] - other_avg)
        if cosine:
            weight = cosine_similarity(our_user_ratings, other_user_ratings)
        elif manhattan:
            weight = manhattan_distance(our_user_ratings, other_user_ratings)
        elif euclidean:
            weight = euclidean_distance(our_user_ratings, other_user_ratings)
        else:
            weight = pearson_correlation(our_user_ratings, other_user_ratings, our_avg, other_avg)
        if genres:
            genre_weight = get_genre_rating(our_user_ratings, other_user_ratings, movies, corpus, N)
            final_weight = weight * (abs(genre_weight) / 2)
            # print(final_weight, weight, genre_weight)
            weight = final_weight
        numer += diff * weight
        denom += abs(weight)
    return our_avg + (numer / denom)


def get_genre_rating(our_user_ratings, other_user_ratings, movies, corpus, N,
                     augmented=False, boolean=False, logarithmic=False, smooth=False, probablistic=False):
    genre_frequencies = {}
    for mid, _ in other_user_ratings.items():
        for genre in movies[mid].genres:
            try:
                genre_frequencies[genre] += 1
            except KeyError:
                genre_frequencies[genre] = 1
    # genre_frequencies contains the frequencies of genres in the document
    # for each query term (genre in our_user_ratings), compute the tf-idf for the term
    total = 0
    explored = set()
    for mid, rating in our_user_ratings.items():
        for genre in movies[mid].genres:
            if genre not in explored:
                explored.add(genre)
                total += tf_idf(genre_frequencies, movies, genre, corpus, N,
                                augmented, boolean, logarithmic, smooth, probablistic)
    return total


def tf_idf(genre_frequencies, movies, genre, corpus, N, augmented, boolean, logarithmic, smooth, probablistic):
    res = term_frequency(genre_frequencies, genre, augmented, boolean, logarithmic)
    res *= inverse_document_frequency(genre_frequencies, movies, genre, corpus, N, smooth, probablistic)
    return res


def term_frequency(genre_frequencies, genre, augmented, boolean, logarithmic):
    if augmented:
        return 0.5 + (0.5 * (genre_frequencies.get(genre, 0) / max(freq for _, freq in genre_frequencies.items())))
    elif boolean:
        return 1 if genre in genre_frequencies else 0
    elif logarithmic:
        return log(1 + genre_frequencies.get(genre, 0))
    else:
        return genre_frequencies.get(genre, 0) / sum(freq for _, freq in genre_frequencies.items())


def inverse_document_frequency(genre_frequencies, movies, genre, corpus, N, smooth=False, probablistic=False):
    try:
        total = corpus[genre]
    except KeyError:
        return 0
    if smooth:
        return log(1 + N / total)
    elif probablistic:
        return log((N - total) / total)
    else:
        return log(N / total)


def get_movie_ids_from_webdb_ids(ids, full, tmdb=False):
    """Given a list of IMDb IDs, return a list of Movie IDs corresponding to the same movie in the database"""
    filename = 'ml-latest/links.csv' if full else 'ml-latest-small/links.csv'
    with open(filename) as file:
        reader = csv.reader(file)
        if tmdb:
            return [mid for mid, _, tmdb_id in reader if tmdb_id in ids]
        else:
            return [mid for mid, imdb_id, _ in reader if imdb_id in ids]


def get_movies_from_ids(movie_ids, genres, full):
    """Return a mapping from Movie IDs to a name, genres pair"""
    filename = 'ml-latest/movies.csv' if full else 'ml-latest-small/movies.csv'
    with open(filename) as file:
        reader = csv.reader(file)
        Movie = namedtuple('Movie', ['title', 'genres'])
        if genres:
            return {mid: Movie(title, genres.split('|')) for mid, title, genres in reader}
        else:
            return {mid: Movie(title, genres.split('|')) for mid, title, genres in reader if mid in movie_ids}        


def get_relevant_user_ratings(user_id, movie_ids, full):
    """Given a User ID and a list of movies, return the User ID's ratings
    as well as all movies rated by any user who rated any of the movie_ids"""
    filename = 'ml-latest/ratings.csv' if full else 'ml-latest-small/ratings.csv'
    with open(filename) as file:
        reader = csv.reader(file)
        # skip header
        next(reader)
        # List of movie ratings for relevant users (ones who have rated any given movie id)
        all_other_user_ratings = []
        # Running list of the current user id ratings
        user_ratings = {}
        current_uid = 0
        insert = False
        for uid, mid, rating, _ in reader:
            if current_uid != uid:
                if user_id == current_uid:
                    our_user_ratings = user_ratings
                elif insert:
                    all_other_user_ratings.append(user_ratings)
                current_uid = uid
                user_ratings = {}
                insert = False
            if mid in movie_ids:
                insert = True
            # TODO Just save this as a float and convert later? Or, only convert once we know we're storing it?
            # Either way, there's some runtime optimization to be had here, perhaps a couple seconds for full dataset?
            user_ratings[mid] = float(rating)
        # Required for last user in CSV
        if user_id == current_uid:
            our_user_ratings = user_ratings
        elif insert:
            all_other_user_ratings.append(user_ratings)
    return our_user_ratings, all_other_user_ratings


def round_stars(score):
    return round(score * 2) / 2


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
