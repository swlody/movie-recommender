from __future__ import print_function, division
from argparse import ArgumentParser
from collections import namedtuple
from math import sqrt
import csv


def init_parser():
    parser = ArgumentParser(description="Given a User ID from the MovieLens database, "
                                        "predict the user's scores for specified movies.")
    parser.add_argument('user_id', metavar='user-id',
                        help='The User ID from ratings.csv to predict ratings for.')
    parser.add_argument('ids', metavar='movies', nargs='+',
                        help='A list of movies (IMDb IDs) to predict the ratings of.')
    parser.add_argument('-f', '--full', action='store_true',
                        help="Use the full dataset rather than the small dataset.")
    parser.add_argument('-m', '--movielens', action='store_true',
                        help='Specify that the IDs are MovieLens IDs rather than IMDb IDs.')
    parser.add_argument('-t', '--tmdb', action='store_true',
                        help='Specify that IDs are TMDb IDs rather than IMDb IDs.')
    parser.add_argument('-c', '--cosine', action='store_true',
                        help="User cosine similarity instead of Pearson correlation to "
                        "calculate distances for collaborative filtering.")
    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()
    if args.movielens:
        movie_ids = args.ids
    elif args.tmdb:
        movie_ids = get_movie_ids_from_webdb_ids(args.ids, args.full, tmdb=True)
    else:
        movie_ids = get_movie_ids_from_webdb_ids(args.ids, args.full)
    for movie, rating in get_predicted_ratings(args.user_id, movie_ids, args.full, args.cosine):
        print(movie, "| Predicted rating:", round(rating) / 2, "stars")


def cross_validate(full):
    all_movies = get_movies_from_ids(None, full)
    all_user_ratings = get_relevant_user_ratings(0, all_movies, full)
    # For some (or all) users in the table, split their ratings into k equal subsets
    # run get_rating on k-1 subsets of the movies that they've rated, witholding the k'th subset
    # compare each predicted rating to each actual rating of the movie and average the difference over all trials


def get_predicted_ratings(user_id, movie_ids, full, cosine):
    """For a given User ID, predict ratings for each movie in the list of MovieLens IDs using collaborative filtering.
    "full" specifies that that full database should be used, and "cosine" specifies that 
    the cosine similarity distance function should be used instead of Pearson correlation"""
    our_user_ratings, all_other_user_ratings = get_relevant_user_ratings(user_id, movie_ids, full)
    movies = get_movies_from_ids(movie_ids, full)
    rating_sum = sum(rating for _, rating in our_user_ratings.items())
    our_avg = rating_sum / len(our_user_ratings)
    for mid in movie_ids:
        yield movies[mid], get_rating(mid, our_user_ratings, all_other_user_ratings, rating_sum, our_avg, cosine)


def pearson_correlation(our_user_ratings, other_user_ratings, our_avg, other_avg):
    """Computer the Pearson correlation between two users, given a list of
    their ratings and their precomputed average rating"""
    numer = 0
    for mid, our_rating in our_user_ratings.items():
        if mid in other_user_ratings:
            numer += (our_rating - our_avg) * (other_user_ratings[mid] - other_avg)
    denom = sum((rating - our_avg) ** 2 for mid, rating in our_user_ratings.items())
    denom *= sum((rating - other_avg) ** 2 for mid, rating in other_user_ratings.items())
    return numer / sqrt(denom)


def cosine_similarity(our_user_ratings, other_user_ratings, rating_sum):
    """Computer the cosine similarity between two users, given a list of
    their ratings and their precomputed average rating"""
    numer = 0
    for mid, our_rating in our_user_ratings.items():
        if mid in other_user_ratings:
            numer += our_rating * other_user_ratings[mid]
    denom = (rating_sum ** 2) * (sum(rating for _, rating in other_user_ratings.items()) ** 2)
    return numer / sqrt(denom)


def get_rating(mid, our_user_ratings, all_other_user_ratings, rating_sum, our_avg, cosine):
    """Given a Movie ID and a list of our user's ratings, return the predicted rating for each movie
    using collaborative filtering with a distance function of either cosine similarity or Pearson correlation"""
    numer = 0
    denom = 0
    for other_user_ratings in all_other_user_ratings:
        if mid not in other_user_ratings:
            continue
        other_avg = sum(rating for _, rating in other_user_ratings.items()) / len(other_user_ratings)
        diff = (other_user_ratings[mid] - other_avg)
        if cosine:
            weight = cosine_similarity(our_user_ratings, other_user_ratings, rating_sum)
        else:
            weight = pearson_correlation(our_user_ratings, other_user_ratings, our_avg, other_avg)
        numer += diff * weight
        denom += abs(weight)
    return our_avg + (numer / denom)


def get_movie_ids_from_webdb_ids(ids, full, tmdb=False):
    """Given a list of IMDb IDs, return a list of Movie IDs corresponding to the same movie in the database"""
    filename = 'ml-latest/links.csv' if full else 'ml-latest-small/links.csv'
    with open(filename) as file:
        reader = csv.reader(file)
        if tmdb:
            return [mid for mid, _, tmdb_id in reader if tmdb_id in ids]
        else:
            return [mid for mid, imdb_id, _ in reader if imdb_id in ids]


def get_movies_from_ids(movie_ids, full):
    """Return a mapping from Movie IDs to a name, genres pair"""
    filename = 'ml-latest/movies.csv' if full else 'ml-latest-small/movies.csv'
    with open(filename) as file:
        reader = csv.reader(file)
        Movie = namedtuple('Movie', ['name', 'genres'])
        if movie_ids is None:
            return {mid: Movie(name, genres.split('|')) for mid, name, genres in reader}
        else:
            return {mid: Movie(name, genres.split('|')) for mid, name, genres in reader if mid in movie_ids}


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
            if mid in movie_ids or user_id == '0':
                insert = True
            # Just save this as a float and convert later?
            user_ratings[mid] = int(float(rating) * 2)
        # Required for last user in CSV
        if user_id == current_uid:
            our_user_ratings = user_ratings
        elif insert:
            all_other_user_ratings.append(user_ratings)
    return all_other_user_ratings if user_id == '0' else (our_user_ratings, all_other_user_ratings)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
