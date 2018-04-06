from __future__ import print_function
from argparse import ArgumentParser
from collections import namedtuple
import csv


def init_parser():
    # TODO Should allow user to specify TVDB / MovieLens movie ids instead of IMDb, if desired
    parser = ArgumentParser(description="Recommend a similar movie.")
    parser.add_argument('user_id', metavar='user-id',
                        help='The User ID from ratings.csv to predict the ratings of.')
    parser.add_argument('imdb_ids', metavar='movies', nargs='+',
                        help='A list of movies (IMDB IDs) to predict the ratings of.')
    parser.add_argument('-f', '--full', action='store_true',
                        help="Use the full dataset rather than the small dataset.")
    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()
    movie_ids = get_movie_ids_from_imdb_ids(args.imdb_ids, args.full)
    for movie, rating in get_predicted_ratings(args.user_id, movie_ids, args.full):
        print("Movie:", movie, "Predicted rating:", rating)


def get_predicted_ratings(user_id, movie_ids, full):
    movie_names = get_movie_names(movie_ids, full)
    our_users_ratings, other_user_ratings = get_relevant_user_ratings(user_id, movie_ids, full)
    for movie_id in movie_ids:
        yield movie_names[movie_id], get_rating(movie_id)


def get_movie_ids_from_imdb_ids(imdb_ids, full):
    """Given a list of IMDb IDs, return a list of Movie IDs corresponding to the same movie in the database"""
    filename = 'ml-latest/links.csv' if full else 'ml-latest-small/links.csv'
    with open(filename) as file:
        csvreader = csv.reader(file)
        return [movie_id for movie_id, imdb_id, tvdb_id in csvreader if imdb_id in imdb_ids]


def get_movie_names(movie_ids, full):
    """Return a mapping from Movie IDs to titles"""
    filename = 'ml-latest/movies.csv' if full else 'ml-latest-small/movies.csv'
    with open(filename) as file:
        csvreader = csv.reader(file)
        return {movie_id: title for movie_id, title, genres in csvreader if movie_id in movie_ids}


def get_relevant_user_ratings(user_id, movie_ids, full):
    """Given a User ID and a list of movies, return the User ID's ratings
    as well as all movies rated by any user who rated any of the movie_ids"""
    filename = 'ml-latest/ratings.csv' if full else 'ml-latest-small/ratings.csv'
    with open(filename) as file:
        csvreader = csv.reader(file)
        # Mapping from user ids to a list of movie ratings
        other_user_ratings = {}
        # Running list of the current user id ratings
        user_ratings = []
        Rating = namedtuple('Rating', ['movie_id', 'rating'])
        curr_uid = 0
        insert = False
        for uid, mid, rating, timestamp in csvreader:
            if curr_uid != uid:
                if user_id == curr_uid:
                    our_user_ratings = user_ratings
                elif insert:
                    other_user_ratings[curr_uid] = user_ratings
                curr_uid = uid
                user_ratings = []
                insert = False
            if mid in movie_ids:
                insert = True
            user_ratings.append(Rating(mid, rating))
    other_user_ratings = []
    return our_user_ratings, other_user_ratings


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
