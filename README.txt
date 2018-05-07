Uses GroupLens dataset from https://grouplens.org/datasets/movielens/ - either ml-latest (-f/--full) or ml-latest-small (default).

Using the full dataset right now takes a significant amount of time now - around 25-30 seconds on an SSD with an i5, because the entire dataset needs to be read in from file each time the program is run.

usage: recommendmovie.py [-h] [-g] [-f] [-m] [-t] [-r [percent]]
                         [-p | -c | -e]
                         [user-id] [movies [movies ...]]

Given a User ID from the MovieLens database, predict the user's scores for
specified movies.

positional arguments:
  user-id               The User ID from ratings.csv to predict ratings for.
  movies                A list of movies (IMDb IDs) to predict the ratings of.

optional arguments:
  -h, --help            show this help message and exit
  -g, --genres          Use tf-idf for genres as a weighting for collaborative
                        filtering.
  -f, --full            Use the full dataset rather than the small dataset.
  -m, --movielens       Specify that the IDs are MovieLens IDs rather than
                        IMDb IDs.
  -t, --tmdb            Specify that IDs are TMDb IDs rather than IMDb IDs.
  -r [percent], --rmse [percent]
                        Run the cross-validation test routine to calculate
                        RMSE for each distance measure. If a percent is
                        specified, only that percent of users in the dataset
                        will be used for the routine, selected randomly
                        (default: 10).
  -p, --pearson         Use pearson correlation to calculate distances for
                        collaborative filtering (default).
  -c, --cosine          Use cosine similarity instead of Pearson correlation
                        to calculate distances for collaborative filtering.
  -e, --euclidean       Use euclidean distance instead of Pearson correlation
                        to calculate distances for collaborative filtering.

Using pypy for the full dataset saves a lot of time - ~33%


In order to get the predicted score of a movie, run with:

recommendmovie.py [user-id] [movies [movies ...]]

where movies ... is a list of IMDb movie IDs. The program also supports MovieLens and TMDb movie IDs with the associated argument flags.


This program uses collaborative filtering to determine a predicted rating for each movie. The default is a score-based method which uses the distance between users' rating vectors to compute a weighted sum of ratings for a particular movie. The genre-based method uses the frequency with which users watch different genres to determine how different they are, which is used in the weighted sum. Since the genre-based method doesn't take score into account, it is appreciably worse than the score-based method.


The cross-validation method is used for testing. In its current state, it compares three distance measures - cosine similarity, Pearson correlation, and Euclidean distance - for score-based collaborative filtering as well as a genre-based method that uses tf-idf (smooth, logarithmic) with Pearson correlation. I hard-coded a lot of stuff when I was running my analyses, so a shortened version of the method is included, rather than the 100+ line monstrosity I had to use to test all the different combinations of tf-idf methods.


There are several other distance measures and tf-idf methods that are coded in, but are inaccessible using the command-line arguments because they were mostly used for testing and perform worse than the default implementations.


TODO:
a)
	k-fold cross-validation to objectively compare results of different algorithms
	Find the "most-similar" movie, or k movies given a single (or list of) movies
	Try some different methods out:
		Content-based filtering
		Combinations of different techniques (essentially feature-weighted linear stacking?)
		Bayesian networks
		Machine learning / clustering / pattern recognition?
		Markov decision process? Decision tree?
		Recall@k / precision@k
	Higher weights to similar movies with similar genre tags - especially for movies where few ratings are available
		This is essentially collaborative filtering x content-based filtering?

b)
	Allow user to enter movie name instead of ID - fuzzy search
	Improve performance for full dataset:
		Correlation thresholding - only compare to neighbors above a certain threshold
		Best-n-neighbor - best n neighbors with highest correlations
		Keep database in memory and allow new searches
		Convert CSV to database format for fast searches?


