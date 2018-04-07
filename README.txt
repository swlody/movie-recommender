Currently, given a User ID from the GroupLens dataset and a list of movies - specified as either a list of IMDb IDs, TMDb IDs (-t/--tmdb), or MovieLens IDs (-m/--movielens) - return the predicted rating of each movie for the given user. Uses a basic collaborative filtering technique that uses a distance function of either cosine similarity (-c/--cosine) or Pearson correlation (default).

Uses GroupLens dataset from https://grouplens.org/datasets/movielens/ - either ml-latest (-f/--full) or ml-latest-small (default).

Using the full dataset right now takes a significant amount of time now - around 25-30 seconds on an SSD with an i5, because the entire dataset needs to be read in from file each time the program is run.

Usage: recommendmovie.py [-h] [-f] [-m] [-t] [-c] user-id movies [movies ...]

Using pypy for the full dataset saves a lot of time - ~33%

TODO:
a)
	"Cross-validation" to objectively compare results of different algorithms
	Find the "most-similar" movie given a single (or list of) movies
	Try some different methods out:
		Bayesian networks
		Machine learning / clustering? Markov decision process? Decision tree (NN would probably be better)?
		Recall@k / precision@k
	Higher weights to similar movies with similar genre tags - especially for movies where few ratings are available

b)
	Allow user to enter movie name instead of ID - fuzzy search
	Improve performance for full dataset:
		Correlation thresholding - only compare to neighbors above a certain threshold
		Best-n-neighbor - best n neighbors with highest correlations
		Keep database in memory and allow new searches