# Using movie embeddings to recommend movies
"Movie embeddings" are similar to word embeddings but can be used to recommend movies to people who provide one movie, or a number of movies that they like. Movie vectors will be created using a dataset of movie likes of a large number of users. To define movie vectors we use an analog of the Distributional hypothesis for word meaning: you can “understand” a movie by looking at people who liked this particular movie, and asking what other movies they like.

We use a very simple embedding. Let Xi,j be the number of users that liked both movies i and j. Then we train the vectors v1, v2, . . . , for all the movies using an objective function.

The input to this optimization are the Xij counts and it has to find the movie vector vi for each movie i.
To optimize, we use gradient decent.

## Implementation
The Xij counts will come from the MovieLens dataset in which 943 users rated 1682 movies. We have preprocessed the dataset to simplify the task.

The file movieratings has 100,000 rows. The three entries in each row represent movie id, user id and movie rating, respectively. A rating of 1 indicates the user likes the movie, while a rating of 0 indicates the user does not like the movie. movies.csv has 1682 movies. It maps each movie id to its title.

MovieEmbeddings.py has the following functions:
- make_cooccurence : This function creates the co-occurrence matrix. Each entry Xi,j is the number of users who like both movie i and j. Because of the way the matrix is created, it should be symmetric.
- train : This function trains the movie vectors on the MovieLens dataset using gradient decent. We first initialize the movie vectors using standard normal distribution. In our code, we set learning rate η to 0.00001; the dimension of a movie vector k to 300, and run 200 iterations. K is the dimension of the movie vector, i.e. the matrix of the movie vectors is n by k. There are 1682 movie vectors and each movie vector has 300 elements. For debugging purposes, Train should print the value of the cost function at the end of each iteration.
- gradient : This function does the main work for train. It is the inner loop of the optimization which uses gradient descent.
- recommend1 : This function recommends top 20 movies to a user when given a movie that the user likes. Basically, I calculate the cosine similarity score between the given movie vector and all the other movie vectors, and then pick the top 20 movies that have the highest cosine similarity scores.
- recommend2 : Similar to recommend1, this function gets a list of movies from the user, not just a single movie and return top 20 recommendations. It first computes the average of the vectors for these movies, say v, and recommend other movies based upon cosine similarity with v.
