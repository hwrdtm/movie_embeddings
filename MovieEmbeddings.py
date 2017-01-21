import sys
import numpy as np
import pdb
import math
import random
import csv
import os

movies = []
movies_liked = []
movie_ratings = []
X = None
debug = False
test = False

def make_cooccurence(movies, movies_liked):
    # Generate co-occurence matrix skeleton
    X = np.zeros((len(movies), len(movies)))
    user_likes = []

    for idx, row in enumerate(movies_liked):
        current_user_id = row[1]
        if idx + 1 <= len(movies_liked) - 1:
            next_user_id = movies_liked[idx + 1][1]
        else:
            next_user_id = -1

        # Collect rows for single user_id
        user_likes.append(int(row[0]))

        # If next_user_id is different, perform stuff, then empty user_likes for next user
        if next_user_id != current_user_id:
            for ind1, movie1 in enumerate(user_likes):
                for ind2, movie2 in enumerate(user_likes[ind1+1:]):
                    # ind-1 because movies are 1-indexed, but arrays here are 0-indexed
                    X[movie1-1][movie2-1] += 1
                    X[movie2-1][movie1-1] += 1
            # Reset
            user_likes = []

    return X

# Calculate cost function
def calculate_cost(movie_vectors):
    cost = 0

    copy = np.array(movie_vectors)
    dot = np.dot(movie_vectors, np.transpose(copy))
    np.fill_diagonal(dot,0.0)
    cost = np.power((dot - X),2)

    # for i, v1 in enumerate(movie_vectors):
    #     copy = np.array(movie_vectors)
    #     copy[i] = np.zeros((1,300))
    #     cost += np.power(np.dot(copy, v1) - X[i],2)
    #
    #     # OLD
    #     # for j, v2 in enumerate(movie_vectors):
    #     #     # If i == j then gamma = 0
    #     #     if i == j:
    #     #         continue
    #     #     cost += math.pow((np.dot(v1,v2) - X[i][j]), 2)

    return np.sum(cost)

def gradient(movie_vectors, X):
    copy = np.array(movie_vectors)
    dot = np.dot(movie_vectors, np.transpose(copy))
    np.fill_diagonal(dot,0.0)
    gradient_matrix = 2 * np.dot((dot - X),copy)

    return gradient_matrix

# Train movie_vectors
def train(k, learn_rate, iterations, movies, X):
    movie_vectors = np.random.randn(len(movies), k)

    for num in range(iterations):
        # Calculate matrix of gradients
        gradient_matrix = gradient(movie_vectors, X)

        # Update movie_vectors
        movie_vectors -= learn_rate * gradient_matrix

        # print movie_vectors[1][-3:]
        #
        # for i, v1 in enumerate(movie_vectors):
        #     # copy = np.array(movie_vectors)
        #     # copy[i] = np.zeros((1,300))
        #     # row_gradient += 2 * (np.dot(copy, v1) - X[i])
        #
        #
        #     for j, v2 in enumerate(movie_vectors):
        #         # If i == j then gamma = 0
        #         if i == j:
        #             continue
        #         row_gradient += 2 * (np.dot(v1,v2) - X[i][j]) * v2
        #
        #     # Update row
        #     movie_vectors[i] -= learn_rate * row_gradient
        #
        #     row_gradient = 0

        print "Iteration {num} completed.".format(num=num)

        if debug or num < 10:
            # Calculate cost
            cost = calculate_cost(movie_vectors)
            print "Cost: {cost}".format(cost=cost)

    print "Training complete."
    return movie_vectors

def recommend1(movie_id, movie_vecs, movies):
    # a is movie we're given
    # B is the movie_vector matrix
    B = np.array(movie_vecs)
    movie_vec = movie_vecs[movie_id-1]
    dot = np.dot(movie_vec, np.transpose(B))

    a = np.linalg.norm(movie_vec)
    vec_norms = np.linalg.norm(B, axis=1)

    similarities = np.divide(dot, a*vec_norms)
    # Make each element into tuple containing index number before sorting
    similarities = [(ind, val) for ind, val in enumerate(similarities)]
    sorted_similarities = sorted(similarities, key=lambda tup: tup[1], reverse=True)

    # Extract top 20 movie_ids
    top20 = []
    for ind, tup in enumerate(sorted_similarities):
        if len(top20) == 20:
            break
        if tup[0] != (movie_id - 1):
            top20.append(movies[tup[0]])

    return top20

def recommend2(movie_ids, movie_vecs, movies):
    # a is movie we're given
    # B is the movie_vector matrix
    B = np.array(movie_vecs)
    given_movie_vecs = []

    for movie_id in movie_ids:
        given_movie_vecs.append(movie_vecs[movie_id - 1])

    # The average movie_vec
    summation = 0
    for given_movie_vec in given_movie_vecs:
        summation += given_movie_vec

    movie_vec = summation / len(given_movie_vecs)
    dot = np.dot(movie_vec, np.transpose(B))

    a = np.linalg.norm(movie_vec)
    vec_norms = np.linalg.norm(B, axis=1)

    similarities = np.divide(dot, a*vec_norms)
    # Make each element into tuple containing index number before sorting
    similarities = [(ind, val) for ind, val in enumerate(similarities)]
    sorted_similarities = sorted(similarities, key=lambda tup: tup[1], reverse=True)

    # Extract top 20 movie_ids
    top20 = []
    for ind, tup in enumerate(sorted_similarities):
        if len(top20) == 20:
            break
        if tup[0] not in np.subtract(movie_ids,1):
            top20.append(movies[tup[0]])

    return top20

######### MAIN CODE #########
print('Reading inputs...')

if test:
    # Test case
    movie_ratings = [
        [1,1,1],
        [2,1,1],
        [4,1,0],
        [1,2,0],
        [2,2,1],
        [3,2,0],
        [1,3,1],
        [2,3,1],
        [3,3,0],
        [4,3,1],
    ]

    for row in movie_ratings:
        if row[0] not in movies:
            movies.append(row[0])
        if row[2] == 1:
            movies_liked.append(row)

else:
    with open('movieratings.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            movie_id = row[0]
            user_id = row[1]
            movie_rating = row[2]

            # if movie_id not in movies:
            #     movies.append(movie_id)

            if int(movie_rating) == 1:
                movies_liked.append(row)

    movies = np.genfromtxt('movies.csv', delimiter='|',dtype=None)

# If co_occurrence matrix file does not exist
if not os.path.isfile('./cooccurrence.csv.gz'):
    X = make_cooccurence(movies, movies_liked)

    # Save co_occurrence matrix
    np.savetxt('cooccurrence.csv.gz', X, fmt='%i', delimiter=',')
else:
    X = np.loadtxt('cooccurrence.csv.gz', X, delimiter=',')

# Initialize training parameters
k = 300
learn_rate = 0.00001
iterations = 200

# Train the movie_vectors
movie_vectors = train(k, learn_rate, iterations, movies, X)

# Recommend1
lion_king = 71
recommend1 = recommend1(71, movie_vectors, movies)
print recommend1

# Recommend2
sleepless = 88
sex = 708
philadelphia = 478
recommend2 = recommend2([88,478,708], movie_vectors, movies)
print recommend2
