import urllib.request
import zipfile
import os

import pandas as pd
import numpy as np
from scipy import spatial

datasetName = None

ratingsFrame = None
genreFrame = None
possibleGenres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
                  "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

dataFrame = None


def getDataset(datasetType):
    smallDatasetName = "ml-latest-small"
    bigDatasetName = "ml-latest"
    smallDatasetNewName = "data-small"
    bigDatasetNewName = "data-big"
    smallDatasetURL = "http://files.grouplens.org/datasets/movielens/" + \
        smallDatasetName + ".zip"
    bigDatasetURL = "http://files.grouplens.org/datasets/movielens/" + \
        bigDatasetName + ".zip"

    global datasetName

    if datasetType == "small":
        datasetName = smallDatasetNewName
        if not checkDataset(smallDatasetNewName):
            downloadDataset(smallDatasetURL, smallDatasetName,
                            smallDatasetNewName)
    else:
        datasetName = bigDatasetNewName
        if not checkDataset(bigDatasetNewName):
            downloadDataset(bigDatasetURL, bigDatasetName, bigDatasetNewName)


def checkDataset(datasetNewName):
    if(os.path.isdir(datasetNewName)):
        return True
    else:
        return False


def downloadDataset(datasetURL, datasetName, datasetNewName):
    urllib.request.urlretrieve(datasetURL, datasetNewName + ".zip")
    zipExtract = zipfile.ZipFile(datasetNewName + ".zip", 'r')
    zipExtract.extractall()
    zipExtract.close()
    os.rename(datasetName, datasetNewName)
    os.remove(datasetNewName + ".zip")


def readRatingData():
    global ratingsFrame

    ratingsFrameCols = ["user_id", "movie_id", "rating"]
    ratingsFrame = pd.read_csv(datasetName + "/ratings.csv",
                               sep=",", header=0, names=ratingsFrameCols, usecols=range(3))

    ratingsFrame = ratingsFrame.groupby(
        'movie_id').agg({'rating': [np.size, np.mean]})
    ratingsFrame.columns = ratingsFrame.columns.droplevel(0)

    # Normalize the 'size' column to be between 0 (no one rated) and 1 (everyone rated)
    ratingsFrame['size'] = (ratingsFrame-ratingsFrame.min()) / \
        (ratingsFrame.max() - ratingsFrame.min())


def readGenreData():
    global genreFrame

    genreFrameCols = ["movie_id", "title", "genres"]
    genreFrame = pd.read_csv(datasetName + "/movies.csv", sep=",", index_col="movie_id",
                             header=0, names=genreFrameCols, usecols=range(3))

    genreFrame["genres"] = genreFrame["genres"].map(convertGenres)


def convertGenres(g):
    movieGenres = g.split("|")
    movieGenresNew = []
    for genre in possibleGenres:
        if genre in movieGenres:
            movieGenresNew.append(1)
        else:
            movieGenresNew.append(0)
    return movieGenresNew


def mergeData():
    global dataFrame
    dataFrame = pd.DataFrame(index=genreFrame.index)

    dataFrame[["title", "genres"]] = genreFrame[["title", "genres"]].copy()
    dataFrame[["popularity", "rating"]] = ratingsFrame[["size", "mean"]].copy()


def calcDistance(movieAID, movieBID):
    movieA = dataFrame.loc[movieAID]
    movieB = dataFrame.loc[movieBID]
    
    genreDistance = spatial.distance.cosine(movieA["genres"], movieB["genres"])
    popularityDistance = abs(movieA["popularity"] - movieB["popularity"])

    return (genreDistance * 0.8) + (popularityDistance * 0.2)

def getKNeighbors(movieID, k):
    neighbors = []
    kNeighbors = []
    for otherMovieID, row in dataFrame.iterrows():
        if (movieID != otherMovieID):
            distance = calcDistance(movieID, otherMovieID)
            neighbors.append((otherMovieID, float(distance)))
    neighbors.sort(key=lambda tup: tup[1])
    for x in range(k):
        kNeighbors.append(neighbors[x])
    return kNeighbors


def main():
    getDataset("small")

    readRatingData()
    readGenreData()
    mergeData()
   
    results = getKNeighbors(1, 10)

if __name__ == '__main__':
    main()
