import pprint
import requests
import pickle
import pandas as pd
import spacy
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()


def show_blobs():
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    plt.scatter(X[:, 0], X[:, 1])


def load_web():
    secret = 'xxx'

    url = 'https://newsapi.org/v2/everything?'
    parameters = {
        'q': 'big data',  # query phrase
        'pageSize': 20,  # maximum is 100
        'apiKey': secret  # your own API key
    }
    # Make the request
    response = requests.get(url, params=parameters)

    # Convert the response to JSON format and pretty print it
    data = response.json()
    with open('output.pickle', 'wb') as w:
        pickle.dump(data, w)

def load_file():
    with open('output.pickle', 'rb') as r:
        articles = pickle.load(r)
    return articles

def make_df():
    titles = []
    dates = []
    descriptions = []
    for line in load_file()['articles']:
        titles.append(line['title'])
        dates.append(line['publishedAt'])
        descriptions.append(line['description'])
    # print({'titles':titles,'desc':descriptions, 'dates':dates})
    df = pd.DataFrame(data={'titles': titles, 'desc': descriptions, 'dates': dates})
    df = df.drop_duplicates(subset='titles').reset_index(drop=True)
    df = df.dropna()
    print(df.head())
    return df, titles
df, titles = make_df()
nlp = spacy.load('en_core_web_sm')
sent_vecs = {}
docs = []
for title in tqdm(titles):
    doc = nlp(title)
    docs.append(doc)
    sent_vecs.update({title: doc.vector})
sentences = list(sent_vecs.keys())
vectors = list(sent_vecs.values())
print(sentences)
X = np.array(vectors)
n_classes = {}
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.show()

for i in tqdm(np.arange(0.2, 0.3, 0.001)):
    dbscan = DBSCAN(eps=i, min_samples=2, metric='cosine').fit(X)
    n_classes.update({i: len(pd.Series(dbscan.labels_).value_counts())})
    print(dbscan.labels_)
# dbscan=DBSCAN(eps=0.26, min_samples=2, metric='cosine').fit(X)
# print(n_classes.values())
# show_blobs()
# plt.show()
