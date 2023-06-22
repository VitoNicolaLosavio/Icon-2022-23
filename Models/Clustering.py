import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class Clustering:

    def __init__(self, X, num_feature):
        kmeans_kwargs = {
            "init": "random",
            "n_init": num_feature,
            "max_iter": 300,
            "random_state": 42,
        }
        sse = []
        for k in range(1, num_feature):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(X)
            sse.append(kmeans.inertia_)

        plt.style.use("fivethirtyeight")
        plt.plot(range(1, num_feature), sse)
        plt.xticks(range(1, num_feature))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.show()
