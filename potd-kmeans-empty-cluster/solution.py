import numpy as np


def kmeans_empty_cluster_update(
    points: np.ndarray,
    assignments: np.ndarray,
    old_centroids: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    points: array of shape (n, d).
    assignments: array of shape (n,), cluster id in [0, k) for each point.
    old_centroids: array of shape (k, d), the previous centroids.
    k: number of clusters.

    Returns an array of shape (k, d): for each cluster, the mean of its
    assigned points, or the corresponding old centroid unchanged if the
    cluster has no assigned points.
    """
    # TODO: group points by assignment, average each nonempty group,
    # and fall back to old_centroids for empty groups.
    points_group={}
    # print(assignments)
    for i in range(len(assignments)):
        if assignments[i] not in points_group.keys():
            points_group[assignments[i]]=[]
        points_group[assignments[i]].append(i)
    new_centroids=old_centroids.copy()
    for i in points_group.keys():
        data=points_group[i]
        cnt=len(data)
        if cnt==0:
            continue
        for j in range(points.shape[1]):
            new_centroids[i][j]=np.mean(points[data,j])
    return new_centroids
