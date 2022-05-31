import time
import random
import heapq
import numpy as np


# 这个解决办法还是挺巧妙的。由于kd tree算法中返回的是point，而其中指定了point是list类型的。所以创建一个container来继承list类
# 并在其基础上添加索引标签，这样就能够通过返回的点获取其索引了，这也是属于数据代理？
class PointContainer(list):
    def __new__(self, value):
        return super(PointContainer, self).__new__(self, value)

    def set(self, label):
        self.label = label
        return self

    def set_other_data(self, d):
        self.other_data = d
        return self


class KDTree(object):
    """
    A super short KD-Tree for points...
    so concise that you can copypasta into your homework
    without arousing suspicion.

    This implementation only supports Euclidean distance.

    The points can be any array-like type, e.g:
        lists, tuples, numpy arrays.

    Usage:
    1. Make the KD-Tree:
        `kd_tree = KDTree(points, dim)`
    2. You can then use `get_knn` for k nearest neighbors or
       `get_nearest` for the nearest neighbor

    points are be a list of points: [[0, 1, 2], [12.3, 4.5, 2.3], ...]
    """

    def __init__(self, points, dim, dist_sq_func=None):
        """Makes the KD-Tree for fast lookup.

        Parameters
        ----------
        points : list<point>
            A list of points.
        dim : int
            The dimension of the points.
        dist_sq_func : function(point, point), optional
            A function that returns the squared Euclidean distance
            between the two points.
            If omitted, it uses the default implementation.
        """

        if dist_sq_func is None:
            dist_sq_func = lambda a, b: sum((x - b[i]) ** 2
                                            for i, x in enumerate(a))

        def make(points, i=0):
            if len(points) > 1:
                points.sort(key=lambda x: x[i])
                i = (i + 1) % dim
                m = len(points) >> 1
                return [make(points[:m], i), make(points[m + 1:], i),
                        points[m]]
            if len(points) == 1:
                return [None, None, points[0]]

        def add_point(node, point, i=0):
            if node is not None:
                dx = node[2][i] - point[i]
                for j, c in ((0, dx >= 0), (1, dx < 0)):
                    if c and node[j] is None:
                        node[j] = [None, None, point]
                    elif c:
                        add_point(node[j], point, (i + 1) % dim)

        def get_knn(node, point, k, return_dist_sq, heap, i=0, tiebreaker=1):
            if node is not None:
                dist_sq = dist_sq_func(point, node[2])
                dx = node[2][i] - point[i]
                if len(heap) < k:
                    heapq.heappush(heap, (-dist_sq, tiebreaker, node[2]))
                elif dist_sq < -heap[0][0]:
                    heapq.heappushpop(heap, (-dist_sq, tiebreaker, node[2]))
                i = (i + 1) % dim
                # Goes into the left branch, then the right branch if needed
                for b in (dx < 0, dx >= 0)[:1 + (dx * dx < -heap[0][0])]:
                    get_knn(node[b], point, k, return_dist_sq,
                            heap, i, (tiebreaker << 1) | b)
            if tiebreaker == 1:
                return [(-h[0], h[2]) if return_dist_sq else h[2]
                        for h in sorted(heap)][::-1]

        def walk(node):
            if node is not None:
                for j in 0, 1:
                    for x in walk(node[j]):
                        yield x
                yield node[2]

        self._add_point = add_point
        self._get_knn = get_knn
        self._root = make(points)
        self._walk = walk

    def __iter__(self):
        return self._walk(self._root)

    def _add_single_point(self, point):
        """Adds a point to the kd-tree.

        Parameters
        ----------
        point : array-like
            The point list.
        """
        if self._root is None:
            self._root = [None, None, point]
        else:
            self._add_point(self._root, point)

    def add_points(self, points):
        for point in points:
            self._add_single_point(point)

    def get_knn(self, point, k, return_dist_sq=True):
        """Returns k nearest neighbors.

        Parameters
        ----------
        point : array-like
            The point.
        k: int
            The number of nearest neighbors.
        return_dist_sq : boolean
            Whether to return the squared Euclidean distances.

        Returns
        -------
        list<array-like>
            The nearest neighbors.
            If `return_dist_sq` is true, the return will be:
                [(dist_sq, point), ...]
            else:
                [point, ...]
        """
        return self._get_knn(self._root, point, k, return_dist_sq, [])

    def get_nearest(self, point, return_dist_sq=True):
        """Returns the nearest neighbor.

        Parameters
        ----------
        point : array-like
            The point.
        return_dist_sq : boolean
            Whether to return the squared Euclidean distance.

        Returns
        -------
        array-like
            The nearest neighbor.
            If the tree is empty, returns `None`.
            If `return_dist_sq` is true, the return will be:
                (dist_sq, point)
            else:
                point
        """
        l = self._get_knn(self._root, point, 1, return_dist_sq, [])
        return l[0] if len(l) else None


def add_index_label(data, pre=0, others=None):
    labels = [i + pre for i in range(len(data))]
    if others is None:
        points = [PointContainer(p).set(label=l) for p, l in zip(data, labels)]
    else:
        points = [PointContainer(p).set(label=l).set_other_data(d) for p, l, d in zip(data, labels, others)]
    return points


def test_kd_tree(dim, points, additional_points, query_points):
    points = add_index_label(points)
    additional_points = add_index_label(additional_points, pre=len(points))
    query_points = add_index_label(query_points, pre=len(points)+len(additional_points))

    kd_tree_results = []
    kd_tree = KDTree(points, dim)
    print("Adding points!")
    sta = time.time()
    kd_tree.add_points(additional_points)

    print("Adding finish! cost time = ", time.time() - sta)
    # kd_tree_results.append(tuple(kd_tree.get_knn([0] * dim, neighbors)))
    print("Querying k-nearest!")
    sta = time.time()
    for t in query_points:
        sta_2 = time.time()
        res = tuple(kd_tree.get_knn(t, neighbors))
        print("single query time:", time.time() - sta_2)
        knn_indices = [item[1].label for item in res]
        knn_dists = [item[0] for item in res]
        kd_tree_results.extend(knn_indices)
        kd_tree_results.extend(knn_dists)
    print("Querying k-nearest finish! cost time = ", time.time() - sta)

    print("Querying nearest!")
    sta = time.time()
    for t in query_points:
        res = kd_tree.get_nearest(t)
        kd_tree_results.append(res[1].label)
        kd_tree_results.append(res[0])
    print("Querying nearest finish! cost time = ", time.time() - sta)
    return kd_tree_results


def test_naive(dim, points, additional_points, query_points):
    def dist_sq_func(a, b):
        return sum((x - b[i]) ** 2 for i, x in enumerate(a))

    def get_knn_naive(other_points, point, k, return_dist_sq=True):
        neighbors = []
        for i, pp in enumerate(other_points):
            dist_sq = dist_sq_func(point, pp)
            neighbors.append((dist_sq, pp))
        neighbors = sorted(neighbors)[:k]
        return neighbors if return_dist_sq else [n[1] for n in neighbors]

    def get_nearest_naive(other_points, point, return_dist_sq=True):
        nearest = min(other_points, key=lambda p: dist_sq_func(p, point))
        if return_dist_sq:
            return dist_sq_func(nearest, point), nearest
        return nearest

    points = add_index_label(points)
    additional_points = add_index_label(additional_points, pre=len(points))
    query_points = add_index_label(query_points, pre=len(points) + len(additional_points))

    naive_results = []
    all_points = points + additional_points
    # naive_results.append(tuple(get_knn_naive(all_points, [0] * dim, neighbors)))
    for t in query_points:
        res = get_knn_naive(all_points, t, neighbors)
        knn_indices = [item[1].label for item in res]
        knn_dists = [item[0] for item in res]
        naive_results.extend(knn_indices)
        naive_results.extend(knn_dists)
    for t in query_points:
        res = get_nearest_naive(all_points, t)
        naive_results.append(res[1].label)
        naive_results.append(res[0])
    return naive_results


def main():
    def rand_point():
        return [random.uniform(-1, 1) for d in range(dim)]

    dim = 3
    points = [rand_point() for x in range(10000)]
    additional_points = [rand_point() for x in range(100)]
    query_points = [rand_point() for x in range(4000)]

    sta = time.time()
    kd_tree_results = test_kd_tree(dim, points, additional_points, query_points)
    print("Bench Cost Time:", time.time() - sta)
    sta = time.time()
    naive_results = test_naive(dim, points, additional_points, query_points)
    print("Naive Cost Time:", time.time() - sta)

    acc = np.sum(np.ravel(kd_tree_results) == np.ravel(naive_results)) / len(np.ravel(kd_tree_results))
    print("Accuracy:", acc)


if __name__ == '__main__':
    neighbors = 100
    main()
