# coding utf-8

import numpy as np
import copy
from scipy.spatial import Delaunay
import itertools
import heapq

"""
variable:
points: x-y coordination of points, like [[1.0, 2.5], [-2.7, 9.6], ...]
"""

class Graph:
    def __init__(self):
        self.adjacentList = None
        self.lengthList = None

    def getter(self, adjacentList, lengthList):
        self.adjacentList = adjacentList
        self.lengthList = lengthList

class Proximity_Graph:
    def __init__(self, points):
        self.points = points
        self.DT = Graph()
        self.GG = Graph()
        self.RNG = Graph()
        self.MST = Graph()

    def all_activate(self):
        self.get_DT()
        self.get_GG()
        self.get_RNG()
        self.get_MST()

    def get_DT(self):
        if self.DT.adjacentList is None:
            DT = Delaunay(self.points)
            adjacentList = {}
            for p in DT.vertices:
                for i, j in itertools.combinations(p, 2):
                    try:
                        adjacentList[i].add(j)
                    except:
                        adjacentList[i] = {j}
                        # adjacentList[i].append(j)
                    try:
                        adjacentList[j].add(i)
                    except:
                        adjacentList[j] = {i}
                        # adjacentList[j].append(i)

            adjacentList = dict(sorted(adjacentList.items()))
            for k in adjacentList.keys():
                adjacentList[k] = list(adjacentList[k])

            lengthList = {}
            for i in range(len(adjacentList)):
                lens = []
                for j in adjacentList[i]:
                    length = np.linalg.norm(np.array(self.points[i])-np.array(self.points[j]))
                    lens.append(length)
                lengthList[i] = lens

            self.DT.getter(adjacentList, lengthList)

        return self.DT.adjacentList, self.DT.lengthList

    def get_GG(self):
        """
        Gabriel Graph:
        considering the circle which diameter is the edge between point p and q,
        it doesn't have other points than p and q.

        FORTRAN 計算幾何プログラミング 杉原厚吉著 岩波書店　p 305
        the edges of a Gabriel Graph are those of a Delaunay Triangulation
        which cross the corresponding Voronoi edges.

        only you have to check is whether r and s is in the circle or not
        (r and s is the neighbor of both p and q).
        """
        if self.GG.adjacentList is None:
            DT_adjacentList, DT_lengthList = self.get_DT()

            adjacentList = copy.deepcopy(DT_adjacentList)
            lengthList = copy.deepcopy(DT_lengthList)

            edges = set()
            for k, v in adjacentList.items():
                for u in v:
                    edge = tuple(sorted([k, u]))
                    edges.add(edge)
            edges = list(edges)

            for edge in edges:
                p, q = edge # 2 end points
                neighbors = set(self.DT.adjacentList[p]) & set(self.DT.adjacentList[q])
                neighbors = list(neighbors)
                invalidity = [np.dot((np.array(self.points[p])-np.array(self.points[r])),
                                     (np.array(self.points[q])-np.array(self.points[r]))) <= 0.0 for r in neighbors]
                if any(invalidity):
                    # delete edge pq
                    qi = adjacentList[p].index(q)
                    pi = adjacentList[q].index(p)
                    adjacentList[p].pop(qi)
                    lengthList[p].pop(qi)
                    adjacentList[q].pop(pi)
                    lengthList[q].pop(pi)

            self.GG.getter(adjacentList, lengthList)

        return self.GG.adjacentList, self.GG.lengthList

    def get_RNG(self):
        """
        Relative Neighbor Graphs
        a relative neighbor graph consists of the edges which fulfill below,
            d(p, q) <= min max{d(p, r), d(q, r)}.

        FORTRAN 計算幾何プログラミング 杉原厚吉著 岩波書店　p 308
        The Relative Neighborhood Graph of a Finite Planar Set, Toussaint G. T., Pattern Recognition vol. 12, pp. 261-268
        """
        if self.RNG.adjacentList is None:
            GG_adjacentList, GG_lengthList = self.get_GG()

            adjacentList = copy.deepcopy(GG_adjacentList)
            lengthList = copy.deepcopy(GG_lengthList)

            edges = set()
            for k, v in adjacentList.items():
                for u in v:
                    edge = tuple(sorted([k, u]))
                    edges.add(edge)
            edges = list(edges)

            nodes = list(adjacentList.keys())

            for edge in edges:
                p, q = edge
                length = lengthList[p][adjacentList[p].index(q)]
                invalidity = [max(np.linalg.norm(np.array(self.points[p]) - np.array(self.points[r])),
                                  np.linalg.norm(np.array(self.points[q]) - np.array(self.points[r]))) <= length
                              for r in nodes if r != p and r != q]
                #print(edge)
                #print(invalidity)
                if any(invalidity):
                    # delete edge pq
                    qi = adjacentList[p].index(q)
                    pi = adjacentList[q].index(p)
                    adjacentList[p].pop(qi)
                    lengthList[p].pop(qi)
                    adjacentList[q].pop(pi)
                    lengthList[q].pop(pi)

            self.RNG.getter(adjacentList, lengthList)

        return self.RNG.adjacentList, self.RNG.lengthList

    def get_MST(self):
        """
        Minimum Spanning Trees
        """
        if self.MST.adjacentList is None:
            RNG_adjacentList, RNG_lengthList = self.get_RNG()
            nodes = list(RNG_adjacentList.keys())

            MST = kruskal(RNG_adjacentList, RNG_lengthList, nodes)

            adjacentList = {}
            lengthList = {}
            for edge in MST:
                p, q = edge
                try:
                    adjacentList[p].append(q)
                    lengthList[p].append(RNG_lengthList[p][RNG_adjacentList[p].index(q)])
                except KeyError:
                    adjacentList[p] = [q]
                    lengthList[p] = [RNG_lengthList[p][RNG_adjacentList[p].index(q)]]
                try:
                    adjacentList[q].append(p)
                    lengthList[q].append(RNG_lengthList[q][RNG_adjacentList[q].index(p)])
                except KeyError:
                    adjacentList[q] = [p]
                    lengthList[q] = [RNG_lengthList[q][RNG_adjacentList[q].index(p)]]

            self.MST.getter(adjacentList, lengthList)

        return self.MST.adjacentList, self.MST.lengthList

### Kruskal algorithm ###
def connection(target, linked_list, s1, s2):
    """
    this updates the list of the sets of the points which is connected with each other
    :param target: (point1.id, point2.id)
    :param linked_list:
    :param s1: sub graph which has point1
    :param s2: sub graph which has point2
    :return:
    """
    if len(s1) == 0 and len(s2) == 0: # new sub graph
        new_set = {target[0], target[1]}
        linked_list.append(new_set)
    elif len(s1) > 0 and len(s2) == 0: # point1 is already in a sub graph
        linked_list.remove(s1)
        s1.add(target[1])
        linked_list.append(s1)
    elif len(s1) == 0 and len(s2) > 0: # point2 is already in a sub graph
        linked_list.remove(s2)
        s2.add(target[0])
        linked_list.append(s2)
    else: # point1 and point2 is in different sub graphs. unite them
        linked_list.remove(s1)
        linked_list.remove(s2)
        new_set = s1 | s2
        linked_list.append(new_set)
    return linked_list

def kruskal(adjacentList, lengthList, vertices):
    """
    this builds the minimum spanning tree of terminals by Kruskal method
    :param adjacentList: the adjacent list
    :param lengthList: the length list
    :param vertices: the list of vertices
    :return:
    """
    searched = set()  # seached points
    linked = []  # sets of linked points
    mst = []  # branches of minimum spanning tree
    mst_size = 0.0
    Q = []  # length and a pair of points, which are sorted

    t_n = len(vertices) # t_n is terminal number

    for node1 in vertices:
        for node2 in adjacentList[node1]:
            heapq.heappush(Q, (lengthList[node1][adjacentList[node1].index(node2)], node1, node2))

    # this is the first step of while
    u = heapq.heappop(Q)
    new_set = {u[1], u[2]}
    linked.append(new_set)
    mst.append(tuple(sorted([u[1], u[2]])))
    searched.add(u[1])
    searched.add(u[2])

    c = True
    while len(linked) != 1 or c:
        u = heapq.heappop(Q)
        set1 = set()
        set2 = set()
        continue_point = False

        for s in linked:
            if u[1] in s and u[2] in s: # both of u are already searched and linked
                continue_point = True
                break
            else: # each of u must not be in some subgraphs.
                if u[1] in s:
                    set1 = s
                elif u[2] in s:
                    set2 = s
        if continue_point:
            continue
        else:
            mst.append(tuple(sorted([u[1], u[2]])))

        t = (u[1], u[2])
        linked = connection(t, linked, set1, set2)
        searched.add(u[1])
        searched.add(u[2])

        if len(searched) == t_n:
            c = False

    return tuple(sorted(mst))