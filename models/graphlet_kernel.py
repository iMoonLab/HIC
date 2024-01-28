import math
from functools import reduce
from collections import defaultdict
from itertools import combinations, permutations

import torch
import numpy as np


class GraphLet:
    def __init__(self, num_v, e_list=None):
        self.num_v = num_v
        self.e_list = e_list
        self.adj_list = defaultdict(set)
        if e_list:
            for u, v in e_list:
                self.adj_list[u].add(v)
                self.adj_list[v].add(u)
            self.num_e = len(e_list)

    @staticmethod
    def from_adj_list(adj_list):
        _g = GraphLet(len(adj_list))
        _g.adj_list = adj_list
        _g.num_e = sum(len(v) for v in adj_list.values()) // 2
        return _g

    def sub(self, sub_v):
        v_map = {v: i for i, v in enumerate(sub_v)}
        sub_adj_list = {
            v_map[v]: set([v_map[u] for u in self.adj_list[v] & sub_v]) for v in sub_v
        }
        return GraphLet.from_adj_list(sub_adj_list)

    def permute_adj(self, perm):
        v_map = {v: i for i, v in enumerate(perm)}
        perm_adj_list = {
            v_map[v]: set([v_map[u] for u in self.adj_list[v]])
            for v in range(self.num_v)
        }
        return perm_adj_list

    def is_isomorphic(self, g):
        if self.num_v != g.num_v or self.num_e != g.num_e:
            return False
        for perm_v in permutations(range(self.num_v)):
            if self.permute_adj(perm_v) == g.adj_list:
                return True
        return False


class GraphletSampling:
    def __init__(self, k=5, normalize=True, sampling=None, symmetric=True):
        self.normalize = normalize
        if sampling is None:

            def draw_graphlet(g: GraphLet):
                for s in ConSubg(g.adj_list, k, symmetric):
                    yield g.sub(s)

        else:
            a_map = {1: 1, 2: 2, 3: 4, 4: 8, 5: 19, 6: 53, 7: 209, 8: 1253, 9: 13599}
            delta = sampling.get("delta", 0.05)
            epsilon = sampling.get("epsilon", 0.05)
            a = a_map[k]
            self.n_samples = math.ceil(
                2 * (a * np.log10(2) + np.log10(1 / delta)) / (epsilon**2) / 200
            )

            def draw_graphlet(g: GraphLet):
                cur_v = min(g.num_v, k)
                s = list(range(g.num_v))
                for _ in range(self.n_samples):
                    _idx = set(np.random.choice(s, cur_v, replace=False))
                    yield g.sub(_idx)

        self.draw_graphlet = draw_graphlet
        self._graph_bins = list()

    def count2mat(self, count):
        row_idx, col_idx, data = [], [], []
        for idx, g in enumerate(count):
            for lbl, cnt in g.items():
                row_idx.append(idx)
                col_idx.append(lbl)
                data.append(cnt)
        return (
            torch.sparse_coo_tensor(
                torch.tensor([row_idx, col_idx]),
                torch.tensor(data),
                size=(len(count), len(self._graph_bins)),
            )
            .coalesce()
            .float()
        )

    def fit_transform(self, g_list):
        self._count = [defaultdict(int) for _ in range(len(g_list))]
        for g in g_list:
            g["glet"] = GraphLet(g["num_v"], g["dhg"].e[0])
        for g_idx, g in enumerate(g_list):
            print(f"Processing graph {g_idx}/{len(g_list)}")
            for glet in self.draw_graphlet(g["glet"]):
                new_glet = True
                for glet_idx, glet_bin in enumerate(self._graph_bins):
                    if glet.is_isomorphic(glet_bin):
                        self._count[g_idx][glet_idx] += 1
                        new_glet = False
                        break
                if new_glet:
                    self._count[g_idx][len(self._graph_bins)] += 1
                    self._graph_bins.append(glet)
        self.train_cnt = self.count2mat(self._count)
        self.train_ft = self.train_cnt.mm(self.train_cnt.t()).to_dense()
        if self.normalize:
            self.train_ft_diag = torch.diag(self.train_ft)
            self.train_ft = (
                self.train_ft
                / torch.outer(self.train_ft_diag, self.train_ft_diag).sqrt()
            )
            self.train_ft[torch.isnan(self.train_ft)] = 0
            self.train_ft[torch.isinf(self.train_ft)] = 0
        return self.train_ft

    def transform(self, g_list):
        count = [defaultdict(int) for _ in range(len(g_list))]
        for g in g_list:
            g["glet"] = GraphLet(g["num_v"], g["dhg"].e[0])
        for g_idx, g in enumerate(g_list):
            for glet in self.draw_graphlet(g["glet"]):
                for glet_idx, glet_bin in enumerate(self._graph_bins):
                    if glet.is_isomorphic(glet_bin):
                        count[g_idx][glet_idx] += 1
                        break
        test_cnt = self.count2mat(count)
        test_ft = test_cnt.mm(self.train_cnt.t()).to_dense()
        if self.normalize:
            test_ft_diag = torch.diag(test_ft)
            test_ft = test_ft / torch.outer(test_ft_diag, self.train_ft_diag).sqrt()
            test_ft[torch.isnan(test_ft)] = 0
            test_ft[torch.isinf(test_ft)] = 0
        return test_ft


# ConSubg from:
# Karakashian, Shant Kirakos et al. “An Algorithm for Generating All Connected Subgraphs with k Vertices of a Graph.” (2013).
def ConSubg(G, k, symmetric):
    # G: dict of sets
    l = set()
    if symmetric:
        sG = G
        for u in G.keys():
            l |= CombinationsWithV(u, k, sG)
            sGP = dict()
            for v in sG.keys():
                if u != v:
                    sGP[v] = sG[v] - {u}
            sG = sGP
    else:
        for u in G.keys():
            l |= CombinationsWithV(u, k, G)

    return l


def CombinationsWithV(u, k, G_init):
    l = list()
    tree = defaultdict(set)
    treeL = {0: u}
    MarkN = dict()

    def CombinationTree(u, k, G):
        root = u
        l = [set() for i in range(k)]
        l[0].add(u)

        MarkV = dict()

        def BuildTree(nt, depth, k):
            # globals l, MarkN, MarkV, tree
            l[depth] = set(l[depth - 1])
            for v in G[treeL[nt]]:
                if v != nt and v not in l[depth]:
                    ntp = len(treeL)
                    treeL[ntp] = v
                    tree[nt].add(ntp)
                    l[depth].add(v)
                    if not MarkV.get(v, False):
                        MarkN[ntp], MarkV[v] = True, True
                    else:
                        MarkN[ntp] = False
                    if depth + 1 <= k - 1:
                        BuildTree(ntp, depth + 1, k)

        BuildTree(0, 1, k)

    def unionProduct(S1, S2):
        # globals tree, MarkN
        # print("To compare", S1, S2)
        if not len(S1):
            return set()
        elif not len(S2):
            return {S1}
        else:
            return {
                s1 | s2
                for s1 in S1
                for s2 in S2
                for s1p, s2p in [({treeL[i] for i in s1}, {treeL[i] for i in s2})]
                if not len(s1p & {treeL[i] for i in s2})
                and (
                    any(MarkN[j] for j in s2)
                    or all(not len({treeL[j] for j in tree[i]} & s2p) for i in s1)
                )
            }

    # Memoization
    CFM = dict()

    def CombinationsFromTree(root, k):
        # Globals tree
        t = root
        lnodesets = set()
        if k == 1:
            return {frozenset({t})}
        for i in range(1, min(len(tree[t]), k - 1) + 1):
            for NodeComb in combinations(tree[t], i):
                for string in compositions(k - 1, i):
                    fail = False
                    S = list()
                    for pos in range(i):
                        stRoot = NodeComb[pos]
                        size = string[pos]
                        m = CFM.get((stRoot, size), None)
                        if m is None:
                            m = CFM[stRoot, size] = CombinationsFromTree(stRoot, size)

                        S.append(m)
                        if not len(S[-1]):
                            fail = True
                            break
                    if fail:
                        continue
                    for combProduct in reduce(unionProduct, S):
                        lnodesets.add(frozenset(combProduct | {t}))
        return lnodesets

    CombinationTree(u, k, G_init)
    return {frozenset({treeL[f] for f in fs}) for fs in CombinationsFromTree(0, k)}


def compositions(n, k):
    if n < 0 or k < 0:
        return
    elif k == 0:
        if n == 0:
            yield []
        return
    elif k == 1:
        yield [n]
        return
    else:
        for i in range(1, n):
            for comp in compositions(n - i, k - 1):
                yield [i] + comp


if __name__ == "__main__":
    adj_list = {
        0: {1, 5},
        1: {0, 2},
        2: {1, 3, 13},
        3: {9, 2, 4},
        4: {3, 5},
        5: {0, 4, 6},
        6: {8, 5, 7},
        7: {6},
        8: {6},
        9: {11, 10, 3},
        10: {9},
        11: {9, 12, 13},
        12: {11},
        13: {2, 11},
    }
    g = GraphLet.from_adj_list(adj_list)
    for s in ConSubg(g.adj_list, 3, True):
        print(s)
        print(g.sub(s).adj_list)
    print("Testing is_isomorphic")
    g1 = GraphLet(4, [(0, 1), (0, 2), (0, 3)])
    g2 = GraphLet(4, [(3, 1), (3, 2), (3, 0)])
    g3 = GraphLet(4, [(0, 1), (0, 2), (0, 3), (1, 2)])
    print(g1.is_isomorphic(g2))
    print(g1.is_isomorphic(g3))
