from ..utils import load_data

# data from ours
g_list = load_data("MUTAG")
G1 = g_list

# data from grakel
from functools import cmp_to_key
from grakel.datasets import fetch_dataset


def re_map(G):
    tmp = []
    for g in G:
        _g = [set(), dict(), dict()]
        bias = min(g[1].keys())
        for e in g[0]:
            _g[0].add((e[0] - bias, e[1] - bias))
        for k, v in g[1].items():
            _g[1][k - bias] = v
        for k, v in g[2].items():
            _g[2][(k[0] - bias, k[1] - bias)] = v
        tmp.append(_g)
    return tmp


def cmp(a, b):
    if a["num_v"] < b["num_v"]:
        return -1
    elif a["num_v"] > b["num_v"]:
        return 1
    else:
        if a["num_e"] < b["num_e"]:
            return -1
        elif a["num_e"] > b["num_e"]:
            return 1
        else:
            if a["e_list"][-1][1] < b["e_list"][-1][1]:
                return -1
            elif a["e_list"][-1][1] > b["e_list"][-1][1]:
                return 1
            else:
                if a["e_list"][-2][1] < b["e_list"][-2][1]:
                    return -1
                elif a["e_list"][-2][1] > b["e_list"][-2][1]:
                    return 1
                else:
                    return 0


def sort_g(G):
    return sorted(G, key=cmp_to_key(cmp))


MUTAG = fetch_dataset("MUTAG", verbose=False)
G, y = MUTAG.data, MUTAG.target
G = re_map(G)
# transform to the dhg format
from dhg import Graph

tmp = []
v_lbl_set = set()
for g in G:
    num_v = len(g[1])
    num_e = len(g[0])
    tmp.append(
        {
            "num_v": num_v,
            "num_e": num_e,
            "v_lbl": list(g[1].values()),
            "e_list": sorted(list(g[0])),
            "g": Graph(num_v, list(g[0])),
        }
    )
    v_lbl_set.update(g[1].values())
G2 = tmp
v_lbl_map = {v: i for i, v in enumerate(v_lbl_set)}
for g in G2:
    g["v_lbl"] = [v_lbl_map[v] for v in g["v_lbl"]]

G1 = sort_g(G1)
G2 = sort_g(G2)


def test_transformed_data():
    for g1, g2 in zip(G1, G2):
        assert cmp(g1, g2) == 0
