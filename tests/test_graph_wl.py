from pytest import approx
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ..utils import load_data
from ..models import GraphSubtreeKernel

G = load_data("MUTAG")
# ------------------------------------------------------------
# def re_map(G):
#     tmp = []
#     for g in G:
#         _g = [set(), dict(), dict()]
#         bias = min(g[1].keys())
#         for e in g[0]:
#             _g[0].add((e[0]-bias, e[1]-bias))
#         for k, v in g[1].items():
#             _g[1][k-bias] = v
#         for k, v in g[2].items():
#             _g[2][(k[0]-bias, k[1]-bias)] = v
#         tmp.append(_g)
#     return tmp
# from grakel.datasets import fetch_dataset
# MUTAG = fetch_dataset("MUTAG", verbose=False)
# G, y = MUTAG.data, MUTAG.target
# G = re_map(G)
# # transform to the dhg format
# from dhg import Graph
# tmp = []
# v_lbl_set = set()
# for idx, g in enumerate(G):
#     num_v = len(g[1])
#     num_e = len(g[0])
#     tmp.append({
#         "num_v": num_v,
#         "num_e": num_e,
#         "v_lbl": list(g[1].values()),
#         "g_lbl": y[idx],
#         "e_list": sorted(list(g[0])),
#         "g": Graph(num_v, list(g[0])),
#     })
#     v_lbl_set.update(g[1].values())
# G = tmp
# v_lbl_map = {v: i for i, v in enumerate(v_lbl_set)}
# for g in G:
#     g["v_lbl"] = [v_lbl_map[v] for v in g["v_lbl"]]
# ------------------------------------------------------------
y = [-1 if x["g_lbl"][0] != 1 else 1 for x in G]


def test_graph_wl_acc():
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=2)

    m = GraphSubtreeKernel()
    K_train = m.fit_transform(G_train).cpu().numpy()
    K_test = m.transform(G_test).cpu().numpy()
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    # Computes and prints the classification accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", str(round(acc * 100, 2)) + "%")
    assert acc == approx(0.8947, 0.001)
