from pytest import approx
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ..utils import load_data
from ..models import GraphletSampling, GraphLet, ConSubg

def test_ConSubg():
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
    res = []
    for s in ConSubg(g.adj_list, 3, True):
        res.append(s)
    assert (frozenset({0, 1, 2}) in res)
    assert (frozenset({0, 5, 6}) in res)
    assert (frozenset({1, 2, 13}) in res)

def test_subgraph():
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
    sub = g.sub({0, 1, 2})
    assert sub.adj_list == {
        0: {1},
        1: {0, 2},
        2: {1},
    }

def test_isomorphism():
    g1 = GraphLet(4, [(0, 1), (0, 2), (0, 3)])
    g2 = GraphLet(4, [(3, 1), (3, 2), (3, 0)])
    g3 = GraphLet(4, [(0, 1), (0, 2), (0, 3), (1, 2)])
    assert g1.is_isomorphic(g2)
    assert not g1.is_isomorphic(g3)

def test_graphlet_sampling_acc():
    G = load_data("MUTAG")
    y = [-1 if x["g_lbl"][0] != 1 else 1 for x in G]
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=1)

    m = GraphletSampling()
    K_train = m.fit_transform(G_train).cpu().numpy()
    K_test = m.transform(G_test).cpu().numpy()
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    # Computes and prints the classification accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", str(round(acc * 100, 2)) + "%")
    assert acc == approx(0.8421, 0.001)
