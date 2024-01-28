import dhg
import scipy
import numpy as np
from dhg import Graph, Hypergraph
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from skmultilearn.problem_transform import BinaryRelevance

g2hg_func = dhg.Hypergraph.from_graph
hg2g_func = dhg.Graph.from_hypergraph_clique


def load_data(name, root, degree_as_tag, model_type):
    # graph dataset
    if name in ["RG_macro", "RG_sub"]:
        data_type = "graph"
        folder = "RG"
        multi_label = False
    elif name in ["MUTAG", "NCI1", "PROTEINS", "IMDBMULTI", "IMDBBINARY"]:
        data_type = "graph"
        folder = name
        multi_label = False
    elif name in ["RHG_3", "RHG_10", "RHG_table", "RHG_pyramid"]:
        data_type = "hypergraph"
        folder = "RHG"
        multi_label = False
    elif name in ["stream_player"]:
        data_type = "hypergraph"
        folder = "STEAM"
        multi_label = False
    elif name in ["IMDB_dir_genre_m"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = True
    elif name in ["IMDB_dir_form", "IMDB_dir_genre"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = False
    elif name in ["IMDB_wri_genre_m"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = True
    elif name in ["IMDB_wri_form", "IMDB_wri_genre"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = False
    elif name in ["twitter_friend"]:
        data_type = "hypergraph"
        folder = "TWITTER"
        multi_label = False
    else:
        raise NotImplementedError
    if data_type == "graph" and model_type == "hypergraph":
        trans_func = g2hg_func
    elif data_type == "hypergraph" and model_type == "graph":
        trans_func = hg2g_func
    else:
        trans_func = lambda x: x

    # read data
    x_list = []
    with open(f"{root}/{data_type}/{folder}/{name}.txt", "r") as f:
        n_g = int(f.readline().strip())
        for _ in range(n_g):
            row = f.readline().strip().split()
            num_v, num_e = int(row[0]), int(row[1])
            g_lbl = [int(x) for x in row[2:]]
            v_lbl = f.readline().strip().split()
            v_lbl = [[int(x) for x in s.split("/")] for s in v_lbl]
            e_list = []
            for _ in range(num_e):
                row = f.readline().strip().split()
                e_list.append([int(x) for x in row])
            if data_type == "graph":
                d = Graph(num_v, e_list)
            else:
                d = Hypergraph(num_v, e_list)
            d = trans_func(d)
            x_list.append(
                {
                    "num_v": num_v,
                    "num_e": d.num_e,
                    "v_lbl": v_lbl,
                    "g_lbl": g_lbl,
                    "e_list": d.e[0],
                    "dhg": d,
                }
            )
    for x in x_list:
        if degree_as_tag:
            x["v_lbl"] = [int(v) for v in x["dhg"].deg_v]
        if isinstance(x["dhg"], Graph):
            x["e_lbl"] = [2] * x["num_e"]
        else:
            x["e_lbl"] = [int(e) for e in x["dhg"].deg_e]

    v_lbl_set, e_lbl_set, g_lbl_set = set(), set(), set()
    for x in x_list:
        if isinstance(x["v_lbl"][0], list):
            for v_lbl in x["v_lbl"]:
                v_lbl_set.update(v_lbl)
        else:
            v_lbl_set.update(x["v_lbl"])
        e_lbl_set.update(x["e_lbl"])
        g_lbl_set.update(x["g_lbl"])
    # re-map labels
    v_lbl_map = {x: i for i, x in enumerate(sorted(v_lbl_set))}
    e_lbl_map = {x: i for i, x in enumerate(sorted(e_lbl_set))}
    g_lbl_map = {x: i for i, x in enumerate(sorted(g_lbl_set))}
    ft_dim, n_classes = len(v_lbl_set), len(g_lbl_set)
    for x in x_list:
        x["g_lbl"] = [g_lbl_map[c] for c in x["g_lbl"]]
        if isinstance(x["v_lbl"][0], list):
            x["v_lbl"] = [tuple(sorted([v_lbl_map[c] for c in s])) for s in x["v_lbl"]]
        else:
            x["v_lbl"] = [v_lbl_map[c] for c in x["v_lbl"]]
        x["e_lbl"] = [e_lbl_map[c] for c in x["e_lbl"]]
        x["v_ft"] = np.zeros((x["num_v"], ft_dim))
        row_idx, col_idx = [], []
        for v_idx, v_lbls in enumerate(x["v_lbl"]):
            if isinstance(v_lbls, list) or isinstance(v_lbls, tuple):
                for v_lbl in v_lbls:
                    row_idx.append(v_idx)
                    col_idx.append(v_lbl)
            else:
                row_idx.append(v_idx)
                col_idx.append(v_lbls)
        x["v_ft"][row_idx, col_idx] = 1
    y_list = []
    if multi_label:
        for x in x_list:
            tmp = np.zeros(n_classes).astype(int)
            tmp[x["g_lbl"]] = 1
            y_list.append(tmp.tolist())
    else:
        y_list = [g["g_lbl"][0] for g in x_list]
    meta = {
        "multi_label": multi_label,
        "data_type": data_type,
        "ft_dim": ft_dim,
        "n_classes": len(g_lbl_set),
    }
    return x_list, y_list, meta


def separate_data(x_list, y_list, n_fold, seed):
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    n_fold_idx = []
    for train_idx, test_idx in kf.split(x_list, y_list):
        n_fold_idx.append((train_idx, test_idx))
    return n_fold_idx


def train_infer_SVM(train_X, train_Y, test_X, test_Y, multi_label):
    if not multi_label:
        clf = SVC(kernel="precomputed")
    else:
        clf = BinaryRelevance(
            classifier=SVC(kernel="precomputed"),
            require_dense=[True, True],
        )
    clf.fit(train_X, train_Y)
    outputs = clf.predict(test_X)
    test_val, best_res = performance(outputs, test_Y, multi_label)
    return test_val, best_res


# -------------------- Metrics ----------------------------


def performance(preds: np.ndarray, targets: np.ndarray, multi_label: bool):
    if multi_label:
        if isinstance(preds, scipy.sparse.csc_matrix):
            preds = preds.todense()
        else:
            preds = (preds > 0.5).astype(int)
        # multi-label classification metric:
        # https://medium.datadriveninvestor.com/a-survey-of-evaluation-metrics-for-multilabel-classification-bb16e8cd41cd
        # acc = (preds==lbls).mean()
        # Exact Match Ratio (EMR)
        EMR = (preds == targets).all(1).mean()
        # Example-based Accuracy
        EB_acc = (
            np.logical_and(preds, targets).sum(1) / np.logical_or(preds, targets).sum(1)
        ).mean()
        # Example-based Precision
        EB_pre = np.logical_and(preds, targets).sum(1) / preds.sum(1)
        EB_pre[np.isnan(EB_pre)] = 0
        EB_pre = EB_pre.mean()
        res = {"EMR": EMR, "EB_acc": EB_acc, "EB_pre": EB_pre}
        return EMR, res
    else:
        if len(preds.shape) == 2:
            preds = np.argmax(preds, axis=1)
        acc = accuracy_score(targets, preds)
        f1_micro = f1_score(targets, preds, average="micro")
        f1_macro = f1_score(targets, preds, average="macro")
        f1_weighted = f1_score(targets, preds, average="weighted")
        res = {
            "acc": acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }
        return acc, res


if __name__ == "__main__":
    g_list, y_list, meta = load_data("MUTAG", "data", True, "graph")
    print(g_list[0])
    g_list, y_list, meta = load_data("RHG_3", "data", True, "graph")
    print(g_list[0])
