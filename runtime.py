import time
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import dhg
import pandas as pd


from models import (
    HypergraphRootedKernel,
    GraphSubtreeKernel,
    GraphletSampling,
    HypergraphDirectedLineKernel,
    HypergraphSubtreeKernel,
    HypergraphHyedgeKernel,
)
from utils import load_data

multi_label, criterion = None, None

root = Path("data")
data_root = Path("data/hypergraph/Performance")


def gen_hypergraphs_for_n_hg(n_hg):
    n_v = 50
    n_e = n_v * 5
    hg_list = []
    for _ in range(n_hg):
        hg = dhg.random.hypergraph_Gnm(n_v, n_e, method="low_order_first")
        hg_list.append((-1, (hg.num_v, hg.e[0])))
    to_txt(hg_list, data_root / f"p_hg_{n_hg}.txt")


def gen_hypergraphs_for_n_v(n_v_list):
    n_hg = 500
    for n_v in n_v_list:
        hg_list = []
        for _ in range(n_hg):
            n_e = n_v * 5
            hg = dhg.random.hypergraph_Gnm(n_v, n_e, method="low_order_first")
            hg_list.append((-1, (hg.num_v, hg.e[0])))
        to_txt(hg_list, data_root / f"p_v_{n_v}.txt")


def gen_hypergraphs_for_n_e(n_e_list):
    n_hg = 500
    n_v = 50
    for n_e in n_e_list:
        filename = data_root / f"p_e_{n_e}.txt"
        if not filename.exists():
            hg_list = []
            for _ in range(n_hg):
                hg = dhg.random.hypergraph_Gnm(n_v, n_e, method="low_order_first")
                hg_list.append((-1, (hg.num_v, hg.e[0])))
            to_txt(hg_list, filename)


def gen_hypergraphs_for_de(de_list):
    n_hg = 500
    n_v = 50
    for de in de_list:
        filename = data_root / f"p_de_{de}.txt"
        prob_k_list = [1 if _ + 2 == de else 0 for _ in range(n_v - 1)]
        if not filename.exists():
            hg_list = []
            for _ in range(n_hg):
                n_e = n_v * de
                hg = dhg.random.hypergraph_Gnm(
                    n_v, n_e, method="custom", prob_k_list=prob_k_list
                )
                hg_list.append((-1, (hg.num_v, hg.e[0])))
            to_txt(hg_list, filename)


def to_txt(hg_list, filename):
    with open(filename, "w") as f:
        f.write(f"{len(hg_list)}\n")
        for hg in hg_list:
            lbl, (num_v, e_list) = hg
            if isinstance(lbl, list):
                lbl = [str(l) for l in lbl]
                lbl = " ".join(lbl)
            f.write(f"{num_v} {len(e_list)} {lbl}\n")
            v_lbl = ["0" for _ in range(num_v)]
            f.write(" ".join(v_lbl) + "\n")
            for e in e_list:
                e = [str(i) for i in e]
                f.write(" ".join(e) + "\n")


def infer(model_name, x_list):
    if model_name == "graph_subtree":
        model = GraphSubtreeKernel()
    elif model_name == "graphlet_sampling":
        model = GraphletSampling(sampling={})
    elif model_name == "hypergraph_rooted":
        model = HypergraphRootedKernel()
    elif model_name == "hypergraph_directed_line":
        model = HypergraphDirectedLineKernel()
    elif model_name == "hypergraph_subtree":
        model = HypergraphSubtreeKernel()
    elif model_name == "hypergraph_hyedge":
        model = HypergraphHyedgeKernel()
    else:
        raise NotImplementedError

    st = time.time()
    K_train = model.fit_transform(x_list).cpu().numpy()
    duration = time.time() - st
    print(f"Training time: {duration:.4f}s")

    print("--------------------------------------------------")
    return duration


def runtime_for_n_hg(model_names):
    data_name = "p_5000_hg"
    x_list, y_list, meta = load_data(data_name, root, True, "hypergraph")
    n_hg_list = [50, 100, 200, 500, 1000, 2000]
    res = defaultdict(list)
    for n_hg in n_hg_list:
        print(f"n_hg: {n_hg}")
        for model_name in model_names:
            print(f"\t{model_name}")
            t = infer(model_name, deepcopy(x_list[:n_hg]))
            res[model_name].append(t)
    df = pd.DataFrame(res, index=n_hg_list)
    df.to_csv("tmp_p_hg_5000.csv")


def runtime_for_n_v(model_names, n_v_list):
    res = defaultdict(list)
    for n_v in n_v_list:
        print(f"n_v: {n_v}")
        data_name = f"p_v_{n_v}"
        x_list, y_list, meta = load_data(data_name, root, True, "hypergraph")
        for model_name in model_names:
            print(f"\t{model_name}")
            t = infer(model_name, deepcopy(x_list))
            res[model_name].append(t)
    df = pd.DataFrame(res, index=n_v_list)
    df.to_csv("tmp_p_v_200.csv")


def runtime_for_n_e(model_names, n_e_list):
    res = defaultdict(list)
    for n_e in n_e_list:
        print(f"n_e: {n_e}")
        data_name = f"p_e_{n_e}"
        x_list, y_list, meta = load_data(data_name, root, True, "hypergraph")
        for model_name in model_names:
            print(f"\t{model_name}")
            t = infer(model_name, deepcopy(x_list))
            res[model_name].append(t)
    df = pd.DataFrame(res, index=n_e_list)
    df.to_csv("tmp_p_e_200.csv")


def runtime_for_de(model_names, de_list):
    res = defaultdict(list)
    for de in de_list:
        print(f"de: {de}")
        data_name = f"p_de_{de}"
        x_list, y_list, meta = load_data(data_name, root, True, "hypergraph")
        for model_name in model_names:
            print(f"\t{model_name}")
            t = infer(model_name, deepcopy(x_list))
            res[model_name].append(t)
    df = pd.DataFrame(res, index=de_list)
    df.to_csv("tmp_p_de_20.csv")


if __name__ == "__main__":
    # ----------------------------------
    # model_names = ['hypergraph_directed_line', 'hypergraph_rooted', 'hypergraph_subtree', 'hypergraph_hyedge']
    model_names = [
        "hypergraph_directed_line",
        "hypergraph_subtree",
        "hypergraph_hyedge",
    ]
    # ------------ for number of hypergraphs ----------------------
    # gen_hypergraphs_for_n_hg(5000)
    # runtime_for_n_hg(model_names)
    # ------------ for number of vertices --------------------------
    # n_v_list = [10, 20, 30, 40, 50, 100, 150, 200]
    # gen_hypergraphs_for_n_v(n_v_list)
    # runtime_for_n_v(model_names, n_v_list)
    # ------------ for number of hyperedge -------------------------
    # n_e_list = [5, 10, 15, 20, 50, 100, 150, 200]
    # gen_hypergraphs_for_n_e(n_e_list)
    # runtime_for_n_e(model_names, n_e_list)
    # ------------ for number of hyperedge degree ------------------
    de_list = [2, 3, 4, 5, 10, 15, 20]
    # de_list = [5]
    # gen_hypergraphs_for_deg_e(de_list)
    runtime_for_de(model_names, de_list)
