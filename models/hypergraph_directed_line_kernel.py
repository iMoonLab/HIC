from itertools import combinations
from collections import defaultdict

import torch
import numpy as np
from dhg import Hypergraph, Graph, DiGraph


# Hypergraph Kernel from "A Hypergraph Kernel from Isomorphism Tests"
class HypergraphDirectedLineKernel:
    def __init__(self, n_iter=4, degree_as_label=True, normalize=True):
        self.n_iter = n_iter
        self.normalize = normalize
        self.degree_as_label = degree_as_label
        self._in_subtree_map, self._out_subtree_map = {}, {}

    def directed_line_expansion(self, hg_list, e_lbl):
        dl_list = []
        for hg_idx, hg in enumerate(hg_list):
            v_list, v_lbl, e_list = [], [], []
            for e_idx, e in enumerate(hg["e_list"]):
                for a, b in combinations(e, 2):
                    v_list.append((a, b))
                    v_lbl.append(e_lbl[hg_idx][e_idx])
            num_v = len(v_list)
            for src_idx in range(num_v):
                for dst_idx in range(num_v):
                    if v_list[src_idx][1] == v_list[dst_idx][0]:
                        e_list.append((src_idx, dst_idx))
            dl = DiGraph(num_v, e_list)
            dl_list.append(
                {"num_v": num_v, "e_list": e_list, "v_lbl": v_lbl, "dhg": dl}
            )
        return dl_list

    def remap(self, g_list, in_cnt, out_cnt, drop=False):
        for g_idx, g in enumerate(g_list):
            if "in_v_lbl" not in g:
                g["in_v_lbl"] = g["v_lbl"]
            for v_idx in range(g["num_v"]):
                in_cur_lbl = g["in_v_lbl"][v_idx]
                if in_cur_lbl not in self._in_subtree_map:
                    if drop:
                        g["in_v_lbl"][v_idx] = -1
                        continue
                    else:
                        self._in_subtree_map[in_cur_lbl] = len(self._in_subtree_map)
                g["in_v_lbl"][v_idx] = self._in_subtree_map[in_cur_lbl]
                in_cnt[g_idx][self._in_subtree_map[in_cur_lbl]] += 1
            if "out_v_lbl" not in g:
                g["out_v_lbl"] = g["v_lbl"]
            for v_idx in range(g["num_v"]):
                out_cur_lbl = g["out_v_lbl"][v_idx]
                if out_cur_lbl not in self._out_subtree_map:
                    if drop:
                        g["out_v_lbl"][v_idx] = -1
                        continue
                    else:
                        self._out_subtree_map[out_cur_lbl] = len(self._out_subtree_map)
                g["out_v_lbl"][v_idx] = self._out_subtree_map[out_cur_lbl]
                out_cnt[g_idx][self._out_subtree_map[out_cur_lbl]] += 1
        return g_list, in_cnt, out_cnt

    def count2mat(self, in_cnt, out_cnt):
        row_idx, col_idx, data, bias = [], [], [], len(self._in_subtree_map)
        for g_idx, g in enumerate(in_cnt):
            for lbl_idx, cnt in g.items():
                row_idx.append(g_idx)
                col_idx.append(lbl_idx)
                data.append(cnt)
        for g_idx, g in enumerate(out_cnt):
            for lbl_idx, cnt in g.items():
                row_idx.append(g_idx)
                col_idx.append(lbl_idx + bias)
                data.append(cnt)
        return (
            torch.sparse_coo_tensor(
                torch.tensor([row_idx, col_idx]),
                torch.tensor(data),
                size=(len(in_cnt), bias + len(self._out_subtree_map)),
            )
            .coalesce()
            .float()
        )

    def fit_transform(self, hg_list):
        e_lbl = [hg["e_lbl"] for hg in hg_list]
        dl_list = self.directed_line_expansion(hg_list, e_lbl)
        self._in_cnt = [defaultdict(int) for _ in range(len(dl_list))]
        self._out_cnt = [defaultdict(int) for _ in range(len(dl_list))]
        self.remap(dl_list, self._in_cnt, self._out_cnt)
        for _ in range(self.n_iter):
            for dl in dl_list:
                in_tmp, out_tmp = [], []
                for v_idx in range(dl["num_v"]):
                    in_cur_lbl = dl["in_v_lbl"][v_idx]
                    in_nbr_lbl = sorted(
                        [dl["in_v_lbl"][u_idx] for u_idx in dl["dhg"].nbr_v_in(v_idx)]
                    )
                    out_cur_lbl = dl["out_v_lbl"][v_idx]
                    out_nbr_lbl = sorted(
                        [dl["out_v_lbl"][u_idx] for u_idx in dl["dhg"].nbr_v_out(v_idx)]
                    )
                    in_tmp.append(f"{in_cur_lbl},{in_nbr_lbl}")
                    out_tmp.append(f"{out_cur_lbl},{out_nbr_lbl}")
                dl["in_v_lbl"] = in_tmp
                dl["out_v_lbl"] = out_tmp
            self.remap(dl_list, self._in_cnt, self._out_cnt)
        self.train_cnt = self.count2mat(self._in_cnt, self._out_cnt)
        self.train_ft = self.train_cnt.mm(self.train_cnt.t()).to_dense()
        if self.normalize:
            self.train_ft_diag = torch.diag(self.train_ft)
            self.train_ft = (
                self.train_ft
                / torch.outer(self.train_ft_diag, self.train_ft_diag).sqrt()
            )
            self.train_ft[torch.isnan(self.train_ft)] = 0
        return self.train_ft

    def transform(self, hg_list):
        e_lbl = [hg["e_lbl"] for hg in hg_list]
        dl_list = self.directed_line_expansion(hg_list, e_lbl)
        in_cnt, out_cnt = [defaultdict(int) for _ in range(len(dl_list))], [
            defaultdict(int) for _ in range(len(dl_list))
        ]
        self.remap(dl_list, in_cnt, out_cnt, drop=True)
        for _ in range(self.n_iter):
            for hg in dl_list:
                in_tmp, out_tmp = [], []
                for v_idx in range(hg["num_v"]):
                    in_cur_lbl = hg["in_v_lbl"][v_idx]
                    in_nbr_lbl = sorted(
                        [hg["in_v_lbl"][u_idx] for u_idx in hg["dhg"].nbr_v_in(v_idx)]
                    )
                    out_cur_lbl = hg["out_v_lbl"][v_idx]
                    out_nbr_lbl = sorted(
                        [hg["out_v_lbl"][u_idx] for u_idx in hg["dhg"].nbr_v_out(v_idx)]
                    )
                    in_tmp.append(f"{in_cur_lbl},{in_nbr_lbl}")
                    out_tmp.append(f"{out_cur_lbl},{out_nbr_lbl}")
                hg["in_v_lbl"] = in_tmp
                hg["out_v_lbl"] = out_tmp
            self.remap(dl_list, in_cnt, out_cnt, drop=True)
        test_cnt = self.count2mat(in_cnt, out_cnt)
        test_ft = test_cnt.mm(self.train_cnt.t()).to_dense()
        if self.normalize:
            test_ft_diag = torch.sparse.sum(test_cnt * test_cnt, dim=1).to_dense()
            test_ft = test_ft / torch.outer(test_ft_diag, self.train_ft_diag).sqrt()
            test_ft[torch.isnan(test_ft)] = 0
        return test_ft
