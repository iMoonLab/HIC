from itertools import combinations
from collections import defaultdict

import torch
import numpy as np
from dhg import Hypergraph, Graph, DiGraph


# Hypergraph WL Hyperedge Kernel from "Feng et al. 2024 IEEE TPAMI Hypergraph Isomorphism Computation"
class HypergraphHyedgeKernel:
    def __init__(self, n_iter=4, degree_as_label=True, normalize=True):
        self.n_iter = n_iter
        self.normalize = normalize
        self.degree_as_label = degree_as_label
        self._subtree_map = {}
        self._e_map = {}

    def remap_v(self, hg_list, drop=False):
        for hg_idx, hg in enumerate(hg_list):
            for v_idx in range(hg["num_v"]):
                cur_lbl = hg["v_lbl"][v_idx]
                cur_lbl = "v" + str(cur_lbl)
                if cur_lbl not in self._subtree_map:
                    if drop:
                        hg["v_lbl"][v_idx] = -1
                        continue
                    else:
                        self._subtree_map[cur_lbl] = len(self._subtree_map)
                hg["v_lbl"][v_idx] = self._subtree_map[cur_lbl]
        return hg_list

    def remap_e(self, hg_list, cnt, drop=False):
        for hg_idx, hg in enumerate(hg_list):
            for e_idx in range(hg["dhg"].num_e):
                cur_lbl = hg["e_lbl"][e_idx]
                cur_lbl = "e" + str(cur_lbl)
                if cur_lbl not in self._subtree_map:
                    if drop:
                        hg["e_lbl"][e_idx] = -1
                        continue
                    else:
                        self._subtree_map[cur_lbl] = len(self._subtree_map)
                hg["e_lbl"][e_idx] = self._subtree_map[cur_lbl]
        for hg_idx, hg in enumerate(hg_list):
            for e_idx in range(hg["dhg"].num_e):
                cur_nbr_lbl = str(
                    sorted(hg["v_lbl"][v_idx] for v_idx in hg["dhg"].nbr_v(e_idx))
                )
                if cur_nbr_lbl not in self._e_map:
                    if drop:
                        continue
                    self._e_map[cur_nbr_lbl] = len(self._e_map)
                cnt[hg_idx][self._e_map[cur_nbr_lbl]] += 1
        return hg_list, cnt

    def cnt2mat(self, cnt):
        row_idx, col_idx, data = [], [], []
        for idx, g in enumerate(cnt):
            for lbl, c in g.items():
                row_idx.append(idx)
                col_idx.append(lbl)
                data.append(c)
        return (
            torch.sparse_coo_tensor(
                torch.tensor([row_idx, col_idx]),
                torch.tensor(data),
                size=(len(cnt), len(self._e_map)),
            )
            .coalesce()
            .float()
        )

    def fit_transform(self, hg_list):
        # if self.degree_as_label:
        #     for hg in hg_list:
        #         hg["v_lbl"] = [int(v) for v in hg["dhg"].deg_v]
        #         hg["e_lbl"] = [int(e) for e in hg["dhg"].deg_e]
        self._cnt = [defaultdict(int) for _ in range(len(hg_list))]
        self.remap_v(hg_list)
        self.remap_e(hg_list, self._cnt)
        for _ in range(self.n_iter):
            for hg in hg_list:
                tmp = []
                for e_idx in range(hg["dhg"].num_e):
                    cur_lbl = hg["e_lbl"][e_idx]
                    nbr_lbl = sorted(
                        hg["v_lbl"][v_idx] for v_idx in hg["dhg"].nbr_v(e_idx)
                    )
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                hg["e_lbl"] = tmp
            self.remap_e(hg_list, self._cnt)
            for hg in hg_list:
                tmp = []
                for v_idx in range(hg["dhg"].num_v):
                    cur_lbl = hg["v_lbl"][v_idx]
                    nbr_lbl = sorted(
                        hg["e_lbl"][e_idx] for e_idx in hg["dhg"].nbr_e(v_idx)
                    )
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                hg["v_lbl"] = tmp
            self.remap_v(hg_list)
        self.train_cnt = self.cnt2mat(self._cnt)
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
        # if self.degree_as_label:
        #     for hg in hg_list:
        #         hg["v_lbl"] = [int(v) for v in hg["dhg"].deg_v]
        #         hg["e_lbl"] = [int(e) for e in hg["dhg"].deg_e]
        cnt = [defaultdict(int) for _ in range(len(hg_list))]
        self.remap_v(hg_list, drop=True)
        self.remap_e(hg_list, cnt, drop=True)
        for _ in range(self.n_iter):
            for hg in hg_list:
                tmp = []
                for e_idx in range(hg["dhg"].num_e):
                    cur_lbl = hg["e_lbl"][e_idx]
                    nbr_lbl = sorted(
                        hg["v_lbl"][v_idx] for v_idx in hg["dhg"].nbr_v(e_idx)
                    )
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                hg["e_lbl"] = tmp
            self.remap_e(hg_list, cnt, drop=True)
            for hg in hg_list:
                tmp = []
                for v_idx in range(hg["dhg"].num_v):
                    cur_lbl = hg["v_lbl"][v_idx]
                    nbr_lbl = sorted(
                        hg["e_lbl"][e_idx] for e_idx in hg["dhg"].nbr_e(v_idx)
                    )
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                hg["v_lbl"] = tmp
            self.remap_v(hg_list, drop=True)
        test_cnt = self.cnt2mat(cnt)
        test_ft = test_cnt.mm(self.train_cnt.t()).to_dense()
        if self.normalize:
            test_ft_diag = torch.sparse.sum(test_cnt * test_cnt, dim=1).to_dense()
            test_ft = test_ft / torch.outer(test_ft_diag, self.train_ft_diag).sqrt()
            test_ft[torch.isnan(test_ft)] = 0
        return test_ft
