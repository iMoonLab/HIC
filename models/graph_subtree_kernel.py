import torch
from collections import defaultdict


class GraphSubtreeKernel:
    def __init__(self, n_iter=4, normalize=True):
        self.n_iter = n_iter
        self.normalize = normalize
        self._subtree_map = {}

    def remap(self, g_list, count, drop=False):
        for g_idx, g in enumerate(g_list):
            for v_idx in range(g["num_v"]):
                cur_lbl = g["v_lbl"][v_idx]
                if cur_lbl not in self._subtree_map:
                    if drop:
                        g["v_lbl"][v_idx] = -1
                        continue
                    else:
                        self._subtree_map[cur_lbl] = len(self._subtree_map)
                g["v_lbl"][v_idx] = self._subtree_map[cur_lbl]
                count[g_idx][self._subtree_map[cur_lbl]] += 1
        return g_list, count

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
                size=(len(count), len(self._subtree_map)),
            )
            .coalesce()
            .float()
        )

    def fit_transform(self, g_list):
        self._count = [defaultdict(int) for _ in range(len(g_list))]
        self.remap(g_list, self._count)
        for _ in range(self.n_iter):
            for g in g_list:
                tmp = []
                for v_idx in range(g["num_v"]):
                    cur_lbl = g["v_lbl"][v_idx]
                    nbr_lbl = sorted(
                        [g["v_lbl"][u_idx] for u_idx in g["dhg"].nbr_v(v_idx)]
                    )
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                g["v_lbl"] = tmp
            self.remap(g_list, self._count)
        self.train_cnt = self.count2mat(self._count)
        self.train_ft = self.train_cnt.mm(self.train_cnt.t()).to_dense()
        if self.normalize:
            self.train_ft_diag = torch.diag(self.train_ft)
            self.train_ft = (
                self.train_ft
                / torch.outer(self.train_ft_diag, self.train_ft_diag).sqrt()
            )
            self.train_ft[torch.isnan(self.train_ft)] = 0
        return self.train_ft

    def transform(self, g_list):
        count = [defaultdict(int) for _ in range(len(g_list))]
        self.remap(g_list, count, drop=True)
        for _ in range(self.n_iter):
            for g in g_list:
                tmp = []
                for v_idx in range(g["num_v"]):
                    cur_lbl = g["v_lbl"][v_idx]
                    nbr_lbl = sorted(
                        [g["v_lbl"][u_idx] for u_idx in g["dhg"].nbr_v(v_idx)]
                    )
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                g["v_lbl"] = tmp
            self.remap(g_list, count, drop=True)
        test_cnt = self.count2mat(count)
        test_ft = test_cnt.mm(self.train_cnt.t()).to_dense()
        if self.normalize:
            test_ft_diag = torch.sparse.sum(test_cnt * test_cnt, dim=1).to_dense()
            test_ft = test_ft / torch.outer(test_ft_diag, self.train_ft_diag).sqrt()
            test_ft[torch.isnan(test_ft)] = 0
        return test_ft
