from collections import defaultdict

import torch
import numpy as np


# Hypergraph Kernel from "Learning from Interpretations: A Rooted Kernel for Ordered Hypergraphs"
class HypergraphRootedKernel:
    def __init__(self, max_walk_len=4, gamma=0.5, normalize=True, degree_as_label=True):
        self.n_seq = 20
        self.gamma = gamma
        self.max_walk_len = max_walk_len
        self.normalize = normalize
        self.degree_as_label = degree_as_label
        self._seq_map = {idx: dict() for idx in range(1, max_walk_len + 1)}

    def count2mat(self, count, l):
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
                size=(len(count), len(self._seq_map[l])),
            )
            .coalesce()
            .float()
        )

    def fuse(self, fts):
        res = None
        for i, ft in fts.items():
            if res is None:
                res = ft * self.gamma**i
            else:
                res += ft * self.gamma**i
        return res

    def fit_transform(self, hg_list):
        # initialize the sequence count
        self._count = {
            idx: [defaultdict(int) for _ in range(len(hg_list))]
            for idx in range(1, self.max_walk_len + 1)
        }
        # initialize the hyperedge feature
        if not self.degree_as_label:
            e_lbl = [hg["e_lbl"] for hg in hg_list]
        else:
            e_lbl = [[int(e) for e in hg["dhg"].deg_e] for hg in hg_list]
        # initialize the hyperedge transfer matrix
        # P = D_e^-1 H^T W D_v^-1 H
        T = [
            hg["dhg"]
            .W_e.mm(hg["dhg"].D_e_neg_1)
            .mm(hg["dhg"].H.t())
            .mm(hg["dhg"].D_v_neg_1)
            .mm(hg["dhg"].H)
            .to_dense()
            .numpy()
            for hg in hg_list
        ]
        # estimate the sequence count
        # N_e, D_e, D_v = [], [], []
        # for hg in hg_list:
        #     N_e.append(hg["dhg"].num_e)
        #     D_e.append(np.mean(hg["dhg"].deg_e))
        #     D_v.append(np.mean(hg["dhg"].deg_v))
        # N_e, D_e, D_v = np.mean(N_e), np.mean(D_e), np.mean(D_v)
        # self.n_seq = int(N_e * (D_e * D_v) ** self.max_walk_len / 100)
        # self.prob_len = [(D_e * D_v)**i for i in range(self.max_walk_len)]
        self.prob_len = [2**i for i in range(1, self.max_walk_len + 1)]
        self.prob_len = np.array(self.prob_len) / np.sum(self.prob_len)
        # walk on the hypergraph
        for hg_idx, hg in enumerate(hg_list):
            # print(f"Processing training hypergraph {hg_idx}/{len(hg_list)}")
            # estimate the sequence count
            if self.n_seq == -1:
                n_seq = int(
                    hg["dhg"].num_e
                    * (np.mean(hg["dhg"].deg_e) * np.mean(hg["dhg"].deg_v))
                    ** self.max_walk_len
                    / 200
                )
            else:
                n_seq = self.n_seq
            num_e = hg["dhg"].num_e
            seq_list = []
            for _ in range(n_seq):
                cur_len = np.random.choice(self.max_walk_len, p=self.prob_len)
                seq_idx = [np.random.choice(num_e)]
                seq = [e_lbl[hg_idx][seq_idx[-1]]]
                for _ in range(cur_len):
                    seq_idx.append(np.random.choice(num_e, p=T[hg_idx][seq_idx[-1]]))
                    seq.append(e_lbl[hg_idx][seq_idx[-1]])
                seq_list.append(seq)
            # count the sequence
            for seq in seq_list:
                code = ",".join([str(s) for s in seq])
                if code not in self._seq_map[len(seq)]:
                    self._seq_map[len(seq)][code] = len(self._seq_map[len(seq)])
                self._count[len(seq)][hg_idx][self._seq_map[len(seq)][code]] += 1
        # compute the kernel matrix
        self.raw_train_cnt = {l: self.count2mat(c, l) for l, c in self._count.items()}
        self.raw_train_ft = {
            l: self.raw_train_cnt[l].mm(self.raw_train_cnt[l].t()).to_dense()
            for l in self.raw_train_cnt
        }
        self.train_ft = self.fuse(self.raw_train_ft)
        if self.normalize:
            self.train_ft_diag = torch.diag(self.train_ft)
            self.train_ft = (
                self.train_ft
                / torch.outer(self.train_ft_diag, self.train_ft_diag).sqrt()
            )
            self.train_ft[torch.isnan(self.train_ft)] = 0
        return self.train_ft

    def transform(self, hg_list):
        # initialize the sequence count
        count = {
            idx: [defaultdict(int) for _ in range(len(hg_list))]
            for idx in range(1, self.max_walk_len + 1)
        }
        # initialize the hyperedge feature
        if not self.degree_as_label:
            e_lbl = [hg["e_lbl"] for hg in hg_list]
        else:
            e_lbl = [[int(e) for e in hg["dhg"].deg_e] for hg in hg_list]
        # initialize the hyperedge transfer matrix
        # P = D_e^-1 H^T W D_v^-1 H
        T = [
            hg["dhg"]
            .W_e.mm(hg["dhg"].D_e_neg_1)
            .mm(hg["dhg"].H.t())
            .mm(hg["dhg"].D_v_neg_1)
            .mm(hg["dhg"].H)
            .to_dense()
            .numpy()
            for hg in hg_list
        ]
        # walk on the hypergraph
        for hg_idx, hg in enumerate(hg_list):
            print(f"Processing testing hypergraph {hg_idx}/{len(hg_list)}")
            # estimate the sequence count
            if self.n_seq == -1:
                n_seq = int(
                    hg["dhg"].num_e
                    * (np.mean(hg["dhg"].deg_e) * np.mean(hg["dhg"].deg_v))
                    ** self.max_walk_len
                    / 200
                )
            else:
                n_seq = self.n_seq
            num_e = hg["dhg"].num_e
            seq_list = []
            for _ in range(n_seq):
                cur_len = np.random.choice(self.max_walk_len, p=self.prob_len)
                seq_idx = [np.random.choice(num_e)]
                seq = [e_lbl[hg_idx][seq_idx[-1]]]
                for _ in range(cur_len):
                    seq_idx.append(np.random.choice(num_e, p=T[hg_idx][seq_idx[-1]]))
                    seq.append(e_lbl[hg_idx][seq_idx[-1]])
                seq_list.append(seq)
            # count the sequence
            for seq in seq_list:
                code = ",".join([str(s) for s in seq])
                if code not in self._seq_map[len(seq)]:
                    continue
                count[len(seq)][hg_idx][self._seq_map[len(seq)][code]] += 1
        # compute the kernel matrix
        raw_test_cnt = {l: self.count2mat(c, l) for l, c in count.items()}
        raw_test_ft = {
            l: raw_test_cnt[l].mm(self.raw_train_cnt[l].t()).to_dense()
            for l in raw_test_cnt
        }
        test_ft = self.fuse(raw_test_ft)
        if self.normalize:
            test_ft_diag = self.fuse(
                {
                    l: torch.sparse.sum(tc * tc, dim=1).to_dense()
                    for l, tc in raw_test_cnt.items()
                }
            )
            test_ft = test_ft / torch.outer(test_ft_diag, self.train_ft_diag).sqrt()
            test_ft[torch.isnan(test_ft)] = 0
        return test_ft
