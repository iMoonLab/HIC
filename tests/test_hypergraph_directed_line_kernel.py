from pytest import approx
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import BinaryRelevance

from dhg import Graph

from utils import load_data
from models import HypergraphRootedKernel, GraphSubtreeKernel, GraphletSampling, HypergraphDirectedLineKernel


G = load_data("RHG_6_seed_0")
# y = [-1 if x["g_lbl"] != 1 else 1 for x in G]
y = [x["g_lbl"] for x in G]
y = MultiLabelBinarizer().fit_transform(y)

# for g in G:
#     g["dhg"] = Graph.from_hypergraph_clique(g["dhg"])
#     g["num_e"] = g["dhg"].num_e
#     g["e_list"] = g["dhg"].e[0]
# m = WLSubtree(n_iter=4, normalize=True)
# ---------------------------------------
# for g in G:
#     g["dhg"] = Graph.from_hypergraph_clique(g["dhg"])
#     g["num_e"] = g["dhg"].num_e
#     g["e_list"] = g["dhg"].e[0]
# m = GraphletSampling(normalize=True, sampling={})
# ---------------------------------------
# m = HypergraphRootedKernel(normalize=True)
# ---------------------------------------
m = HypergraphDirectedLineKernel(normalize=True)

G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=0)
K_train = m.fit_transform(G_train).cpu().numpy()
K_test = m.transform(G_test).cpu().numpy()
print(f"feature extraction done")
# clf = SVC(kernel="precomputed")
clf = BinaryRelevance(
    classifier=SVC(kernel="precomputed"),
    require_dense=[True, True],
)
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", str(round(acc * 100, 2)) + "%")
print("Hamming Distance: ", hamming_loss(y_test, y_pred))
# assert acc == approx(0.8947, 0.001)
