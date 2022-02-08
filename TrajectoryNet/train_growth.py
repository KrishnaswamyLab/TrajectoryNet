import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import scprep
import torch
import time

from TrajectoryNet import dataset
from .optimal_transport.sinkhorn_knopp_unbalanced import sinkhorn_knopp_unbalanced


eb_data = dataset.EBData("pcs", max_dim=5)


def get_transform_matrix(gamma, a, epsilon=1e-8):
    """Return matrix such that T @ a = b
    gamma : gamma @ 1 = a; gamma^T @ 1 = b
    """
    return (np.diag(1.0 / (a + epsilon)) @ gamma).T


def get_growth_coeffs(gamma, a, epsilon=1e-8, normalize=False):
    T = get_transform_matrix(gamma, a, epsilon)
    unnormalized_coeffs = np.sum(T, axis=0)
    if not normalize:
        return unnormalized_coeffs
    return unnormalized_coeffs / np.sum(unnormalized_coeffs) * len(unnormalized_coeffs)


data, labels = eb_data.data, eb_data.get_times()

# Compute couplings

timepoints = np.unique(labels)
print("timepoints", timepoints)

dfs = [data[labels == tp] for tp in timepoints]
pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4)]


def get_all_growth_coeffs(alpha):
    gcs = []
    for a_ind, b_ind in pairs:
        start = time.time()
        print(a_ind, b_ind)
        a, b = dfs[a_ind], dfs[b_ind]
        m, n = a.shape[0], b.shape[0]
        M = cdist(a, b)
        entropy_reg = 0.1
        reg_1, reg_2 = alpha, 10000
        gamma = sinkhorn_knopp_unbalanced(
            np.ones(m) / m, np.ones(n) / n, M, entropy_reg, reg_1, reg_2
        )
        gc = get_growth_coeffs(gamma, np.ones(m) / m)
        gcs.append(gc)
        end = time.time()
        print("%s to %s took %0.2f sec" % (a_ind, b_ind, end - start))
    print(gcs)
    return gcs


gcs = np.load("../data/growth/gcs.npy", allow_pickle=True)


class GrowthNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(6, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def train(leaveout_tp):
    # Data, timepoint
    X = np.concatenate([data, labels[:, None]], axis=1)[
        (labels != timepoints[-1]) & (labels != leaveout_tp)
    ]
    if leaveout_tp == 1:
        Y = np.concatenate([gcs[4], gcs[2], gcs[3]])
    elif leaveout_tp == 2:
        Y = np.concatenate([gcs[5], gcs[0], gcs[3]])
    elif leaveout_tp == 3:
        Y = np.concatenate([gcs[6], gcs[0], gcs[1]])
    elif leaveout_tp == -1:
        Y = np.concatenate(gcs[:4])
    else:
        raise RuntimeError("Unknown leavout_tp %d" % leaveout_tp)
    print(X.shape, Y.shape)
    assert X.shape[0] == Y.shape[0]
    Y = Y[:, np.newaxis]

    model = GrowthNet().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    for it in range(100000):
        optimizer.zero_grad()
        batch_idx = np.random.randint(len(X), size=256)
        x = torch.from_numpy(X[batch_idx, :]).type(torch.float32).to(device)
        y = torch.from_numpy(Y[batch_idx, :]).type(torch.float32).to(device)
        negative_samples = np.concatenate(
            [
                np.random.uniform(size=(256, X.shape[1] - 1)) * 8 - 4,
                np.random.choice(timepoints, size=(256, 1)),
            ],
            axis=1,
        )
        negative_samples = (
            torch.from_numpy(negative_samples).type(torch.float32).to(device)
        )
        x = torch.cat([x, negative_samples])
        y = torch.cat([y, torch.ones_like(y)])
        pred = model(x)
        loss = torch.nn.MSELoss()
        output = loss(pred, y)
        output.backward()
        optimizer.step()
        if it % 100 == 0:
            print(it, output)

    torch.save(model, ("model_%d" % leaveout_tp))


# train(1)
# train(2)
# train(3)


def trajectory_to_video(savedir):
    import subprocess
    import os

    bashCommand = "ffmpeg -y -i {} {}".format(
        os.path.join(savedir, "viz-%05d.jpg"), os.path.join(savedir, "traj.mp4")
    )
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


trajectory_to_video("../data/growth/viz/")
