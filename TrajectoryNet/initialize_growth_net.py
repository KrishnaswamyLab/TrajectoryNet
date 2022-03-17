"""
Initializes the growth network using multiscale sampling from Bermanis et al. 2013.


"""
from lib.growth_net import GrowthNet
from optimal_transport.sinkhorn_knopp_unbalanced import sinkhorn_knopp_unbalanced
from scipy.spatial.distance import cdist

import numpy as np
import os
import time
import torch


def get_transform_matrix(gamma, a, epsilon=1e-8):
    """ Return matrix such that T @ a = b 
    gamma : gamma @ 1 = a; gamma^T @ 1 = b
    """
    return (np.diag(1.0 / (a + epsilon)) @ gamma).T


def get_growth_coeffs(gamma, a, epsilon=1e-8, normalize=False):
    T = get_transform_matrix(gamma, a, epsilon)
    unnormalized_coeffs = np.sum(T, axis=0)
    if not normalize:
        return unnormalized_coeffs
    return unnormalized_coeffs / np.sum(unnormalized_coeffs) * len(unnormalized_coeffs)


def calc_discrete_ot(args):
    """

    """
    print("Training growth network with alpha = %0.2f" % args.alpha)
    ds = args.data
    data, labels = ds.get_data(), ds.get_times()
    # args.timepoints, args.int_tps
    data_by_time = [data[labels == tp] for tp in args.timepoints]

    if args.leaveout_timepoint != -1:
        raise NotImplementedError

    growth_coeffs = []
    for i in range(len(args.timepoints) - 1):
        start = time.time()
        a, b = data_by_time[i], data_by_time[i + 1]
        m, n = a.shape[0], b.shape[0]
        M = cdist(a, b)
        entropy_reg = 0.1
        reg_1, reg_2 = args.alpha, 10000
        gamma = sinkhorn_knopp_unbalanced(
            np.ones(m) / m, np.ones(n) / n, M, entropy_reg, reg_1, reg_2
        )
        gc = get_growth_coeffs(gamma, np.ones(m) / m, normalize=True)
        log_gc = np.log2(gc)
        growth_coeffs.append(gc)
        end = time.time()
        print(
            "%s to %s took %0.2f sec, std. %0.3f, log std. %0.3f"
            % (i, i + 1, end - start, gc.std(), log_gc.std())
        )
    # , growth_coeffs.mean(axis=0), growth_coeffs.std(axis=0))
    return growth_coeffs


def train_growth_net(args, discrete_ot_coeffs):
    device = torch.device("cpu")
    ds = args.data
    data, labels = ds.get_data(), ds.get_times()

    X = np.concatenate([data, labels[:, None]], axis=1)[
        (labels != args.timepoints[-1]) & (labels != args.leaveout_timepoint)
    ]
    # Note, this assumes that our data is sorted by label
    # TODO fix this
    Y = np.concatenate(discrete_ot_coeffs)[:, np.newaxis]
    np.save(os.path.join(args.save,'growth_coeffs.npy'), Y)
    input_dim = X.shape[1]
    model = GrowthNet(input_dim).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    for it in range(1, 20000 + 1):
        optimizer.zero_grad()
        batch_idx = np.random.randint(len(X), size=256)
        x = torch.from_numpy(X[batch_idx, :]).type(torch.float32).to(device)
        y = torch.from_numpy(Y[batch_idx, :]).type(torch.float32).to(device)
        negative_samples = np.concatenate(
            [
                np.random.uniform(size=(256, X.shape[1] - 1)) * 8 - 4,
                np.random.choice(
                    args.timepoints[args.timepoints != args.leaveout_timepoint],
                    size=(256, 1),
                ),
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
        if it % 1000 == 0:
            print("Batch: %d, mse: %0.3f" % (it, output.item()))
    # torch.save(model, ("model_%d" % leaveout_tp))
    return model


def init_growth_net(args):
    print("Initializing growth network")
    discrete_ot_coeffs = calc_discrete_ot(args)
    model = train_growth_net(args, discrete_ot_coeffs)
    torch.save(model, (os.path.join(args.save, "growth_model.pt")))
    return model
