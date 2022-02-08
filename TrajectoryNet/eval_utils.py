import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from .optimal_transport.emd import earth_mover_distance


def generate_samples(device, args, model, growth_model, n=10000, timepoint=None):
    """generates samples using model and base density

    This is useful for measuring the wasserstein distance between the
    predicted distribution and the true distribution for evaluation
    purposes against other types of models. We should use
    negative log likelihood if possible as it is deterministic and
    more discriminative for this model type.

    TODO: Is this biased???
    """
    z_samples = args.data.base_sample()(n, *args.data.get_shape()).to(device)
    # Forward pass through the model / growth model
    with torch.no_grad():
        int_list = [
            torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
            for it in args.int_tps[: timepoint + 1]
        ]

        logpz = args.data.base_density()(z_samples)
        z = z_samples
        for it in int_list:
            z, logpz = model(z, logpz, integration_times=it, reverse=True)
        z = z.cpu().numpy()
        np.save(os.path.join(args.save, "samples_%0.2f.npy" % timepoint), z)
        logpz = logpz.cpu().numpy()
        plt.scatter(z[:, 0], z[:, 1], s=0.1, alpha=0.5)
        original_data = args.data.get_data()[args.data.get_times() == timepoint]
        idx = np.random.randint(original_data.shape[0], size=n)
        samples = original_data[idx, :]
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
        plt.savefig(os.path.join(args.save, "samples%d.png" % timepoint))
        plt.close()

        pz = np.exp(logpz)
        pz = pz / np.sum(pz)
        print(pz)

        print(
            earth_mover_distance(
                original_data, samples + np.random.randn(*samples.shape) * 0.1
            )
        )

        print(earth_mover_distance(z, original_data))
        print(earth_mover_distance(z, samples))
        # print(earth_mover_distance(z, original_data, weights1=pz.flatten()))
        # print(
        #    earth_mover_distance(
        #        args.data.get_data()[args.data.get_times() == (timepoint - 1)],
        #        original_data,
        #    )
        # )

    if args.use_growth and growth_model is not None:
        raise NotImplementedError(
            "generating samples with growth model is not yet implemented"
        )


def calculate_path_length(device, args, model, data, end_time, n_pts=10000):
    """Calculates the total length of the path from time 0 to timepoint"""
    # z_samples = torch.tensor(data.get_data()).type(torch.float32).to(device)
    z_samples = data.base_sample()(n_pts, *data.get_shape()).to(device)
    model.eval()
    n = 1001
    with torch.no_grad():
        integration_times = (
            torch.tensor(np.linspace(0, end_time, n)).type(torch.float32).to(device)
        )
        # z, _ = model(z_samples, torch.zeros_like(z_samples), integration_times=integration_times, reverse=False)
        z, _ = model(
            z_samples,
            torch.zeros_like(z_samples),
            integration_times=integration_times,
            reverse=True,
        )
        z = z.cpu().numpy()
        z_diff = np.diff(z, axis=0)
        z_lengths = np.sum(np.linalg.norm(z_diff, axis=-1), axis=0)
        total_length = np.mean(z_lengths)
        import ot as pot
        from scipy.spatial.distance import cdist

        emd = pot.emd2(
            np.ones(n_pts) / n_pts,
            np.ones(n_pts) / n_pts,
            cdist(z[-1, :, :], data.get_data()),
        )
        print(total_length, emd)
        plt.scatter(z[-1, :, 0], z[-1, :, 1])
        plt.savefig("test.png")
        plt.close()


def evaluate_mse(device, args, model, growth_model=None):
    if args.use_growth or growth_model is not None:
        print("WARNING: Ignoring growth model and computing anyway")

    paths = args.data.get_paths()

    z_samples = torch.tensor(paths[:, 0, :]).type(torch.float32).to(device)
    # Forward pass through the model / growth model
    with torch.no_grad():
        int_list = [
            torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
            for it in args.int_tps
        ]

        logpz = args.data.base_density()(z_samples)
        z = z_samples
        zs = []
        for it in int_list:
            z, _ = model(z, logpz, integration_times=it, reverse=True)
            zs.append(z.cpu().numpy())
        zs = np.stack(zs)
        np.save(os.path.join(args.save, "path_samples.npy"), zs)

        # logpz = logpz.cpu().numpy()
        # plt.scatter(z[:, 0], z[:, 1], s=0.1, alpha=0.5)
        mses = []
        print(zs.shape, paths[:, 1, :].shape)
        for tpi in range(len(args.timepoints)):
            mses.append(np.mean((paths[:, tpi + 1, :] - zs[tpi]) ** 2, axis=(-2, -1)))
        mses = np.array(mses)
        print(mses.shape)
        np.save(os.path.join(args.save, "mses.npy"), mses)
        return mses


def evaluate_kantorovich_v2(device, args, model, growth_model=None):
    """Eval the model via kantorovich distance on leftout timepoint

    v2 computes samples from subsequent timepoint instead of base distribution.
    this is arguably a fairer comparison to other methods such as WOT which are
    not model based this should accumulate much less numerical error in the
    integration procedure. However fixes to the number of samples to the number in the
    previous timepoint.

    If we have a growth model we should use this to modify the weighting of the
    points over time.
    """
    if args.use_growth or growth_model is not None:
        # raise NotImplementedError(
        #    "generating samples with growth model is not yet implemented"
        # )
        print("WARNING: Ignoring growth model and computing anyway")

    # Backward pass through the model / growth model
    with torch.no_grad():
        int_times = torch.tensor(
            [
                args.int_tps[args.leaveout_timepoint],
                args.int_tps[args.leaveout_timepoint + 1],
            ]
        )
        int_times = int_times.type(torch.float32).to(device)
        next_z = args.data.get_data()[
            args.data.get_times() == args.leaveout_timepoint + 1
        ]
        next_z = torch.from_numpy(next_z).type(torch.float32).to(device)
        prev_z = args.data.get_data()[
            args.data.get_times() == args.leaveout_timepoint - 1
        ]
        prev_z = torch.from_numpy(prev_z).type(torch.float32).to(device)
        zero = torch.zeros(next_z.shape[0], 1).to(device)
        z_backward, _ = model.chain[0](next_z, zero, integration_times=int_times)
        z_backward = z_backward.cpu().numpy()
        int_times = torch.tensor(
            [
                args.int_tps[args.leaveout_timepoint - 1],
                args.int_tps[args.leaveout_timepoint],
            ]
        )
        zero = torch.zeros(prev_z.shape[0], 1).to(device)
        z_forward, _ = model.chain[0](
            prev_z, zero, integration_times=int_times, reverse=True
        )
        z_forward = z_forward.cpu().numpy()

        emds = []
        for tpi in [args.leaveout_timepoint]:
            original_data = args.data.get_data()[
                args.data.get_times() == args.timepoints[tpi]
            ]
            emds.append(earth_mover_distance(z_backward, original_data))
            emds.append(earth_mover_distance(z_forward, original_data))

        emds = np.array(emds)
        np.save(os.path.join(args.save, "emds_v2.npy"), emds)
        return emds


def evaluate_kantorovich(device, args, model, growth_model=None, n=10000):
    """Eval the model via kantorovich distance on all timepoints

    compute samples forward from the starting parametric distribution keeping track
    of growth rate to scale the final distribution.

    The growth model is a single model of time independent cell growth /
    death rate defined as a variation from uniform.

    If we have a growth model we should use this to modify the weighting of the
    points over time.
    """
    if args.use_growth or growth_model is not None:
        # raise NotImplementedError(
        #    "generating samples with growth model is not yet implemented"
        # )
        print("WARNING: Ignoring growth model and computing anyway")

    z_samples = args.data.base_sample()(n, *args.data.get_shape()).to(device)
    # Forward pass through the model / growth model
    with torch.no_grad():
        int_list = []
        for i, it in enumerate(args.int_tps):
            if i == 0:
                prev = 0.0
            else:
                prev = args.int_tps[i - 1]
            int_list.append(torch.tensor([prev, it]).type(torch.float32).to(device))

        # int_list = [
        #    torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
        #    for it in args.int_tps
        # ]
        print(args.int_tps)

        logpz = args.data.base_density()(z_samples)
        z = z_samples
        zs = []
        growthrates = [torch.ones(z_samples.shape[0], 1).to(device)]
        for it, tp in zip(int_list, args.timepoints):
            z, _ = model(z, logpz, integration_times=it, reverse=True)
            zs.append(z.cpu().numpy())
            if args.use_growth:
                time_state = tp * torch.ones(z.shape[0], 1).to(device)
                full_state = torch.cat([z, time_state], 1)
                # Multiply growth rates together to get total mass along path
                growthrates.append(
                    torch.clamp(growth_model(full_state), 1e-4, 1e4) * growthrates[-1]
                )
        zs = np.stack(zs)
        if args.use_growth:
            growthrates = growthrates[1:]
            growthrates = torch.stack(growthrates)
            growthrates = growthrates.cpu().numpy()
            np.save(os.path.join(args.save, "sample_weights.npy"), growthrates)
        np.save(os.path.join(args.save, "samples.npy"), zs)

        # logpz = logpz.cpu().numpy()
        # plt.scatter(z[:, 0], z[:, 1], s=0.1, alpha=0.5)
        emds = []
        for tpi in range(len(args.timepoints)):
            original_data = args.data.get_data()[
                args.data.get_times() == args.timepoints[tpi]
            ]
            if args.use_growth:
                emds.append(
                    earth_mover_distance(
                        zs[tpi], original_data, weights1=growthrates[tpi].flatten()
                    )
                )
            else:
                emds.append(earth_mover_distance(zs[tpi], original_data))

        # Add validation point kantorovich distance evaluation
        if args.data.has_validation_samples():
            for tpi in np.unique(args.data.val_labels):
                original_data = args.data.val_data[
                    args.data.val_labels == args.timepoints[tpi]
                ]
                if args.use_growth:
                    emds.append(
                        earth_mover_distance(
                            zs[tpi], original_data, weights1=growthrates[tpi].flatten()
                        )
                    )
                else:
                    emds.append(earth_mover_distance(zs[tpi], original_data))

        emds = np.array(emds)
        print(emds)
        np.save(os.path.join(args.save, "emds.npy"), emds)
        return emds


def evaluate(device, args, model, growth_model=None):
    """Eval the model via negative log likelihood on all timepoints

    Compute loss by integrating backwards from the last time step
    At each time step integrate back one time step, and concatenate that
    to samples of the empirical distribution at that previous timestep
    repeating over and over to calculate the likelihood of samples in
    later timepoints iteratively, making sure that the ODE is evaluated
    at every time step to calculate those later points.

    The growth model is a single model of time independent cell growth /
    death rate defined as a variation from uniform.
    """
    use_growth = args.use_growth and growth_model is not None

    # Backward pass accumulating losses, previous state and deltas
    deltas = []
    zs = []
    z = None
    for i, (itp, tp) in enumerate(zip(args.int_tps[::-1], args.timepoints[::-1])):
        # tp counts down from last
        integration_times = torch.tensor([itp - args.time_scale, itp])
        integration_times = integration_times.type(torch.float32).to(device)

        x = args.data.get_data()[args.data.get_times() == tp]
        x = torch.from_numpy(x).type(torch.float32).to(device)

        if i > 0:
            x = torch.cat((z, x))
            zs.append(z)
        zero = torch.zeros(x.shape[0], 1).to(x)

        # transform to previous timepoint
        z, delta_logp = model(x, zero, integration_times=integration_times)
        deltas.append(delta_logp)

    logpz = args.data.base_density()(z)

    # build growth rates
    if use_growth:
        growthrates = [torch.ones_like(logpz)]
        for z_state, tp in zip(zs[::-1], args.timepoints[::-1][1:]):
            # Full state includes time parameter to growth_model
            time_state = tp * torch.ones(z_state.shape[0], 1).to(z_state)
            full_state = torch.cat([z_state, time_state], 1)
            growthrates.append(growth_model(full_state))

    # Accumulate losses
    losses = []
    logps = [logpz]
    for i, (delta_logp, tp) in enumerate(zip(deltas[::-1], args.timepoints)):
        n_cells_in_tp = np.sum(args.data.get_times() == tp)
        logpx = logps[-1] - delta_logp
        if use_growth:
            logpx += torch.log(growthrates[i])
        logps.append(logpx[:-n_cells_in_tp])
        losses.append(-torch.sum(logpx[-n_cells_in_tp:]))
    losses = torch.stack(losses).cpu().numpy()
    np.save(os.path.join(args.save, "nll.npy"), losses)
    return losses
