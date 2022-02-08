import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

from TrajectoryNet import dataset, eval_utils
from TrajectoryNet.parse import parser
from TrajectoryNet.lib.growth_net import GrowthNet
from TrajectoryNet.lib.viz_scrna import trajectory_to_video, save_vectors
from TrajectoryNet.lib.viz_scrna import (
    save_trajectory_density,
    save_2d_trajectory,
    save_2d_trajectory_v2,
)

from TrajectoryNet.train_misc import (
    set_cnf_options,
    count_nfe,
    count_parameters,
    count_total_time,
    add_spectral_norm,
    spectral_norm_power_iteration,
    create_regularization_fns,
    get_regularization,
    append_regularization_to_log,
    build_model_tabular,
)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_trajectory(
    prior_logdensity,
    prior_sampler,
    model,
    data_samples,
    savedir,
    ntimes=101,
    end_times=None,
    memory=0.01,
    device="cpu",
):
    model.eval()

    #  Sample from prior
    z_samples = prior_sampler(1000, 2).to(device)

    # sample from a grid
    npts = 100
    side = np.linspace(-4, 4, npts)
    xx, yy = np.meshgrid(side, side)
    xx = torch.from_numpy(xx).type(torch.float32).to(device)
    yy = torch.from_numpy(yy).type(torch.float32).to(device)
    z_grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)

    with torch.no_grad():
        # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = prior_logdensity(z_samples)
        logp_grid = prior_logdensity(z_grid)
        t = 0
        for cnf in model.chain:

            # Construct integration_list
            if end_times is None:
                end_times = [(cnf.sqrt_end_time * cnf.sqrt_end_time)]
            integration_list = [torch.linspace(0, end_times[0], ntimes).to(device)]
            for i, et in enumerate(end_times[1:]):
                integration_list.append(
                    torch.linspace(end_times[i], et, ntimes).to(device)
                )
            full_times = torch.cat(integration_list, 0)
            print(full_times.shape)

            # Integrate over evenly spaced samples
            z_traj, logpz = cnf(
                z_samples,
                logp_samples,
                integration_times=integration_list[0],
                reverse=True,
            )
            full_traj = [(z_traj, logpz)]
            for int_times in integration_list[1:]:
                prev_z, prev_logp = full_traj[-1]
                z_traj, logpz = cnf(
                    prev_z[-1], prev_logp[-1], integration_times=int_times, reverse=True
                )
                full_traj.append((z_traj[1:], logpz[1:]))
            full_zip = list(zip(*full_traj))
            z_traj = torch.cat(full_zip[0], 0)
            # z_logp = torch.cat(full_zip[1], 0)
            z_traj = z_traj.cpu().numpy()

            grid_z_traj, grid_logpz_traj = [], []
            inds = torch.arange(0, z_grid.shape[0]).to(torch.int64)
            for ii in torch.split(inds, int(z_grid.shape[0] * memory)):
                _grid_z_traj, _grid_logpz_traj = cnf(
                    z_grid[ii],
                    logp_grid[ii],
                    integration_times=integration_list[0],
                    reverse=True,
                )
                full_traj = [(_grid_z_traj, _grid_logpz_traj)]
                for int_times in integration_list[1:]:
                    prev_z, prev_logp = full_traj[-1]
                    _grid_z_traj, _grid_logpz_traj = cnf(
                        prev_z[-1],
                        prev_logp[-1],
                        integration_times=int_times,
                        reverse=True,
                    )
                    full_traj.append((_grid_z_traj, _grid_logpz_traj))
                full_zip = list(zip(*full_traj))
                _grid_z_traj = torch.cat(full_zip[0], 0).cpu().numpy()
                _grid_logpz_traj = torch.cat(full_zip[1], 0).cpu().numpy()
                print(_grid_z_traj.shape)
                grid_z_traj.append(_grid_z_traj)
                grid_logpz_traj.append(_grid_logpz_traj)

            grid_z_traj = np.concatenate(grid_z_traj, axis=1)
            grid_logpz_traj = np.concatenate(grid_logpz_traj, axis=1)

            plt.figure(figsize=(8, 8))
            for _ in range(z_traj.shape[0]):

                plt.clf()

                # plot target potential function
                ax = plt.subplot(1, 1, 1, aspect="equal")

                """
                ax.hist2d(data_samples[:, 0], data_samples[:, 1], range=[[-4, 4], [-4, 4]], bins=200)
                ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_title("Target", fontsize=32)

                """
                # plot the density
                # ax = plt.subplot(2, 2, 2, aspect="equal")

                z, logqz = grid_z_traj[t], grid_logpz_traj[t]

                xx = z[:, 0].reshape(npts, npts)
                yy = z[:, 1].reshape(npts, npts)
                qz = np.exp(logqz).reshape(npts, npts)
                rgb = plt.cm.Spectral(t / z_traj.shape[0])
                print(t, rgb)
                background_color = "white"
                cvals = [0, np.percentile(qz, 0.1)]
                colors = [
                    background_color,
                    rgb,
                ]
                norm = plt.Normalize(min(cvals), max(cvals))
                tuples = list(zip(map(norm, cvals), colors))
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
                from matplotlib.colors import LogNorm

                plt.pcolormesh(
                    xx,
                    yy,
                    qz,
                    # norm=LogNorm(vmin=qz.min(), vmax=qz.max()),
                    cmap=cmap,
                )
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                cmap = matplotlib.cm.get_cmap(None)
                ax.set_facecolor(background_color)
                ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_title("Density", fontsize=32)

                """
                # plot the samples
                ax = plt.subplot(2, 2, 3, aspect="equal")

                zk = z_traj[t]
                ax.hist2d(zk[:, 0], zk[:, 1], range=[[-4, 4], [-4, 4]], bins=200)
                ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_title("Samples", fontsize=32)

                # plot vector field
                ax = plt.subplot(2, 2, 4, aspect="equal")

                K = 13j
                y, x = np.mgrid[-4:4:K, -4:4:K]
                K = int(K.imag)
                zs = torch.from_numpy(np.stack([x, y], -1).reshape(K * K, 2)).to(device, torch.float32)
                logps = torch.zeros(zs.shape[0], 1).to(device, torch.float32)
                dydt = cnf.odefunc(full_times[t], (zs, logps))[0]
                dydt = -dydt.cpu().detach().numpy()
                dydt = dydt.reshape(K, K, 2)

                logmag = 2 * np.log(np.hypot(dydt[:, :, 0], dydt[:, :, 1]))
                ax.quiver(
                    x, y, dydt[:, :, 0], -dydt[:, :, 1],
                    # x, y, dydt[:, :, 0], dydt[:, :, 1],
                    np.exp(logmag), cmap="coolwarm", scale=20., width=0.015, pivot="mid"
                )
                ax.set_xlim(-4, 4)
                ax.set_ylim(4, -4)
                #ax.set_ylim(-4, 4)
                ax.axis("off")
                ax.set_title("Vector Field", fontsize=32)
                """

                makedirs(savedir)
                plt.savefig(os.path.join(savedir, f"viz-{t:05d}.jpg"))
                t += 1


def get_trajectory_samples(device, model, data, n=2000):
    ntimes = 5
    model.eval()
    z_samples = data.base_sample()(n, 2).to(device)

    integration_list = [torch.linspace(0, args.int_tps[0], ntimes).to(device)]
    for i, et in enumerate(args.int_tps[1:]):
        integration_list.append(torch.linspace(args.int_tps[i], et, ntimes).to(device))
    print(integration_list)


def plot_output(device, args, model, data):
    # logger.info('Plotting trajectory to {}'.format(save_traj_dir))
    data_samples = data.get_data()[data.sample_index(2000, 0)]
    start_points = data.base_sample()(1000, 2)
    # start_points = data.get_data()[idx]
    # start_points = torch.from_numpy(start_points).type(torch.float32)
    """
    save_vectors(
        data.base_density(),
        model,
        start_points,
        data.get_data()[data.get_times() == 1],
        data.get_times()[data.get_times() == 1],
        args.save,
        device=device,
        end_times=args.int_tps,
        ntimes=100,
        memory=1.0,
        lim=1.5,
    )
    save_traj_dir = os.path.join(args.save, "trajectory_2d")
    save_2d_trajectory_v2(
        data.base_density(),
        data.base_sample(),
        model,
        data_samples,
        save_traj_dir,
        device=device,
        end_times=args.int_tps,
        ntimes=3,
        memory=1.0,
        limit=2.5,
    )
    """

    density_dir = os.path.join(args.save, "density2")
    save_trajectory_density(
        data.base_density(),
        model,
        data_samples,
        density_dir,
        device=device,
        end_times=args.int_tps,
        ntimes=100,
        memory=1,
    )
    trajectory_to_video(density_dir)


def integrate_backwards(
    end_samples, model, savedir, ntimes=100, memory=0.1, device="cpu"
):
    """Integrate some samples backwards and save the results."""
    with torch.no_grad():
        z = torch.from_numpy(end_samples).type(torch.float32).to(device)
        zero = torch.zeros(z.shape[0], 1).to(z)
        cnf = model.chain[0]

        zs = [z]
        deltas = []
        int_tps = np.linspace(args.int_tps[0], args.int_tps[-1], ntimes)
        for i, itp in enumerate(int_tps[::-1][:-1]):
            # tp counts down from last
            timescale = int_tps[1] - int_tps[0]
            integration_times = torch.tensor([itp - timescale, itp])
            # integration_times = torch.tensor([np.linspace(itp - args.time_scale, itp, ntimes)])
            integration_times = integration_times.type(torch.float32).to(device)

            # transform to previous timepoint
            z, delta_logp = cnf(zs[-1], zero, integration_times=integration_times)
            zs.append(z)
            deltas.append(delta_logp)
        zs = torch.stack(zs, 0)
        zs = zs.cpu().numpy()
        np.save(os.path.join(savedir, "backward_trajectories.npy"), zs)


def main(args):
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    if args.use_cpu:
        device = torch.device("cpu")

    data = dataset.SCData.factory(args.dataset, args)

    args.timepoints = data.get_unique_times()

    # Use maximum timepoint to establish integration_times
    # as some timepoints may be left out for validation etc.
    args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale

    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, data.get_shape()[0], regularization_fns).to(
        device
    )
    if args.use_growth:
        growth_model_path = data.get_growth_net_path()
        # growth_model_path = "/home/atong/TrajectoryNet/data/externel/growth_model_v2.ckpt"
        growth_model = torch.load(growth_model_path, map_location=device)
    if args.spectral_norm:
        add_spectral_norm(model)
    set_cnf_options(args, model)

    state_dict = torch.load(args.save + "/checkpt.pth", map_location=device)
    model.load_state_dict(state_dict["state_dict"])

    # plot_output(device, args, model, data)
    # exit()
    # get_trajectory_samples(device, model, data)

    args.data = data
    args.timepoints = args.data.get_unique_times()
    args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale

    print("integrating backwards")
    # end_time_data = data.data_dict[args.embedding_name]
    end_time_data = data.get_data()[
        args.data.get_times() == np.max(args.data.get_times())
    ]
    # np.random.permutation(end_time_data)
    # rand_idx = np.random.randint(end_time_data.shape[0], size=5000)
    # end_time_data = end_time_data[rand_idx,:]
    integrate_backwards(end_time_data, model, args.save, ntimes=100, device=device)
    exit()
    losses_list = []
    # for factor in np.linspace(0.05, 0.95, 19):
    # for factor in np.linspace(0.91, 0.99, 9):
    if args.dataset == "CHAFFER":  # Do timepoint adjustment
        print("adjusting_timepoints")
        lt = args.leaveout_timepoint
        if lt == 1:
            factor = 0.6799872494335812
            factor = 0.95
        elif lt == 2:
            factor = 0.2905983814032348
            factor = 0.01
        else:
            raise RuntimeError("Unknown timepoint %d" % args.leaveout_timepoint)
        args.int_tps[lt] = (1 - factor) * args.int_tps[lt - 1] + factor * args.int_tps[
            lt + 1
        ]
    losses = eval_utils.evaluate_kantorovich_v2(device, args, model)
    losses_list.append(losses)
    print(np.array(losses_list))
    np.save(os.path.join(args.save, "emd_list"), np.array(losses_list))
    # zs = np.load(os.path.join(args.save, 'backward_trajectories'))
    # losses = eval_utils.evaluate_mse(device, args, model)
    # losses = eval_utils.evaluate_kantorovich(device, args, model)
    # print(losses)
    # eval_utils.generate_samples(device, args, model, growth_model, timepoint=args.timepoints[-1])
    # eval_utils.calculate_path_length(device, args, model, data, args.int_tps[-1])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
