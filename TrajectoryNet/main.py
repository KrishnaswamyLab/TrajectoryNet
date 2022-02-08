""" main.py

Learns ODE from scrna data

"""
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from TrajectoryNet.lib.growth_net import GrowthNet
from TrajectoryNet.lib import utils
from TrajectoryNet.lib.visualize_flow import visualize_transform
from TrajectoryNet.lib.viz_scrna import (
    save_trajectory,
    trajectory_to_video,
    save_vectors,
)
from TrajectoryNet.lib.viz_scrna import save_trajectory_density


# from train_misc import standard_normal_logprob
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

from TrajectoryNet import dataset
from TrajectoryNet.parse import parser

matplotlib.use("Agg")


def get_transforms(device, args, model, integration_times):
    """
    Given a list of integration points,
    returns a function giving integration times
    """

    def sample_fn(z, logpz=None):
        int_list = [
            torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
            for it in integration_times
        ]
        if logpz is not None:
            # TODO this works right?
            for it in int_list:
                z, logpz = model(z, logpz, integration_times=it, reverse=True)
            return z, logpz
        else:
            for it in int_list:
                z = model(z, integration_times=it, reverse=True)
            return z

    def density_fn(x, logpx=None):
        int_list = [
            torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
            for it in integration_times[::-1]
        ]
        if logpx is not None:
            for it in int_list:
                x, logpx = model(x, logpx, integration_times=it, reverse=False)
            return x, logpx
        else:
            for it in int_list:
                x = model(x, integration_times=it, reverse=False)
            return x

    return sample_fn, density_fn


def compute_loss(device, args, model, growth_model, logger, full_data):
    """
    Compute loss by integrating backwards from the last time step
    At each time step integrate back one time step, and concatenate that
    to samples of the empirical distribution at that previous timestep
    repeating over and over to calculate the likelihood of samples in
    later timepoints iteratively, making sure that the ODE is evaluated
    at every time step to calculate those later points.

    The growth model is a single model of time independent cell growth /
    death rate defined as a variation from uniform.
    """

    # Backward pass accumulating losses, previous state and deltas
    deltas = []
    zs = []
    z = None
    interp_loss = 0.0
    for i, (itp, tp) in enumerate(zip(args.int_tps[::-1], args.timepoints[::-1])):
        # tp counts down from last
        integration_times = torch.tensor([itp - args.time_scale, itp])
        integration_times = integration_times.type(torch.float32).to(device)
        # integration_times.requires_grad = True

        # load data and add noise
        idx = args.data.sample_index(args.batch_size, tp)
        x = args.data.get_data()[idx]
        if args.training_noise > 0.0:
            x += np.random.randn(*x.shape) * args.training_noise
        x = torch.from_numpy(x).type(torch.float32).to(device)

        if i > 0:
            x = torch.cat((z, x))
            zs.append(z)
        zero = torch.zeros(x.shape[0], 1).to(x)

        # transform to previous timepoint
        z, delta_logp = model(x, zero, integration_times=integration_times)
        deltas.append(delta_logp)

        # Straightline regularization
        # Integrate to random point at time t and assert close to (1 - t) * end + t * start
        if args.interp_reg:
            t = np.random.rand()
            int_t = torch.tensor([itp - t * args.time_scale, itp])
            int_t = int_t.type(torch.float32).to(device)
            int_x = model(x, integration_times=int_t)
            int_x = int_x.detach()
            actual_int_x = x * (1 - t) + z * t
            interp_loss += F.mse_loss(int_x, actual_int_x)
    if args.interp_reg:
        print("interp_loss", interp_loss)

    logpz = args.data.base_density()(z)

    # build growth rates
    if args.use_growth:
        growthrates = [torch.ones_like(logpz)]
        for z_state, tp in zip(zs[::-1], args.timepoints[:-1]):
            # Full state includes time parameter to growth_model
            time_state = tp * torch.ones(z_state.shape[0], 1).to(z_state)
            full_state = torch.cat([z_state, time_state], 1)
            growthrates.append(growth_model(full_state))

    # Accumulate losses
    losses = []
    logps = [logpz]
    for i, delta_logp in enumerate(deltas[::-1]):
        logpx = logps[-1] - delta_logp
        if args.use_growth:
            logpx += torch.log(torch.clamp(growthrates[i], 1e-4, 1e4))
        logps.append(logpx[: -args.batch_size])
        losses.append(-torch.mean(logpx[-args.batch_size :]))
    losses = torch.stack(losses)
    weights = torch.ones_like(losses).to(logpx)
    if args.leaveout_timepoint >= 0:
        weights[args.leaveout_timepoint] = 0
    losses = torch.mean(losses * weights)

    # Direction regularization
    if args.vecint:
        similarity_loss = 0
        for i, (itp, tp) in enumerate(zip(args.int_tps, args.timepoints)):
            itp = torch.tensor(itp).type(torch.float32).to(device)
            idx = args.data.sample_index(args.batch_size, tp)
            x = args.data.get_data()[idx]
            v = args.data.get_velocity()[idx]
            x = torch.from_numpy(x).type(torch.float32).to(device)
            v = torch.from_numpy(v).type(torch.float32).to(device)
            x += torch.randn_like(x) * 0.1
            # Only penalizes at the time / place of visible samples
            direction = -model.chain[0].odefunc.odefunc.diffeq(itp, x)
            if args.use_magnitude:
                similarity_loss += torch.mean(F.mse_loss(direction, v))
            else:
                similarity_loss -= torch.mean(F.cosine_similarity(direction, v))
        logger.info(similarity_loss)
        losses += similarity_loss * args.vecint

    # Density regularization
    if args.top_k_reg > 0:
        density_loss = 0
        tp_z_map = dict(zip(args.timepoints[:-1], zs[::-1]))
        if args.leaveout_timepoint not in tp_z_map:
            idx = args.data.sample_index(args.batch_size, tp)
            x = args.data.get_data()[idx]
            if args.training_noise > 0.0:
                x += np.random.randn(*x.shape) * args.training_noise
            x = torch.from_numpy(x).type(torch.float32).to(device)
            t = np.random.rand()
            int_t = torch.tensor([itp - t * args.time_scale, itp])
            int_t = int_t.type(torch.float32).to(device)
            int_x = model(x, integration_times=int_t)
            samples_05 = int_x
        else:
            # If we are leaving out a timepoint the regularize there
            samples_05 = tp_z_map[args.leaveout_timepoint]

        # Calculate distance to 5 closest neighbors
        # WARNING: This currently fails in the backward pass with cuda on pytorch < 1.4.0
        #          works on CPU. Fixed in pytorch 1.5.0
        # RuntimeError: CUDA error: invalid configuration argument
        # The workaround is to run on cpu on pytorch <= 1.4.0 or upgrade
        cdist = torch.cdist(samples_05, full_data)
        values, _ = torch.topk(cdist, 5, dim=1, largest=False, sorted=False)
        # Hinge loss
        hinge_value = 0.1
        values -= hinge_value
        values[values < 0] = 0
        density_loss = torch.mean(values)
        print("Density Loss", density_loss.item())
        losses += density_loss * args.top_k_reg
    losses += interp_loss
    return losses


def train(
    device, args, model, growth_model, regularization_coeffs, regularization_fns, logger
):
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    nfef_meter = utils.RunningAverageMeter(0.93)
    nfeb_meter = utils.RunningAverageMeter(0.93)
    tt_meter = utils.RunningAverageMeter(0.93)

    full_data = (
        torch.from_numpy(
            args.data.get_data()[args.data.get_times() != args.leaveout_timepoint]
        )
        .type(torch.float32)
        .to(device)
    )

    best_loss = float("inf")
    if args.use_growth:
        growth_model.eval()
    end = time.time()
    for itr in range(1, args.niters + 1):
        model.train()
        optimizer.zero_grad()

        # Train
        if args.spectral_norm:
            spectral_norm_power_iteration(model, 1)

        loss = compute_loss(device, args, model, growth_model, logger, full_data)
        loss_meter.update(loss.item())

        if len(regularization_coeffs) > 0:
            # Only regularize on the last timepoint
            reg_states = get_regularization(model, regularization_coeffs)
            reg_loss = sum(
                reg_state * coeff
                for reg_state, coeff in zip(reg_states, regularization_coeffs)
                if coeff != 0
            )
            loss = loss + reg_loss
        total_time = count_total_time(model)
        nfe_forward = count_nfe(model)

        loss.backward()
        optimizer.step()

        # Eval
        nfe_total = count_nfe(model)
        nfe_backward = nfe_total - nfe_forward
        nfef_meter.update(nfe_forward)
        nfeb_meter.update(nfe_backward)
        time_meter.update(time.time() - end)
        tt_meter.update(total_time)

        log_message = (
            "Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) |"
            " NFE Forward {:.0f}({:.1f})"
            " | NFE Backward {:.0f}({:.1f})".format(
                itr,
                time_meter.val,
                time_meter.avg,
                loss_meter.val,
                loss_meter.avg,
                nfef_meter.val,
                nfef_meter.avg,
                nfeb_meter.val,
                nfeb_meter.avg,
            )
        )
        if len(regularization_coeffs) > 0:
            log_message = append_regularization_to_log(
                log_message, regularization_fns, reg_states
            )
        logger.info(log_message)

        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                train_eval(
                    device, args, model, growth_model, itr, best_loss, logger, full_data
                )

        if itr % args.viz_freq == 0:
            if args.data.get_shape()[0] > 2:
                logger.warning("Skipping vis as data dimension is >2")
            else:
                with torch.no_grad():
                    visualize(device, args, model, itr)
        if itr % args.save_freq == 0:
            chkpt = {
                "state_dict": model.state_dict(),
            }
            if args.use_growth:
                chkpt.update({"growth_state_dict": growth_model.state_dict()})
            utils.save_checkpoint(
                chkpt,
                args.save,
                epoch=itr,
            )
        end = time.time()
    logger.info("Training has finished.")


def train_eval(device, args, model, growth_model, itr, best_loss, logger, full_data):
    model.eval()
    test_loss = compute_loss(device, args, model, growth_model, logger, full_data)
    test_nfe = count_nfe(model)
    log_message = "[TEST] Iter {:04d} | Test Loss {:.6f} |" " NFE {:.0f}".format(
        itr, test_loss, test_nfe
    )
    logger.info(log_message)
    utils.makedirs(args.save)
    with open(os.path.join(args.save, "train_eval.csv"), "a") as f:
        import csv

        writer = csv.writer(f)
        writer.writerow((itr, test_loss))

    if test_loss.item() < best_loss:
        best_loss = test_loss.item()
        chkpt = {
            "state_dict": model.state_dict(),
        }
        if args.use_growth:
            chkpt.update({"growth_state_dict": growth_model.state_dict()})
        torch.save(
            chkpt,
            os.path.join(args.save, "checkpt.pth"),
        )


def visualize(device, args, model, itr):
    model.eval()
    for i, tp in enumerate(args.timepoints):
        idx = args.data.sample_index(args.viz_batch_size, tp)
        p_samples = args.data.get_data()[idx]
        sample_fn, density_fn = get_transforms(
            device, args, model, args.int_tps[: i + 1]
        )
        plt.figure(figsize=(9, 3))
        visualize_transform(
            p_samples,
            args.data.base_sample(),
            args.data.base_density(),
            transform=sample_fn,
            inverse_transform=density_fn,
            samples=True,
            npts=100,
            device=device,
        )
        fig_filename = os.path.join(
            args.save, "figs", "{:04d}_{:01d}.jpg".format(itr, i)
        )
        utils.makedirs(os.path.dirname(fig_filename))
        plt.savefig(fig_filename)
        plt.close()


def plot_output(device, args, model):
    save_traj_dir = os.path.join(args.save, "trajectory")
    # logger.info('Plotting trajectory to {}'.format(save_traj_dir))
    data_samples = args.data.get_data()[args.data.sample_index(2000, 0)]
    np.random.seed(42)
    start_points = args.data.base_sample()(1000, 2)
    # idx = args.data.sample_index(50, 0)
    # start_points = args.data.get_data()[idx]
    # start_points = torch.from_numpy(start_points).type(torch.float32)
    save_vectors(
        args.data.base_density(),
        model,
        start_points,
        args.data.get_data(),
        args.data.get_times(),
        args.save,
        skip_first=(not args.data.known_base_density()),
        device=device,
        end_times=args.int_tps,
        ntimes=100,
    )
    save_trajectory(
        args.data.base_density(),
        args.data.base_sample(),
        model,
        data_samples,
        save_traj_dir,
        device=device,
        end_times=args.int_tps,
        ntimes=25,
    )
    trajectory_to_video(save_traj_dir)

    density_dir = os.path.join(args.save, "density2")
    save_trajectory_density(
        args.data.base_density(),
        model,
        data_samples,
        density_dir,
        device=device,
        end_times=args.int_tps,
        ntimes=25,
        memory=0.1,
    )
    trajectory_to_video(density_dir)


def main(args):
    # logger
    print(args.no_display_loss)
    utils.makedirs(args.save)
    logger = utils.get_logger(
        logpath=os.path.join(args.save, "logs"),
        filepath=os.path.abspath(__file__),
        displaying=~args.no_display_loss,
    )

    if args.layer_type == "blend":
        logger.info("!! Setting time_scale from None to 1.0 for Blend layers.")
        args.time_scale = 1.0

    logger.info(args)

    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    if args.use_cpu:
        device = torch.device("cpu")

    args.data = dataset.SCData.factory(args.dataset, args)

    args.timepoints = args.data.get_unique_times()
    # Use maximum timepoint to establish integration_times
    # as some timepoints may be left out for validation etc.
    args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale

    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, args.data.get_shape()[0], regularization_fns).to(
        device
    )
    growth_model = None
    if args.use_growth:
        if args.leaveout_timepoint == -1:
            growth_model_path = "../data/externel/growth_model_v2.ckpt"
        elif args.leaveout_timepoint in [1, 2, 3]:
            assert args.max_dim == 5
            growth_model_path = "../data/growth/model_%d" % args.leaveout_timepoint
        else:
            print("WARNING: Cannot use growth with this timepoint")

        growth_model = torch.load(growth_model_path, map_location=device)
    if args.spectral_norm:
        add_spectral_norm(model)
    set_cnf_options(args, model)

    if args.test:
        state_dict = torch.load(args.save + "/checkpt.pth", map_location=device)
        model.load_state_dict(state_dict["state_dict"])
        # if "growth_state_dict" not in state_dict:
        #    print("error growth model note in save")
        #    growth_model = None
        # else:
        #    checkpt = torch.load(args.save + "/checkpt.pth", map_location=device)
        #    growth_model.load_state_dict(checkpt["growth_state_dict"])
        # TODO can we load the arguments from the save?
        # eval_utils.generate_samples(
        #    device, args, model, growth_model, timepoint=args.leaveout_timepoint
        # )
        # with torch.no_grad():
        #    evaluate(device, args, model, growth_model)
    #    exit()
    else:
        logger.info(model)
        n_param = count_parameters(model)
        logger.info("Number of trainable parameters: {}".format(n_param))

        train(
            device,
            args,
            model,
            growth_model,
            regularization_coeffs,
            regularization_fns,
            logger,
        )

    if args.data.data.shape[1] == 2:
        plot_output(device, args, model)


if __name__ == "__main__":

    args = parser.parse_args()
    main(args)
