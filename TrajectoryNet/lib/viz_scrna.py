import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


# def standard_normal_logprob(z):
#     logZ = -0.5 * math.log(2 * math.pi)
#     return torch.sum(logZ - z.pow(2) / 2, 1, keepdim=True)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def save_2d_trajectory_v2(prior_logdensity, prior_sampler, model, data_samples, savedir, ntimes=5, end_times=None, memory=0.01, device='cpu', limit=4):
    """ Save the trajectory as a series of photos such that we can easily display on paper / poster """
    model.eval()

    #  Sample from prior
    z_samples = prior_sampler(2000, 2).to(device)

    with torch.no_grad():
        # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = prior_logdensity(z_samples)
        t = 0
        for cnf in model.chain:

            # Construct integration_list
            if end_times is None:
                end_times = [(cnf.sqrt_end_time * cnf.sqrt_end_time)]
            integration_list = [torch.linspace(0, end_times[0], ntimes).to(device)]
            for i, et in enumerate(end_times[1:]):
                integration_list.append(torch.linspace(end_times[i], et, ntimes).to(device))
            full_times = torch.cat(integration_list, 0)
            print('integration_list', integration_list)
            

            # Integrate over evenly spaced samples
            z_traj, logpz = cnf(z_samples, logp_samples, integration_times=integration_list[0], reverse=True)
            full_traj = [(z_traj, logpz)]
            for i, int_times in enumerate(integration_list[1:]):
                prev_z, prev_logp = full_traj[-1]
                z_traj, logpz = cnf(prev_z[-1], prev_logp[-1], integration_times=int_times, reverse=True)
                full_traj.append((z_traj[1:], logpz[1:]))
            full_zip = list(zip(*full_traj))
            z_traj = torch.cat(full_zip[0], 0)
            #z_logp = torch.cat(full_zip[1], 0)
            z_traj = z_traj.cpu().numpy()

            width = z_traj.shape[0]
            plt.figure(figsize=(8, 8))
            fig, axes = plt.subplots(1, width, figsize=(4*width, 4), sharex=True, sharey=True)
            axes = axes.flatten()
            for w in range(width):
                # plot the density
                ax = axes[w]
                K = 13j
                y, x = np.mgrid[-0.5:2.5:K, -1.5:1.5:K]
                #y, x = np.mgrid[-limit:limit:K, -limit:limit:K]
                K = int(K.imag)
                zs = torch.from_numpy(np.stack([x, y], -1).reshape(K * K, 2)).to(device, torch.float32)
                logps = torch.zeros(zs.shape[0], 1).to(device, torch.float32)
                dydt = cnf.odefunc(full_times[t], (zs, logps))[0]
                dydt = -dydt.cpu().detach().numpy()
                dydt = dydt.reshape(K, K, 2)

                logmag = 2 * np.log(np.hypot(dydt[:, :, 0], dydt[:, :, 1]))
                ax.quiver(
                    #x, y, dydt[:, :, 0], -dydt[:, :, 1],
                     x, y, dydt[:, :, 0], dydt[:, :, 1],
                    np.exp(logmag), cmap="coolwarm", scale=20., width=0.015, pivot="mid"
                )
                ax.set_xlim(-limit, limit)
                ax.set_ylim(limit, -limit)
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-0.5, 2.5)
                ax.axis("off")

                ax.scatter(z_traj[w,:,0], z_traj[w,:,1], c='k', s=0.5)
                #ax.set_title("Vector Field", fontsize=32)
                t += 1

            makedirs(savedir)
            plt.tight_layout(pad=0.0)
            plt.savefig(os.path.join(savedir, "vector_plot.jpg"))
            plt.close
def save_2d_trajectory(prior_logdensity, prior_sampler, model, data_samples, savedir, ntimes=5, end_times=None, memory=0.01, device='cpu'):
    """ Save the trajectory as a series of photos such that we can easily display on paper / poster """
    model.eval()

    #  Sample from prior
    z_samples = prior_sampler(2000, 2).to(device)

    # sample from a grid
    npts = 100
    limit = 1.5
    side = np.linspace(-limit, limit, npts)
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
                integration_list.append(torch.linspace(end_times[i], et, ntimes).to(device))
            full_times = torch.cat(integration_list, 0)
            print('integration_list', integration_list)
            

            # Integrate over evenly spaced samples
            z_traj, logpz = cnf(z_samples, logp_samples, integration_times=integration_list[0], reverse=True)
            full_traj = [(z_traj, logpz)]
            for i, int_times in enumerate(integration_list[1:]):
                prev_z, prev_logp = full_traj[-1]
                z_traj, logpz = cnf(prev_z[-1], prev_logp[-1], integration_times=int_times, reverse=True)
                full_traj.append((z_traj[1:], logpz[1:]))
            full_zip = list(zip(*full_traj))
            z_traj = torch.cat(full_zip[0], 0)
            #z_logp = torch.cat(full_zip[1], 0)
            z_traj = z_traj.cpu().numpy()

            grid_z_traj, grid_logpz_traj = [], []
            inds = torch.arange(0, z_grid.shape[0]).to(torch.int64)
            for ii in torch.split(inds, int(z_grid.shape[0] * memory)):
                _grid_z_traj, _grid_logpz_traj = cnf(
                    z_grid[ii], logp_grid[ii], integration_times=integration_list[0], reverse=True
                )
                full_traj = [(_grid_z_traj, _grid_logpz_traj)]
                for int_times in integration_list[1:]:
                    prev_z, prev_logp = full_traj[-1]
                    _grid_z_traj, _grid_logpz_traj = cnf(
                        prev_z[-1], prev_logp[-1], integration_times=int_times, reverse=True
                    )
                    full_traj.append((_grid_z_traj, _grid_logpz_traj))
                full_zip = list(zip(*full_traj))
                _grid_z_traj = torch.cat(full_zip[0], 0).cpu().numpy()
                _grid_logpz_traj = torch.cat(full_zip[1], 0).cpu().numpy()
                grid_z_traj.append(_grid_z_traj)
                grid_logpz_traj.append(_grid_logpz_traj)
                
            grid_z_traj = np.concatenate(grid_z_traj, axis=1)
            grid_logpz_traj = np.concatenate(grid_logpz_traj, axis=1)

            width = z_traj.shape[0]
            plt.figure(figsize=(8, 8))
            fig, axes = plt.subplots(2, width, figsize=(4*width, 8), sharex=True, sharey=True)
            axes = axes.flatten()
            for w in range(width):
                # plot the density
                ax = axes[w]

                z, logqz = grid_z_traj[t], grid_logpz_traj[t]

                xx = z[:, 0].reshape(npts, npts)
                yy = z[:, 1].reshape(npts, npts)
                qz = np.exp(logqz).reshape(npts, npts)

                ax.pcolormesh(xx, yy, qz)
                ax.set_xlim(-limit, limit)
                ax.set_ylim(-limit, limit)
                cmap = matplotlib.cm.get_cmap(None)
                ax.set_facecolor(cmap(0.))
                ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                #ax.set_title("Density", fontsize=32)

                # plot vector field
                ax = axes[w+width]

                K = 13j
                y, x = np.mgrid[-limit:limit:K, -limit:limit:K]
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
                ax.set_xlim(-limit, limit)
                ax.set_ylim(limit, -limit)
                ax.axis("off")
                #ax.set_title("Vector Field", fontsize=32)
                t += 1

            makedirs(savedir)
            plt.tight_layout(pad=0.0)
            plt.savefig(os.path.join(savedir, "vector_plot.jpg"))
            plt.close


def save_vectors(prior_logdensity, model, data_samples, full_data, labels, savedir, skip_first=False, ntimes=101, end_times=None, memory=0.01, device='cpu', lim=4):
    model.eval()

    #  Sample from prior
    z_samples = data_samples.to(device)

    with torch.no_grad():
        # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = prior_logdensity(z_samples)
        t = 0
        for cnf in model.chain:
            # Construct integration_list
            if end_times is None:
                end_times = [(cnf.sqrt_end_time * cnf.sqrt_end_time)]
            # integration_list = []
            integration_list = [torch.linspace(0, end_times[0], ntimes).to(device)]

            # Start integration at first end_time
            for i, et in enumerate(end_times[1:]):
                integration_list.append(torch.linspace(end_times[i], et, ntimes).to(device))
            # if len(end_times) == 1:
            #     integration_list = [torch.linspace(0, end_times[0], ntimes).to(device)]
            # print(integration_list)


            # Integrate over evenly spaced samples
            z_traj, logpz = cnf(z_samples, logp_samples, integration_times=integration_list[0], reverse=True)
            full_traj = [(z_traj, logpz)]
            for int_times in integration_list[1:]:
                prev_z, prev_logp = full_traj[-1]
                z_traj, logpz = cnf(prev_z[-1], prev_logp[-1], integration_times=int_times, reverse=True)
                full_traj.append((z_traj, logpz))
            full_zip = list(zip(*full_traj))
            z_traj = torch.cat(full_zip[0], 0)
            z_traj = z_traj.cpu().numpy()

            # mask out stray negative points
            pos_mask = full_data[:,1] >=0
            full_data = full_data[pos_mask]
            labels = labels[pos_mask]
            print(np.unique(labels))

            plt.figure(figsize=(8, 8))
            ax = plt.subplot(aspect="equal")
            ax.scatter(full_data[:,0], full_data[:,1], c=labels.astype(np.int32), cmap='tab10', s=0.5, alpha=1)
            # If we do not have a known base density then skip vectors for the first integration.

            z_traj = np.swapaxes(z_traj, 0, 1)
            if skip_first:
                z_traj = z_traj[:, ntimes:, :]
            ax.scatter(z_traj[:,0,0], z_traj[:,0,1], s=20, c='k')
            for zk in z_traj:
            #for zk in z_traj[:,ntimes:,:]:
                ax.scatter(zk[:,0], zk[:,1], s=1, c = np.linspace(0,1,zk.shape[0]), cmap='Spectral')
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            #ax.set_ylim(4, -4)
            makedirs(savedir)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(savedir, f"vectors.jpg"), dpi=300)
            t += 1

def save_trajectory_density(prior_logdensity, model, data_samples, savedir, ntimes=101, end_times=None, memory=0.01, device='cpu'):
    model.eval()

    # sample from a grid
    #Jnpts = 100
    npts = 800
    side = np.linspace(-4, 4, npts)
    xx, yy = np.meshgrid(side, side)
    xx = torch.from_numpy(xx).type(torch.float32).to(device)
    yy = torch.from_numpy(yy).type(torch.float32).to(device)
    z_grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)

    with torch.no_grad():
        # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_grid = prior_logdensity(z_grid)
        t = 0
        for cnf in model.chain:
            # Construct integration_list
            if end_times is None:
                end_times = [(cnf.sqrt_end_time * cnf.sqrt_end_time)]
            integration_list = [torch.linspace(0, end_times[0], ntimes).to(device)]
            for i, et in enumerate(end_times[1:]):
                integration_list.append(torch.linspace(end_times[i], et, ntimes).to(device))
            full_times = torch.cat(integration_list, 0)

            grid_z_traj, grid_logpz_traj = [], []
            inds = torch.arange(0, z_grid.shape[0]).to(torch.int64)
            for ii in torch.split(inds, int(z_grid.shape[0] * memory)):
                _grid_z_traj, _grid_logpz_traj = cnf(
                    z_grid[ii], logp_grid[ii], integration_times=integration_list[0], reverse=True
                )
                full_traj = [(_grid_z_traj, _grid_logpz_traj)]
                for int_times in integration_list[1:]:
                    prev_z, prev_logp = full_traj[-1]
                    _grid_z_traj, _grid_logpz_traj = cnf(
                        prev_z[-1], prev_logp[-1], integration_times=int_times, reverse=True
                    )
                    full_traj.append((_grid_z_traj, _grid_logpz_traj))
                full_zip = list(zip(*full_traj))
                _grid_z_traj = torch.cat(full_zip[0], 0).cpu().numpy()
                _grid_logpz_traj = torch.cat(full_zip[1], 0).cpu().numpy()
                print(_grid_z_traj.shape)
                grid_z_traj.append(_grid_z_traj)
                grid_logpz_traj.append(_grid_logpz_traj)
                
            grid_z_traj = np.concatenate(grid_z_traj, axis=1)[ntimes:]
            grid_logpz_traj = np.concatenate(grid_logpz_traj, axis=1)[ntimes:]
            

            #plt.figure(figsize=(8, 8))
            #fig, axes = plt.subplots(2,1, gridspec_kw={'height_ratios': [7, 1]}, figsize=(5,7))
            for _ in range(grid_z_traj.shape[0]):
                fig, axes = plt.subplots(2,1, gridspec_kw={'height_ratios': [7, 1]}, figsize=(8,8))
                #plt.clf()
                ax = axes[0]
                # Density
                z, logqz = grid_z_traj[t], grid_logpz_traj[t]

                xx = z[:, 0].reshape(npts, npts)
                yy = z[:, 1].reshape(npts, npts)
                qz = np.exp(logqz).reshape(npts, npts)

                ax.pcolormesh(xx, yy, qz)
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                cmap = matplotlib.cm.get_cmap(None)
                ax.set_facecolor(cmap(0.))
                #ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_title("Density", fontsize=32)

                ax=axes[1]

                # Colorbar
                cb = matplotlib.colorbar.ColorbarBase(ax,
                        #cmap='Spectral', 
                        cmap=plt.cm.Spectral,
                        orientation='horizontal')
                #cb.set_ticks(np.linspace(0,1,4))
                #cb.set_ticklabels(['48HR', 'Day 12', 'Day 18', 'Day 30'])
                #cb.set_ticklabels(['E12.5', 'E14.5', 'E16.0', 'E17.5'])
                #cb.set_ticks(np.linspace(0,1,5))
                #cb.set_ticklabels(np.arange(len(end_times)))
                ax.axvline(t / grid_z_traj.shape[0], c='k', linewidth=15)
                ax.set_title('Time')

                print('making dir: %s' % savedir)
                makedirs(savedir)
                plt.savefig(os.path.join(savedir, f"viz-{t:05d}.jpg"))
                plt.close()
                t += 1


def save_trajectory(prior_logdensity, prior_sampler, model, data_samples, savedir, ntimes=101, end_times=None, memory=0.01, device='cpu'):
    model.eval()

    #  Sample from prior
    z_samples = prior_sampler(2000, 2).to(device)

    # sample from a grid
    npts = 800
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
                integration_list.append(torch.linspace(end_times[i], et, ntimes).to(device))
            full_times = torch.cat(integration_list, 0)
            print(full_times.shape)

            # Integrate over evenly spaced samples
            z_traj, logpz = cnf(z_samples, logp_samples, integration_times=integration_list[0], reverse=True)
            full_traj = [(z_traj, logpz)]
            for int_times in integration_list[1:]:
                prev_z, prev_logp = full_traj[-1]
                z_traj, logpz = cnf(prev_z[-1], prev_logp[-1], integration_times=int_times, reverse=True)
                full_traj.append((z_traj, logpz))
            full_zip = list(zip(*full_traj))
            z_traj = torch.cat(full_zip[0], 0)
            #z_logp = torch.cat(full_zip[1], 0)
            z_traj = z_traj.cpu().numpy()

            grid_z_traj, grid_logpz_traj = [], []
            inds = torch.arange(0, z_grid.shape[0]).to(torch.int64)
            for ii in torch.split(inds, int(z_grid.shape[0] * memory)):
                _grid_z_traj, _grid_logpz_traj = cnf(
                    z_grid[ii], logp_grid[ii], integration_times=integration_list[0], reverse=True
                )
                full_traj = [(_grid_z_traj, _grid_logpz_traj)]
                for int_times in integration_list[1:]:
                    prev_z, prev_logp = full_traj[-1]
                    _grid_z_traj, _grid_logpz_traj = cnf(
                        prev_z[-1], prev_logp[-1], integration_times=int_times, reverse=True
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
                ax = plt.subplot(2, 2, 1, aspect="equal")

                ax.hist2d(data_samples[:, 0], data_samples[:, 1], range=[[-4, 4], [-4, 4]], bins=200)
                ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_title("Target", fontsize=32)

                # plot the density
                ax = plt.subplot(2, 2, 2, aspect="equal")

                z, logqz = grid_z_traj[t], grid_logpz_traj[t]

                xx = z[:, 0].reshape(npts, npts)
                yy = z[:, 1].reshape(npts, npts)
                qz = np.exp(logqz).reshape(npts, npts)

                plt.pcolormesh(xx, yy, qz)
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                cmap = matplotlib.cm.get_cmap(None)
                ax.set_facecolor(cmap(0.))
                ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_title("Density", fontsize=32)

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

                makedirs(savedir)
                plt.savefig(os.path.join(savedir, f"viz-{t:05d}.jpg"))
                t += 1


def trajectory_to_video(savedir):
    import subprocess
    bashCommand = 'ffmpeg -y -i {} {}'.format(os.path.join(savedir, 'viz-%05d.jpg'), os.path.join(savedir, 'traj.mp4'))
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


if __name__ == '__main__':
    import argparse
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

    import lib.toy_data as toy_data
    from train_misc import count_parameters
    from train_misc import set_cnf_options, add_spectral_norm, create_regularization_fns
    from train_misc import build_model_tabular

    def get_ckpt_model_and_data(args):
        # Load checkpoint.
        checkpt = torch.load(args.checkpt, map_location=lambda storage, loc: storage)
        ckpt_args = checkpt['args']
        state_dict = checkpt['state_dict']

        # Construct model and restore checkpoint.
        regularization_fns, regularization_coeffs = create_regularization_fns(ckpt_args)
        model = build_model_tabular(ckpt_args, 2, regularization_fns).to(device)
        if ckpt_args.spectral_norm: add_spectral_norm(model)
        set_cnf_options(ckpt_args, model)

        model.load_state_dict(state_dict)
        model.to(device)

        print(model)
        print("Number of trainable parameters: {}".format(count_parameters(model)))

        # Load samples from dataset
        data_samples = toy_data.inf_train_gen(ckpt_args.data, batch_size=2000)

        return model, data_samples

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpt', type=str, required=True)
    parser.add_argument('--ntimes', type=int, default=101)
    parser.add_argument('--memory', type=float, default=0.01, help='Higher this number, the more memory is consumed.')
    parser.add_argument('--save', type=str, default='trajectory')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, data_samples = get_ckpt_model_and_data(args)
    save_trajectory(model, data_samples, args.save, ntimes=args.ntimes, memory=args.memory, device=device)
    trajectory_to_video(args.save)
