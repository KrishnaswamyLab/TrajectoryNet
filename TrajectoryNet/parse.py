import argparse
from .lib.layers import odefunc

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams"]

parser = argparse.ArgumentParser("Continuous Normalizing Flow")
parser.add_argument("--test", action="store_true")
parser.add_argument("--dataset", type=str, default="EB")
parser.add_argument("--use_growth", action="store_true")
parser.add_argument("--use_density", action="store_true")
parser.add_argument("--leaveout_timepoint", type=int, default=-1)
parser.add_argument(
    "--layer_type",
    type=str,
    default="concatsquash",
    choices=[
        "ignore",
        "concat",
        "concat_v2",
        "squash",
        "concatsquash",
        "concatcoord",
        "hyper",
        "blend",
    ],
)
parser.add_argument("--max_dim", type=int, default=10)
parser.add_argument("--dims", type=str, default="64-64-64")
parser.add_argument("--num_blocks", type=int, default=1, help="Number of stacked CNFs.")
parser.add_argument("--time_scale", type=float, default=0.5)
parser.add_argument("--train_T", type=eval, default=True)
parser.add_argument(
    "--divergence_fn",
    type=str,
    default="brute_force",
    choices=["brute_force", "approximate"],
)
parser.add_argument(
    "--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES
)
parser.add_argument("--stochastic", action="store_true")

parser.add_argument(
    "--alpha", type=float, default=0.0, help="loss weight parameter for growth model"
)
parser.add_argument("--solver", type=str, default="dopri5", choices=SOLVERS)
parser.add_argument("--atol", type=float, default=1e-5)
parser.add_argument("--rtol", type=float, default=1e-5)
parser.add_argument(
    "--step_size", type=float, default=None, help="Optional fixed step size."
)

parser.add_argument("--test_solver", type=str, default=None, choices=SOLVERS + [None])
parser.add_argument("--test_atol", type=float, default=None)
parser.add_argument("--test_rtol", type=float, default=None)

parser.add_argument("--residual", action="store_true")
parser.add_argument("--rademacher", action="store_true")
parser.add_argument("--spectral_norm", action="store_true")
parser.add_argument("--batch_norm", action="store_true")
parser.add_argument("--bn_lag", type=float, default=0)

parser.add_argument("--niters", type=int, default=10000)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--test_batch_size", type=int, default=1000)
parser.add_argument("--viz_batch_size", type=int, default=2000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-5)

# Track quantities
parser.add_argument("--l1int", type=float, default=None, help="int_t ||f||_1")
parser.add_argument("--l2int", type=float, default=None, help="int_t ||f||_2")
parser.add_argument("--sl2int", type=float, default=None, help="int_t ||f||_2^2")
parser.add_argument(
    "--dl2int", type=float, default=None, help="int_t ||f^T df/dt||_2"
)  # f df/dx?
parser.add_argument(
    "--dtl2int", type=float, default=None, help="int_t ||f^T df/dx + df/dt||_2"
)
parser.add_argument("--JFrobint", type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument(
    "--JdiagFrobint", type=float, default=None, help="int_t ||df_i/dx_i||_F"
)
parser.add_argument(
    "--JoffdiagFrobint", type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F"
)
parser.add_argument("--vecint", type=float, default=None, help="regularize direction")
parser.add_argument(
    "--use_magnitude",
    action="store_true",
    help="regularize direction using MSE loss instead of cosine loss",
)

parser.add_argument(
    "--interp_reg", type=float, default=None, help="regularize interpolation"
)

parser.add_argument("--save", type=str, default="../results/tmp")
parser.add_argument("--save_freq", type=int, default=1000)
parser.add_argument("--viz_freq", type=int, default=100)
parser.add_argument("--viz_freq_growth", type=int, default=100)
parser.add_argument("--val_freq", type=int, default=100)
parser.add_argument("--log_freq", type=int, default=10)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--no_display_loss", action="store_false")
parser.add_argument(
    "--top_k_reg", type=float, default=0.0, help="density following regularization"
)
parser.add_argument("--training_noise", type=float, default=0.1)
parser.add_argument(
    "--embedding_name",
    type=str,
    default="pca",
    help="choose embedding name to perform TrajectoryNet on",
)
parser.add_argument("--whiten", action="store_true", help="Whiten data before running TrajectoryNet")
