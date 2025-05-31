# from attrdict import AttrDict
from box import Box
import logging
from collections import OrderedDict
import argparse
import wandb
import datetime
import os

logger = logging.getLogger("logger")


def GetArgs():
    """" collects command line arguments """

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', type=str, help='configuration file')
    argparser.add_argument('--wandb_project_name',  type=str, help='wandb project name')
    argparser.add_argument('--run',  type=str, help='Run mode - for directory organization')
    argparser.add_argument('--experiment',  type=str, choices=['mnist'], help='Experiment type')
    argparser.add_argument('--batch_size',  type=int, help='Batch size')
    argparser.add_argument('--epochs',  type=int, help='Number of epochs')
    argparser.add_argument('--lr',  type=float, help='Learning rate')
    argparser.add_argument('--n',  type=int, help='Sample size')
    argparser.add_argument('--k',  type=int, help='Number of iterations')
    argparser.add_argument('--eps',  type=float, help='Epsilon value for regularisation')
    argparser.add_argument('--cost',  type=str, choices=['quad', 'quad_gw', 'ip_gw'], help='Cost function')
    argparser.add_argument('--alg',  type=str, choices=['ne_mot', 'sinkhorn_mot', 'ne_mgw', 'sinkhorn_gw'],
                        help='Algorithm')
    argparser.add_argument('--hidden_dim',  type=int, help='Dimension of hidden layers')
    argparser.add_argument('--mod',  type=str, choices=['mot', 'mgw'], help='Model type')
    argparser.add_argument('--seed',  type=int, help='Random seed')
    argparser.add_argument('--data_dist',  type=str, choices=['uniform', 'gauss', 'gmm'], help='Data distribution type')
    argparser.add_argument('--dims',  nargs='+', type=int, help='Dimensions of the data')
    argparser.add_argument('--dim', type=int, help='Dimensions of the data')
    argparser.add_argument('--device',  type=str, choices=['gpu', 'cpu'], help='Device to use')
    argparser.add_argument('--cuda_visible',  type=int, help='CUDA visible device')
    argparser.add_argument('--using_wandb',  type=int, help='Use Weights & Biases logging')
    argparser.add_argument('--cost_graph',  type=str, choices=['full', 'circle', 'tree'],
                        help='Graphical structure of the cost function')
    argparser.add_argument('--schedule_gamma',  type=float, help='scheduler multiplier')
    argparser.add_argument('--schedule',  type=int, help='scheduling flag')
    argparser.add_argument('--schedule_step',  type=int, help='number of epochs between each scheduler step')
    argparser.add_argument('--max_grad_norm',  type=float, help='gradient norm calipping value')
    argparser.add_argument('--clip_grads',  type=int, help='gradient clipping flag')
    argparser.add_argument('--check_P_sum',  type=int, help='check P sum flag for debug purposes')
    argparser.add_argument('--regularize_pariwise_coupling',  type=int, help='pairwise coupling regularization flag')
    argparser.add_argument('--regularize_pariwise_coupling_reg',  type=float, help='pairwise coupling regularization coefficient')
    argparser.add_argument('--euler',  type=int, help='euler flows case flag')
    argparser.add_argument('--calc_ot_cost',  type=int, help='a flag to skip the ot cost claculation')
    argparser.add_argument('--cost_implement',  type=str, choices=['simplified'], help='cost implementation type')
    argparser.add_argument('--gauss_std',  type=float, help='gaussian standard deviation')
    argparser.add_argument('--dataset',  type=str, choices=['mnist', 'synthetic', 'cifar'], help='dataset choice')
    argparser.add_argument('--norm_by_k',  type=str, choices=['0', '1'], help='cost normalization by k')
    argparser.add_argument(
    "--mnist_marginals",
    type=lambda s: [int(ch) for ch in s],   # "123" ➜ [1, 2, 3]
    default=[],                              # sensible default
    help="specific mnist marginals choice"
    )





    argparser.set_defaults(quiet=False)

    args = argparser.parse_args()
    return args


def PreprocessMeta():
    """
    steps:
    0. get config
    1. parse args
    2. initiate wandb
    """
    args = GetArgs()
    config = GetConfig(args)

    # add wandb:
    if config.using_wandb:
        wandb_proj = "mot" if not hasattr(config, 'wandb_project_name') else config.wandb_project_name
        wandb.init(project=wandb_proj,
                   entity=config.wandb_entity,
                   config=config)
    return config


def GetConfig(args):
    config = {
        'run': 'debug',
        'experiment': 'mnist',
        'batch_size': 32,
        'epochs': 100,
        'lr': 1e-5,
        'n': 2500,
        'k': 3,
        'eps': 1,
        'cost': 'quad',  # options - quad, quad_gw, ip_gw
        'alg': 'ne_mot',  # options - ne_mot, sinkhorn_mot,ne_mgw, sinkhorn_gw
        'hidden_dim': 32,
        'mod': 'mot',  # options - mot, mgw
        'seed': 1,
        'data_dist': 'gmm',   # options - uniform, gauss, gmm
        'dataset': 'synthetic',  #options - mnist, synthetic
        'gauss_std': 1,
        # 'dims': [1,1,1,1,1,1,1,1],
        # 'dims': [100,100,100,100,100,100,100,100],
        'dim': 100,
        'device': 'gpu',
        'cuda_visible': 3,
        'using_wandb': 0,
        'cost_graph': 'full',  # The cost function graphical structure for decomposition. Options-full,circle,tree
        'cost_implement': 'simplified',
        'enc_dim': 784,
        'norm_by_k': 1,


        "wandb_entity": "dortsur",

        "schedule": 1,
        "schedule_step": 5,
        "schedule_gamma": 0.5,


        "clip_grads": 1,
        "max_grad_norm": 0.001,

        "check_P_sum": 0,

        "euler": 0,

        "regularize_pariwise_coupling": 0,
        "regularize_pariwise_coupling_reg": 10.0,

        "normalize_plan": 0,

        "calc_ot_cost": 1,

        # tree params:
        "tree_ord": 'pre',      # traversal method for tree creation
        "tree_type": 'bin',     # tree creation setting, either bin (binary), star or custom (from a given string)


        # GW params:
        # 'dims': [1, 2, 3],
        # 'gw_ns': [5000, 5000, 5000],
        'gw_same_n': 1,
        'gw_use_convex_eps': 1,
        'A_mgw_opt': 'autograd',

        'save_results': True
    }
    # TD: ADJUST DIMS TO K
    config['batch_size'] = min(config['batch_size'], config['n'])

    # if len(config['dims']) != config['k']:
    #     config['dims'] = list(range(1, config['k']+1))

    # Turn into Bunch object
    config = Config(config)

    # Add args values to the config attributes
    for key in sorted(vars(args)):
        val = getattr(args, key)
        if val is not None:
            setattr(config, key, val)


    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    config.figDir = f"results/{config.run}/{config.alg}/n_{config.n}/eps_{config.eps}/k_{config.k}/d_{config.dim}/_stamp_{now_str}"
    if config.dataset == 'mnist':
        config.figDir = f"results/{config.run}/{config.dataset}/{config.alg}/n_{config.n}/eps_{config.eps}/k_{config.k}/_stamp_{now_str}"
    if config.mnist_marginals != []:
        config.figDir = f"results/{config.run}/{config.dataset}/{config.alg}/n_{config.n}/eps_{config.eps}/k_{config.k}/maginals_{config.mnist_marginals}/_stamp_{now_str}"
    os.makedirs(config.figDir, exist_ok=True)

    config.print()
    return config


class Config(Box):
    """ class for handling dictionary as class attributes """
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
    def print(self):
        line_width = 132
        line = "-" * line_width
        logger.info(line + "\n" +
              "| {:^35s} | {:^90} |\n".format('Feature', 'Value') +
              "=" * line_width)
        for key, val in sorted(self.items(), key= lambda x: x[0]):
            if isinstance(val, OrderedDict):
                raise NotImplementedError("Nested configs are not implemented")
            else:
                logger.info("| {:35s} | {:90} |\n".format(key, str(val)) + line)
        logger.info("\n")