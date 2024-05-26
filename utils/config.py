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
    argparser.add_argument('--run',  type=str, help='Run mode')
    argparser.add_argument('--experiment',  type=str, help='Experiment type')
    argparser.add_argument('--batch_size',  type=int, help='Batch size')
    argparser.add_argument('--epochs',  type=int, help='Number of epochs')
    argparser.add_argument('--lr',  type=float, help='Learning rate')
    argparser.add_argument('--n',  type=int, help='Sample size')
    argparser.add_argument('--k',  type=int, help='Number of iterations')
    argparser.add_argument('--eps',  type=float, help='Epsilon value for regularisation')
    argparser.add_argument('--cost',  type=str, choices=['quad', 'quad_gw', 'ip_gw'], help='Cost function')
    argparser.add_argument('--alg',  type=str, choices=['ne_mot', 'sinkhorn_mot', 'ne_gw', 'sinkhorn_gw'],
                        help='Algorithm')
    argparser.add_argument('--hidden_dim',  type=int, help='Dimension of hidden layers')
    argparser.add_argument('--mod',  type=str, choices=['mot', 'mgw'], help='Model type')
    argparser.add_argument('--seed',  type=int, help='Random seed')
    argparser.add_argument('--data_dist',  type=str, help='Data distribution type')
    argparser.add_argument('--dims',  nargs='+', type=int, help='Dimensions of the data')
    argparser.add_argument('--dim', type=int, help='Dimensions of the data')
    argparser.add_argument('--device',  type=str, help='Device to use')
    argparser.add_argument('--cuda_visible',  type=int, help='CUDA visible device')
    argparser.add_argument('--using_wandb',  type=int, help='Use Weights & Biases logging')
    argparser.add_argument('--cost_graph',  type=str, choices=['full', 'circle', 'tree'],
                        help='Graphical structure of the cost function')
    argparser.add_argument('--schedule_gamma',  type=float, help='scheduler multiplier')
    argparser.add_argument('--schedule',  type=int, help='scheduling flag')
    argparser.add_argument('--schedule_step',  type=int, help='number of epochs between each scheduler step')
    argparser.add_argument('--max_grad_norm',  type=float, help='gradient norm calipping value')
    argparser.add_argument('--clip_grads',  type=int, help='gradient clipping flag')





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
        'experiment': 'synthetic_mot',
        'batch_size': 64,
        'epochs': 150,
        'lr': 5e-4,
        'n': 5000,
        'k': 3,
        'eps': 0.05,
        'cost': 'quad',  # options - quad, quad_gw, ip_gw
        'alg': 'ne_mot',  # options - ne_mot, sinkhorn_mot,ne_gw, sinkhorn_gw
        'hidden_dim': 32,
        'mod': 'mot',  # options - mot, mgw
        'seed': 42,
        'data_dist': 'uniform',
        # 'dims': [1,1,1,1,1,1,1,1],
        # 'dims': [100,100,100,100,100,100,100,100],
        'dim': 15,
        'device': 'gpu',
        'cuda_visible': 3,
        'using_wandb': 0,
        'cost_graph': 'full',  # The cost function graphical structure for decomposition. Options - full, circle(, tree),


        "wandb_entity": "dortsur",

        "schedule": 0,
        "schedule_step": 2,
        "schedule_gamma": 0.5,

        "clip_grads": 1,
        "max_grad_norm": 1.0,


        # GW params:
        'dims': [1,2,3],
        'gw_ns': [5000,5000,5000],
        'gw_same_n': 1,
        'gw_use_convex_eps': 1
    }
    # TD: ADJUST DIMS TO K
    config['batch_size'] = min(config['batch_size'], config['n'])

    # Turn into Bunch object
    config = Config(config)

    # Add args values to the config attributes
    for key in sorted(vars(args)):
        val = getattr(args, key)
        if val is not None:
            setattr(config, key, val)


    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    config.figDir = f"results/{config.run}/{config.alg}/k_{config.k}/n_{config.n}/eps_{config.eps}/_stamp_{now_str}"
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