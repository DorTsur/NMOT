import numpy as np
import torch.nn as nn
import torch
import wandb
from torch.utils.data import DataLoader
import torch.nn.functional as F
from timeit import default_timer as timer
from utils.data_fns import QuadCost, QuadCostGW, MultiTensorDataset, rotate, translate, perspective_warp
from utils.tree_fns import create_tree
import pickle
import os
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, value_and_grad
import optax
from functools import partial
from jax.tree_util import tree_map
import pickle
import os
from timeit import default_timer as timer
import random
from jax.nn import relu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import kornia
import pdb
from torchvision import models
from tqdm import tqdm


############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
# Neural Estimation models
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################


class MOT_NE_alg():
    """
    Multi-Marginal Optimal Transport Neural Estimation Algorithm.
    
    This class implements a neural estimation approach for solving multi-marginal optimal transport problems.
    It uses a collection of neural networks to estimate the optimal transport plan between multiple distributions.
    
    Attributes:
        models (list): Collection of neural networks, one for each marginal
        k (int): Number of marginals
        num_epochs (int): Number of training epochs
        batch_size (int): Size of training batches
        eps (float): Regularization parameter for the transport problem
        cost_graph (str): Structure of cost connections ('full', 'circle', 'tree')
        device (torch.device): Device to run computations on
        using_wandb (bool): Whether to use Weights & Biases for logging
    """
    def __init__(self, params, device, d):
        self.models = []
        self.params = params
        if params.dataset == 'mnist':
            self.k = min(10,params.k)
            d = 784
        else:
            self.k = params['k']
        self.num_epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.eps = params['eps']
        
        for i in range(self.k):
            model = Net_EOT(dim=d, K=min(6*d,80), deeper=True, data=self.params.dataset).to(device)
            self.models.append(model)
            
        self.cost = params['cost']
        self.opt = [torch.optim.Adam(list(self.models[i].parameters()), lr=params['lr']) for i in range(self.k)]

        if params.schedule:
            self.scheduler = [torch.optim.lr_scheduler.StepLR(opt, step_size=params.schedule_step, gamma=params.schedule_gamma) for opt in self.opt]

        if self.params.cost_graph == 'tree':
            self.tree_root = create_tree(self.params)
        else:
            self.tree_root = None

        self.device = device
        self.cost_graph = params['cost_graph']
        self.using_wandb = params['using_wandb']


    def train_mot(self, X):
        """
        Train the neural networks to estimate the optimal transport plan.
        
        Args:
            X: Input data tensor of shape (batch_size, feature_dim, k) where k is the number of marginals
            
        The training process:
        1. For each epoch:
            - Process data in batches
            - For each marginal:
                - Calculate potentials (phi) using the neural networks
                - Compute exponential term based on cost structure
                - Update network parameters using gradient descent
        2. Track loss and timing metrics
        """
        if self.params.efficient_dataset:
            num_batches = self.params.n//self.params.batch_size
            x_b = random.sample(range(1, num_batches+1), num_batches)
        else:
            x_b = DataLoader(X, batch_size=self.batch_size, shuffle=True)
        self.tot_loss = []
        self.times = []
        for epoch in range(self.num_epochs):
            l = []
            t0 = timer()
            # for _, data in enumerate(x_b):
            for data in tqdm(x_b):
                self.zero_grad_models()
                # data.to(self.device)
                # print(f'data shape {data.shape}')
                for k_ind in range(self.k):
                    # k-margin loss
                    # phi = [self.models[i](data[:,:,i]) for i in range(self.k)]  # each phi[i] should have shape (b,1)
                    phi = [self.models[i](data[...,i]) for i in range(self.k)]  # each phi[i] should have shape (b,1)
                    # print(f'phi0 shape {phi[0].shape}')
                    e_term = self.calc_exp_term(phi, data)
                    loss = -(sum(phi).mean() - self.eps*e_term)
                    if self.params.regularize_pariwise_coupling and self.params.cost_graph != 'full':
                        # print('regularize_pariwise_coupling')
                        pairwise_reg = self.calc_pairwise_coupling_regularizer(phi, data)
                        reg_loss = loss + self.params.regularize_pariwise_coupling_reg * pairwise_reg
                        reg_loss.backward()
                    ####
                    # bimargin loss (for debugging)
                    # cost = self.calc_cost(data)
                    # loss = dual_loss(pred_x = phi[0], pred_y = phi[1], cost=cost, eps=self.eps)
                    ####
                    else:
                        loss.backward()

                    if self.params.clip_grads:
                        torch.nn.utils.clip_grad_norm_(self.models[k_ind].parameters(), self.params.max_grad_norm)

                    self.opt[k_ind].step()
                    l.append(loss.item())

            print_plan_overall = False
            if print_plan_overall:
                phi_all = [self.models[i](X[:,:,i]) for i in range(self.k)]
                print(self.calc_exp_term(phi_all, X))

            l = np.mean(l)
            self.tot_loss.append(-l+ self.eps)
            epoch_time = timer()-t0
            self.times.append(epoch_time)
            print(f'finished epoch {epoch}, loss={-l+ self.eps:.5f}, took {epoch_time:.2f} seconds')

            print_debug = True
            if epoch%10==0 and print_debug and self.params.cost_graph != 'full' and self.params.calc_ot_cost and self.params.cost_graph != 'tree':
                P = self.calc_plan(X)
                ot_cost = self.calc_ot_cost(P,X)
                print(f'ot_cost {ot_cost}')

            if self.params.schedule:
                for sched in self.scheduler:
                    sched.step()
                    lr = sched.get_last_lr()[0]
                    # print(f'updated learning rate {lr}')

            # print(f'finished epoch {epoch}, loss={l / i}')
        self.models_to_eval

    def save_results(self,X=None):
        tot_loss = np.mean(self.tot_loss[-10:])
        avg_time = np.mean(self.times)
        
        data_to_save = {
            'avg_loss': tot_loss,
            'avg_time': avg_time,
            'tot_loss': self.tot_loss,
            'times': self.times,
            'params': self.params,
            # 'plan': plan,
        }
        if self.params.cost_graph != 'full' and self.params.calc_ot_cost and self.params.cost_graph != 'tree':
            plan = self.calc_plan(X)
            ###
            ot_cost = self.calc_ot_cost(plan,X)
            ###
            data_to_save['ot_cost'] = ot_cost
        else:
            ot_cost = 0
        # Save path
        path = os.path.join(self.params.figDir, 'results.pkl')

        # Saving the data using pickle
        with open(path, 'wb') as file:
            pickle.dump(data_to_save, file)

        # Save tot_loss and avg_time to a text file
        txt_path = os.path.join(self.params.figDir, 'results_summary.txt')
        with open(txt_path, 'w') as txt_file:
            txt_file.write(f"Total Loss: {tot_loss}\n")
            txt_file.write(f"Average Time: {avg_time}\n")

        if self.using_wandb:
            wandb.log({'tot_loss': tot_loss,
                       'avg_time': avg_time
                       })
            if self.params.cost_graph != 'full':
                wandb.log(({
                       'ot_cost': ot_cost
                       }))

        if self.params.cost_graph != 'full':
            print(f'Finished run, loss is {tot_loss:.5f}, average epoch time is {avg_time:.3f} seconds, ot_cost {ot_cost:.5f}')
        else:
            print(f'Finished run, loss is {tot_loss:.5f}, average epoch time is {avg_time:.3f} seconds')

    def calc_ot_cost(self, P, X):
        C = self.calc_cost(X)
        # phi = [self.models[i](X[:, :, i]) for i in range(self.k)]
        # e_term = torch.eye(phi[0].shape[0]).to(self.device)
        # for i in range(self.k):
        #     L = torch.exp((0.5 * (phi[i] + phi[(i + 1) % self.k].T) - C[i]) / self.eps)
        #     e_term = (e_term @ L)
        # normal = torch.trace(e_term)
        ot_cost = sum([torch.sum(c * p) for (c, p) in zip(C, P)])
        return ot_cost
        # return ot_cost/normal

    def calc_exp_term(self, phi, x):
        """
        Calculate the exponential term in the dual OT formulation.
        
        Args:
            phi (list): List of potential functions from each neural network
            x: Input data tensor
            
        Returns:
            float: Mean of the exponential term across the batch
            
        The calculation depends on the cost_graph structure:
        - 'full': Computes all pairwise interactions
        - 'circle': Only computes interactions between adjacent marginals
        - 'tree': Follows a tree structure for interactions
        """
        # calc loss tensor

        if self.params.cost_implement == 'simplified':
            reduced_phi = torch.sum(torch.concatenate(phi, axis=1), axis=1)
            if self.params.dataset == 'cifar':
                # calc scores pairwise
                cost = calc_deep_feature_cost(x)
                cost = cost/cost.max()
                e_term = torch.exp((reduced_phi - cost.sum(axis=(1,2)))/self.eps)
                return e_term.mean()
            if self.cost_graph == 'full':
                # calculate the simplified loss
                # calc reduced phi term:

                # cal reduced cost term:
                # print(f'x shape {x.shape}')
                diffs = x.unsqueeze(-1) - x.unsqueeze(-2)
                cost = 0.5*torch.norm(diffs, dim=1) ** 2
                if self.params.dataset == 'mnist':
                    #COSSIM:
                    cost = (x.unsqueeze(-1) * x.unsqueeze(-2)).sum(dim=1)
                    cost = cost/cost.max()
                    # cost = ssim_matrix(x)
                if self.params.norm_by_k:
                    cost = cost / self.k
                e_term = torch.exp((reduced_phi - cost.sum(axis=(1,2)))/self.eps)
            elif self.cost_graph == 'circle':
                shifted_x = torch.roll(x, shifts=1, dims=-1)
                diffs = x - shifted_x
                cost = torch.norm(diffs, dim=1) ** 2
                e_term = torch.exp((reduced_phi - cost.sum(axis=(-1))) / self.eps)
            return e_term.mean()

        # COMBINATORIAL IMPLEMENTATION
        c = self.calc_cost(x)
        if self.cost_graph == 'circle':
            n = phi[0].shape[0]
            e_term = torch.eye(n).to(self.device)
            for i in range(self.k):
                L = torch.exp((0.5*(phi[i] + phi[ (i+1)%self.k ].T) - c[i])/self.eps)
                e_term = (e_term @ L)*(1/n)
            return torch.trace(e_term)
            # # calc mapping:
            # reshaped_term = []
            # reshaped_c = []
            # for index, vec in enumerate(phi):
            #     # Create a shape of length k with 1s except at the index position
            #     shape = [1] * self.k
            #     shape[index] = -1
            #     reshaped_c.append(c[index].reshape(shape))
            #     reshaped_term.append(vec.reshape(shape))
            # reshaped_term = sum(reshaped_term)
            # c = sum(reshaped_c)
        elif self.cost_graph == 'full':
            reshaped_term = []
            for index, vec in enumerate(phi):
                # Create a shape of length k with 1s except at the index position
                shape = [1] * self.k
                shape[index] = -1
                # shape[-index] = -1
                reshaped_term.append(vec.reshape(shape))
            reshaped_term = sum(reshaped_term)
            # reshaped_term = phi[0][None, :] + phi[1][:, None]
            return torch.mean(torch.exp((reshaped_term-c)/self.eps))
        elif self.cost_graph == 'tree':
            n = phi[0].shape[0]
            e_term = self.calc_exp_term_tree(n,phi,c)
            return e_term

    def zero_grad_models(self):
        for opt in self.opt:
            opt.zero_grad()


    def calc_exp_term_tree(self, n, phi, c):
        """
        We traverse the tree and aggregate the multiplications.
        """

        def traverse_and_calculate(node):
            # Aggregate children node calculation into V
            V = torch.ones(size=(n,1)).cuda()
            for child in node.children:
                V = V*traverse_and_calculate(child)

            # If we're at the root then we need to calculate
            if node.is_root_flag:
                # there is a vector and the beginning
                L = 1 / n * torch.exp((phi[node.index] ) / self.eps).t()
                return L @ V

            ones = torch.ones(size=(n, 1)).cuda()
            L = 1/n*torch.exp( ( ones@phi[node.index].t() - c[node.index] )/self.eps )

            return L @ V

        # Traverse the tree starting from the root and calculate the matrices
        return traverse_and_calculate(self.tree_root).squeeze()


    def calc_cost(self, data):
        """
        calculates the cost over bacthed data
        :param data:
        :return:
        """
        if self.cost == 'quad':
            if self.cost_graph == 'circle' and self.params.euler == 1:
                cost = QuadCost(data, mod='euler')
            else:
                cost = QuadCost(data, mod=self.cost_graph, root=self.tree_root)
        elif self.cost == 'quad_gw':
            # IMPLEMENT - cost = QuadCostGW(data, self.matrices)
            pass
        elif self.cost == 'ip_gw':
            # IMPLEMENT - cost = IPCostGW(data, self.matrices)
            pass
        
        if self.params.dataset == 'mnist':
            cost = cost/data.shape[1]
        # NOW - BROADCAST!!
        return cost

    
    def models_to_eval(self):
        for model in self.models:
            model.eval()

    def calc_plan(self, X):
        """
        Calculate the optimal transport plan based on the trained neural networks.
        
        Args:
            X: Input data tensor
            
        Returns:
            tensor: The optimal transport plan
            
        The plan calculation depends on the cost_graph structure:
        - For 'circle': Computes pairwise plans between adjacent marginals
        - For other structures: Computes the full plan based on all potentials
        """
        if self.cost_graph == 'circle':
            phi = [self.models[i](X[:, :, i]) for i in range(self.k)]
            c = self.calc_cost(X)
            exp_terms = [torch.exp((0.5*(phi[i] + phi[ (i+1)%self.k ].T) - c[i])/self.eps) for i in range(self.k)]
            ot_plan = []
            for i in range(self.k):
                P = self.calc_pairwise_plan(exp_terms,i, (i+1)%self.k )
                if self.params.normalize_plan:
                    P = P/torch.sum(P)
                ot_plan.append(P)
                if self.params.check_P_sum and self.params.using_wandb:
                    print('h')
                    wandb.log({f'ot_plan_{i}_sum': torch.sum(ot_plan[i]).item()})
        else:
            phi = [self.models[i](X[:, :, i]) for i in range(self.k)]
            reshaped_term = []
            for index, vec in enumerate(phi):
                # Create a shape of length k with 1s except at the index position
                shape = [1] * self.k
                shape[index] = -1
                # shape[-index] = -1
                reshaped_term.append(vec.reshape(shape))
            reshaped_term = sum(reshaped_term)
            # reshaped_term = phi[0][None, :] + phi[1][:, None]
            c = self.calc_exp_term(phi, X)
            ot_plan = torch.exp((reshaped_term - c) / self.eps)
        return ot_plan

    def calc_pairwise_plan(self, L, i, verbose=False):
        """
        Calculate the pairwise optimal transport plan between marginals.
        
        Args:
            L (list): List of transport matrices
            i (int): Index of the source marginal
            verbose (bool): Whether to print debugging information
            
        Returns:
            tensor: The normalized pairwise transport plan
            
        Computes the plan by:
        1. Calculating product of matrices up to index i-1
        2. Calculating product of matrices after index i
        3. Computing the final plan through matrix operations
        """
        # Compute A: Product of matrices up to index i-1
        if i > 0:
            A = L[0]
            for k in range(1, i):
                A = A @ L[k]
        else:
            # Use identity matrix if no matrices before index i
            size = L[i].shape[0]
            A = torch.eye(size, dtype=L[i].dtype, device=L[i].device)

        # Compute B: Product of matrices from index i+1 to end
        if i + 1 < len(L):
            B = L[i + 1]
            for k in range(i + 2, len(L)):
                B = B @ L[k]
        else:
            # Use identity matrix if no matrices after index i
            size = L[i].shape[1]
            B = torch.eye(size, dtype=L[i].dtype, device=L[i].device)

        # Calculate A.T @ B.T
        C = A.T @ B.T

        # Element-wise multiplication with L[i]
        output = C * L[i]

        # if verbose:
        #     print(output.sum())

        return output/output.sum()

    def calc_pairwise_coupling_regularizer(self, phi, x):
        c = self.calc_cost(x)
        exp_terms = [torch.exp((0.5*(phi[i] + phi[ (i+1)%self.k ].T) - c[i])/self.eps) for i in range(self.k)]
        reg = 0
        for i in range(self.k):
            ot_plan = self.calc_pairwise_plan(exp_terms, i, (i + 1) % self.k, verbose=False)
            reg += (torch.sum(ot_plan)-1.0).abs()
        return reg


class Net_EOT(nn.Module):
    """
    Neural network for Entropy-regularized Optimal Transport estimation.
    
    A flexible neural architecture that can handle different input dimensions and data types.
    Supports both standard feedforward networks and convolutional networks for image data.
    
    Args:
        dim (int): Input dimension
        K (int): Hidden layer width multiplier
        deeper (bool): Whether to use a deeper architecture
        data (str): Type of data ('mnist', 'cifar', None)
    """
    def __init__(self, dim, K, deeper=False, data=None):
        super(Net_EOT, self).__init__()
        self.deeper = deeper
        self.data = data
        
        if self.data == "cifar":
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc_c1 = nn.Linear(128 * 4 * 4, 256)
            self.fc_c2 = nn.Linear(256, 1)
        elif data == 'mnist':
            self.fc1 = nn.Linear(dim, 2 * dim)
            self.fc2 = nn.Linear(2 * dim, 2 * dim)
            self.fc3 = nn.Linear(2 * dim, 32)
            self.fc4 = nn.Linear(32, 1)
        elif deeper:
            self.fc1 = nn.Linear(dim, 10 * K)
            self.fc2 = nn.Linear(10 * K, 10 * K)
            self.fc3 = nn.Linear(10 * K, K)
            self.fc4 = nn.Linear(K, 1)
        else:
            self.fc1 = nn.Linear(dim, K)
            self.fc2 = nn.Linear(K, 1)

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor
            
        Returns:
            tensor: Network output (potential function value)
            
        The architecture adapts based on the data type:
        - For MNIST: Uses a 4-layer MLP with specific dimensions
        - For CIFAR: Uses a CNN with 3 conv layers followed by FC layers
        - For other data: Uses either a 2-layer or 4-layer MLP based on 'deeper' flag
        """
        if self.data == 'mnist':
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            x = x / 255.0
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x2 = self.fc4(x)
        elif self.data == "cifar":
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc_c1(x))
            return self.fc_c2(x)
        elif self.deeper:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x2 = self.fc4(x)
        else:
            x1 = F.relu(self.fc1(x))
            x2 = self.fc2(x1)
        return x2


class _FrozenResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # up to conv5_x
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

        for p in self.features.parameters():
            p.requires_grad = False   # encoder is frozen

    def forward(self, x):             # x : (N, 3, 32, 32)
        z = self.pool(self.features(x)).flatten(1)   # (N, 512)
        return z



############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
# Support functions
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################



class DeepFeatureL2(nn.Module):
    """
    Differentiable distance for CIFAR images.
    • Input : x, y   tensors of shape (B, 3, 32, 32) in [0,1], already mean/std-normalised.
    • Output: d      tensor (B,) where smaller ⇒ images are more alike.
    """
    def __init__(self, device="cpu"):
        super().__init__()

        # 1) Pre-trained backbone (frozen)
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # keep layers up to conv5_x
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.to(device).eval()

        # 2) Tiny head that global-avg-pools to a 512-d vector (still differentiable)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # ------------------------------------------------------------
    def forward(self, x, y):
        # both x and y : (B,3,32,32)   - ensure same device
        z1 = self.avgpool(self.encoder(x)).flatten(1)      # (B, 512)
        z2 = self.avgpool(self.encoder(y)).flatten(1)

        # squared L2 in feature space  – size (B,)
        return (z1 - z2).pow(2).sum(dim=1)

def ssim_matrix(x: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Compute a (K × K) SSIM matrix for each element of a batched tensor of
    vectorised MNIST images.

    Parameters
    ----------
    x : torch.Tensor
        Shape (B, 784, K).  Values can be in [0, 1] or [0, 255].
    window_size : int, optional
        Size of the Gaussian window used inside SSIM (default 11).

    Returns
    -------
    torch.Tensor
        SSIM matrix of shape (B, K, K), differentiable.
    """
    # pdb.set_trace()
    B, N, K = x.shape
    assert N == 784, "Second dimension must be 784 (28x28)."

    # Put channel first and reshape: (B, K, 1, 28, 28)
    imgs = x.permute(0, 2, 1).reshape(B, K, 1, 28, 28)

    # Normalise to [0,1] if needed
    if imgs.max() > 1:
        imgs = imgs / 255.0

    # Pre-allocate the similarity tensor
    S = torch.empty(B, K, K, device=imgs.device, dtype=imgs.dtype)

    # Kornia's ssim_loss expects (N, C, H, W);
    # we compare every pair (i, j) in a double loop.
    # K is small (≤10 for MNIST digits), so the loop is negligible.
    for i in range(K):
        for j in range(K):
            dssim = kornia.losses.ssim_loss(
                imgs[:, i], imgs[:, j], window_size=window_size, reduction='mean'
            )  # Reduce to a scalar (B,)
            S[:, i, j] = 1.0 - 2.0 * dssim  # convert DSSIM → SSIM

    return S

# single shared instance on the proper device


# ------------------------------------------------------------------
# 2)  Pair-wise cost builder
# ------------------------------------------------------------------
def calc_deep_feature_cost(batch: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    batch : (B, 3, 32, 32, k) tensor
        A minibatch containing k CIFAR images per "sample".
        Assumed already in [0,1] and channel-wise normalised
        (e.g. with CIFAR-10/100 mean & std).

    Returns
    -------
    cost : (B, k, k) tensor
        Squared L2 distances in ResNet-18 feature space.
        cost[b, i, j] is small when images `i` and `j` in the
        same sample `b` look similar; zero on the diagonal.
    """
    _encoder = _FrozenResNet18().to("cuda" if torch.cuda.is_available() else "cpu").eval()
    device = next(_encoder.parameters()).device
    B, C, H, W, k = batch.shape
    assert C == 3 and H == 32 and W == 32, "Expect CIFAR sized tensors"

    # ——— reshape so we can embed all images in one forward pass ———
    imgs = batch.permute(0, 4, 1, 2, 3).reshape(B * k, C, H, W).to(device)

    # embeddings: (B*k, 512)
    feats = _encoder(imgs)                         

    # reshape back → (B, k, 512)
    feats = feats.view(B, k, -1)

    # ——— pair-wise squared L2 in feature space ———
    # feats[:, :, None, :] − feats[:, None, :, :] → (B, k, k, 512)
    d2 = (feats[:, :, None, :] - feats[:, None, :, :]).pow(2).sum(-1)

    return d2      # shape (B, k, k)




############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
# Additional code
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################


class A_model(nn.Module):
    def __init__(self, d_in, d_out):
        super(A_model, self).__init__()
        self.A = nn.Parameter(torch.full((d_in, d_out), 1e-3))

    def forward(self, x):
        x1, x2 = x
        return x1 @ self.A @ x2.T

class MGW_NE_alg(MOT_NE_alg):
    def __init__(self, params, device, X):
        """
        EMGW agent
        built upon the MOT_NE_alg.

        """
        super().__init__(params, device)
        self.initialize_alg(X)
        # same optimizer for all matrices:

    def initialize_alg(self,X):
        """
        Initialize the A matrices of the MGW problem
        """
        if self.params.A_mgw_opt == 'autograd':
            # INIT As at models
            if self.cost_graph == 'full':
                # all (i,j) options
                pass
            elif self.cost_graph == 'circle':
                # k  matrices
                As = []
                self.opt_A = []
                for i in range(self.k):
                    A = A_model(self.params.dims[i],self.params.dims[(i+1)%self.k])
                    self.opt_A.append(torch.optim.Adam(A.parameters(), lr=1e-4))
                    As.append(A.cuda())
            elif self.cost_graph == 'tree':
                # k-1 matrices
                pass
            self.A_matrices = As
        else:
            # Doing first order optimization:
            self.tolerance = 1e-4
            self.max_iter = 100
            self.K_const = 32
            self.M = 1
            self.L = max(64,32*32*(1/9+4/(45*self.params.dims[0]))/self.eps-64)
            norms = [torch.norm(x, dim=1) for x in X]
            self.C_1 = [-4*norms[i][:,None]*norms[(i+1)%self.k][None,:] for i in range(self.k)]
            self.S1 = 0 # TD
            self.A_matrices = [torch.ones(self.params.dims[i],self.params.dims[(i+1)%self.k]) * 1e-5 for i in range(self.k)]
            self.C_matrices = [torch.ones(self.params.dims[i], self.params.dims[(i + 1) % self.k]) * 1e-5 for i in range(self.k)]

    def train_with_oracle(self, X):
        nemot_n_epochs = 5
        x_b = MultiTensorDataset(X)
        x_b = DataLoader(x_b, batch_size=self.batch_size, shuffle=True)

        self.tot_loss = []
        self.times = []

        for iter in range(self.max_iter):

            for epoch in range(nemot_n_epochs):
                l = []
                # perform an epoch
                for i, data in enumerate(x_b):
                    self.zero_grad_models()
                    for k_ind in range(self.k):
                        # k-margin loss
                        phi = [self.models[i](data[i]) for i in
                               range(self.k)]  # each phi[i] should have shape (b,1)
                        e_term = self.calc_exp_term_mgw(phi, data)
                        loss = -(sum(phi).mean() - self.eps * e_term)
                        loss.backward()

                        if self.params.clip_grads:
                            torch.nn.utils.clip_grad_norm_(self.models[k_ind].parameters(),
                                                           self.params.max_grad_norm)
                        self.opt[k_ind].step()
                        l.append(loss.item())
                l = np.mean(l)
                self.tot_loss.append(-l + self.eps)
                print(f'iter: {iter}, finished NEMOT epoch {epoch}, loss: {-l + self.eps:.5f}')

            # PERFORM A SINGLE A UPDATE:
            # THIS IS LIMITED TO SIMPLIFIED PLANS!
            gamma = iter / (4 * self.L)
            tau = 2 / (iter + 2)
            P = self.calc_plan(X)
            print(f'iteration {iter}')
            for i in range(len(self.A_matrices)):
                A = self.A_matrices[i]
                C = self.C_matrices[i]
                grad = 64 * A - 32 * X[i].T @ P[i] @ X[i + 1]
                print(f'grad_i norm is {torch.linalg.norm(grad)}')
                B = torch.where(torch.abs(A - grad / (2 * self.L)) <= self.M / 2, A - grad / (2 * self.L),
                                self.M / 2)
                C = torch.where(torch.abs(C - grad * gamma) <= self.M / 2, A - grad * gamma, self.M / 2)
                A = tau * C + (1 - tau) * B
                self.A_matrices[i] = A
                self.C_matrices[i] = C

                    # if self.params.schedule:
                    #     for sched in self.scheduler:
                    #         sched.step()

    def calc_exp_term_mgw(self,phi,x):
        norms = [torch.norm(x_, dim=1) ** 2 for x_ in x]

        if self.params.A_mgw_opt == 'autograd':
            cost_list = [torch.diagonal(-4* norms[i][:,None]* norms[(i+1)%self.k][None,:] -32 * self.A_matrices[i]( (x[i], x[(i+1)%self.k])) ) for i in range(self.k)]
        else:
            cost_list = [torch.diagonal(
                -4 * norms[i][:, None] * norms[(i + 1) % self.k][None, :] - 32 * x[i] @ self.A_matrices[i].cuda() @ x[
                    (i + 1) % self.k].T) for i in range(self.k)]

        reduced_phi = torch.sum(torch.concatenate(phi, axis=1), axis=1)
        cost = sum(cost_list)
        e_term = torch.exp((reduced_phi - cost) / self.eps)
        return torch.mean(e_term)

    def calc_plan_mgw(self, X):
        # TD!!! plan calculation!!!
        if self.cost_graph == 'circle':
            phi = [self.models[i](X[i]) for i in range(self.k)]
            c = self.calc_cost_mgw(X)
            exp_terms = [torch.exp((0.5*(phi[i] + phi[ (i+1)%self.k ].T) - c[i])/self.eps) for i in range(self.k)]
            ot_plan = []
            for i in range(self.k):
                P = self.calc_pairwise_plan(exp_terms,i, (i+1)%self.k )
                if self.params.normalize_plan:
                    P = P/torch.sum(P)
                ot_plan.append(P)
                if self.params.check_P_sum and self.params.using_wandb:
                    print('h')
                    wandb.log({f'ot_plan_{i}_sum': torch.sum(ot_plan[i]).item()})
        else:
            phi = [self.models[i](X[:, :, i]) for i in range(self.k)]
            reshaped_term = []
            for index, vec in enumerate(phi):
                # Create a shape of length k with 1s except at the index position
                shape = [1] * self.k
                shape[index] = -1
                # shape[-index] = -1
                reshaped_term.append(vec.reshape(shape))
            reshaped_term = sum(reshaped_term)
            # reshaped_term = phi[0][None, :] + phi[1][:, None]
            c = self.calc_exp_term(phi, X)
            ot_plan = torch.exp((reshaped_term - c) / self.eps)
        return ot_plan


    def train_mgw_combined(self, X):
        """
        Training both A matrices and MGW via automatic differentiation
        :param X:
        :return:
        """
        x_b = MultiTensorDataset(X)
        x_b = DataLoader(x_b, batch_size=self.batch_size, shuffle=True)
        self.tot_loss = []
        self.times = []
        nemot_epochs = 3
        for epoch in range(self.num_epochs):
            A_opt_flag = epoch > 9 and epoch%nemot_epochs==0
            t0 = timer()
            l = []
            if A_opt_flag:
                # train A
                l = []
                for i, data in enumerate(x_b):
                    self.zero_grad_models()
                    for k_ind in range(self.k):
                        phi = [self.models[i](data[i]) for i in
                               range(self.k)]  # each phi[i] should have shape (b,1)
                        e_term = self.calc_exp_term_mgw(phi, data)
                        loss_ = sum(phi).mean() - self.eps * e_term
                        frob_norms = 32*sum([model.A.norm(p='fro')**2 for model in self.A_matrices])
                        loss = loss_ + frob_norms
                        loss.backward()

                        # if self.params.clip_grads:
                        #     torch.nn.utils.clip_grad_norm_(self.A_matrices[k_ind].parameters(),
                        #                                    self.params.max_grad_norm)
                        self.opt_A[k_ind].step()
                    l.append(loss.item())
                self.tot_loss.append(np.mean(l) + self.eps)
            else:
                # train NEMOT
                for i, data in enumerate(x_b):
                    self.zero_grad_models()
                    for k_ind in range(self.k):
                        # k-margin loss
                        phi = [self.models[i](data[i]) for i in
                               range(self.k)]  # each phi[i] should have shape (b,1)
                        e_term = self.calc_exp_term_mgw(phi, data)
                        loss = -(sum(phi).mean() - self.eps * e_term)
                        loss.backward()

                        if self.params.clip_grads:
                            torch.nn.utils.clip_grad_norm_(self.models[k_ind].parameters(),
                                                           self.params.max_grad_norm)
                        self.opt[k_ind].step()
                        l.append(loss.item())
            epoch_time = timer() - t0
            l = np.mean(l)
            self.times.append(epoch_time)
            if A_opt_flag:
                print(f'finished epoch {epoch}, A opt loss: {l + self.eps:.5f}')
            else:
                print(f'finished epoch {epoch}, loss: {-l + self.eps:.5f}')

        #




        #
        # X_b = [DataLoader(x, batch_size=self.batch_size, shuffle=True) for x in X]
        # self.tot_loss = []
        # self.times = []
        # self.nemot_n_epochs = 5
        # for epoch in range(self.num_epochs):
        #     l = []
        #     t0 = timer()
        #     if epoch%self.nemot_n_epochs == 0 and epoch>0:  # M MATRICES EPOCH
        #         # train A matrices this epoch:
        #         # for i, data in enumerate(zip(X_b)):
        #         for data in zip(X_b):
        #             self.opt_A.zero_grad()
        #             # k-margin loss
        #             phi = [self.models[i](data[i]) for i in
        #                    range(self.k)]  # each phi[i] should have shape (b,1)
        #             e_term = self.calc_exp_term(phi, data)
        #             loss = -(sum(phi).mean() - self.eps * e_term)+32*sum([torch.norm(A, p='fro') for A in self.A_matrices])
        #             loss.backward()
        #
        #             if self.params.clip_grads:
        #                 torch.nn.utils.clip_grad_norm_(self.A_matrices, self.params.max_grad_norm)
        #
        #             self.opt_A.step()
        #             l.append(loss.item())
        #
        #         print_plan_overall = False
        #         if print_plan_overall:
        #             phi_all = [self.models[i](X[:, :, i]) for i in range(self.k)]
        #             print(self.calc_exp_term(phi_all, X))
        #
        #         l = np.mean(l)
        #         self.tot_loss.append(-l + self.eps)
        #         epoch_time = timer() - t0
        #         self.times.append(epoch_time)
        #         print(f'finished A_matrices epoch {epoch}, loss={-l + self.eps:.5f}, took {epoch_time:.2f} seconds')
        #
        #     else:
        #         # train NEMOT this epoch:
        #         for data in zip(*X_b):
        #             self.zero_grad_models()
        #             # data.to(self.device)
        #             for k_ind in range(self.k):
        #                 # k-margin loss
        #                 phi = [self.models[i](data[:, :, i]) for i in range(self.k)]  # each phi[i] should have shape (b,1)
        #                 e_term = self.calc_exp_term(phi, data)
        #                 loss = -(sum(phi).mean() - self.eps * e_term)
        #                 if self.params.regularize_pariwise_coupling and self.params.cost_graph != 'full':
        #                     # print('regularize_pariwise_coupling')
        #                     pairwise_reg = self.calc_pairwise_coupling_regularizer(phi, data)
        #                     reg_loss = loss + self.params.regularize_pariwise_coupling_reg * pairwise_reg
        #                     reg_loss.backward()
        #                 else:
        #                     loss.backward()
        #
        #                 if self.params.clip_grads:
        #                     torch.nn.utils.clip_grad_norm_(self.models[k_ind].parameters(), self.params.max_grad_norm)
        #
        #                 self.opt[k_ind].step()
        #                 l.append(loss.item())
        #
        #         print_plan_overall = False
        #         if print_plan_overall:
        #             phi_all = [self.models[i](X[:, :, i]) for i in range(self.k)]
        #             print(self.calc_exp_term(phi_all, X))
        #
        #         l = np.mean(l)
        #         self.tot_loss.append(-l + self.eps)
        #         epoch_time = timer() - t0
        #         self.times.append(epoch_time)
        #         print(f'finished NEMOT epoch {epoch}, loss={-l + self.eps:.5f}, took {epoch_time:.2f} seconds')
        #
        #     print_debug = True
        #     if epoch % 10 == 0 and print_debug and self.params.cost_graph != 'full' and self.params.calc_ot_cost and self.params.cost_graph != 'tree':
        #         P = self.calc_plan(X)
        #         ot_cost = self.calc_ot_cost(P, X)
        #         print(f'ot_cost {ot_cost}')
        #
        #     if self.params.schedule:
        #         for sched in self.scheduler:
        #             sched.step()
        #             lr = sched.get_last_lr()[0]
        #             # print(f'updated learning rate {lr}')
        #
        #     # print(f'finished epoch {epoch}, loss={l / i}')
        # self.models_to_eval

    def save_results(self,X=None):
        S1 = self.calc_S1(X)
        tot_loss = np.mean(self.tot_loss[-10:])
        avg_time = np.mean(self.times)
        data_to_save = {
            'avg_loss': tot_loss,
            'avg_time': avg_time,
            'tot_loss': self.tot_loss,
            'times': self.times,
            'params': self.params,
            # 'plan': plan,
        }
        # if self.params.cost_graph != 'full' and self.params.calc_ot_cost and self.params.cost_graph != 'tree':
        #     plan = self.calc_plan(X)
        #     ###
        #     ot_cost = self.calc_ot_cost(plan,X)
        #     ###
        #     data_to_save['ot_cost'] = ot_cost
        # else:
        #     ot_cost = 0
        # Save path
        path = os.path.join(self.params.figDir, 'results.pkl')

        # Saving the data using pickle
        with open(path, 'wb') as file:
            pickle.dump(data_to_save, file)

        if self.using_wandb:
            wandb.log({'tot_loss': tot_loss+self.eps,
                       'avg_time': avg_time
                       })
            # if self.params.cost_graph != 'full':
            #     wandb.log(({
            #            'ot_cost': ot_cost
            #            }))

        # if self.params.cost_graph != 'full':
        #     print(f'Finished run, loss is {tot_loss:.5f}, average epoch time is {avg_time:.3f} seconds, ot_cost {ot_cost:.5f}')
        # else:
        #     print(f'Finished run, loss is {tot_loss:.5f}, average epoch time is {avg_time:.3f} seconds')

    def calc_cost(self, data):
        """
        calculates the cost over bacthed data
        :param data:
        :return:
        """
        if self.cost == 'quad':
            cost = QuadCostGW(data, self.A_matrices)


        # NOW - BROADCAST!!
        return cost

    def calc_S1(self,X):
        """
        Generalized S1 calculation for a list of tensors [x0, x1, ..., x_{k-1}],
        each of shape (n, d).

        Returns a 1D torch tensor of length (k-1), where result[i] is the
        S1 value computed for the pair (X[i], X[i+1]).
        """
        k = self.k
        if k < 2:
            raise ValueError("Need at least 2 variables to form a pair.")

        results = []

        for i in range(k):
            x = X[i]
            y = X[(i + 1)%k]

            # Ensure x, y are 2D: (n, d)
            if x.dim() != 2 or y.dim() != 2:
                raise ValueError("Each tensor x[i] must be of shape (n, d).")

            n = x.shape[0]

            # 1) Compute norms (squared) along the rows (dim=1)
            #    x_norm_2 and y_norm_2 each of shape (n,)
            x_norm_2 = torch.norm(x, dim=1, p=2) ** 2
            y_norm_2 = torch.norm(y, dim=1, p=2) ** 2

            # Square them again to get x_norm_4, y_norm_4
            x_norm_4 = x_norm_2 ** 2
            y_norm_4 = y_norm_2 ** 2

            # 2) Means along the n dimension
            M2_x = x_norm_2.mean()
            M2_y = y_norm_2.mean()
            M4_x = x_norm_4.mean()
            M4_y = y_norm_4.mean()

            # 3) Average outer products => (d, d) shapes
            sig_x = x.t().matmul(x) / n  # shape (d, d)
            sig_y = y.t().matmul(y) / n  # shape (d, d)

            # 4) Frobenius norms squared
            #    (||sig_x||_F^2, ||sig_y||_F^2)
            #    Using p='fro' in torch.norm, then square it:
            F_x = torch.norm(sig_x, p='fro') ** 2
            F_y = torch.norm(sig_y, p='fro') ** 2

            # 5) S1 formula
            S1_val = (
                    2 * (M4_x + M4_y)
                    + 2 * (M2_x ** 2 + M2_y ** 2)
                    + 4 * (F_x + F_y)
                    - 4 * (M2_x * M2_y)
            )

            # Keep it as a scalar tensor (not converting to Python float)
            results.append(S1_val.unsqueeze(0))

        # Stack into a 1D tensor of length k-1
        return torch.cat(results, dim=0)


class MOT_NE_alg_JAX:
    def __init__(self, params, device=None):  # Device is not explicitly used in JAX, it manages devices
        self.params = params
        self.k = params['k']
        self.num_epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.eps = params['eps']
        self.cost_graph = params['cost_graph']
        self.using_wandb = params['using_wandb']  # Assuming wandb is set up separately
        self.key = jr.PRNGKey(0)  # Initialize a PRNGKey


        # Initialize models and optimizers
        self.models = []
        self.opts = []
        self.opt_states = []
        for i in range(self.k):
            self.key, subkey = jr.split(self.key)
            d = params['dims'][i]
            model = Net_EOT_JAX(dim=d, K=min(6 * d, 80), deeper=True)  # Replace with actual JAX Net_EOT
            params_init = model.init(subkey)  # Assuming your Net_EOT has an init method
            self.models.append((model, params_init))

            optimizer = optax.adam(learning_rate=params['lr'])
            opt_state = optimizer.init(params_init)
            self.opts.append(optimizer)
            self.opt_states.append(opt_state)

        if params.get('schedule', False):
            # Use piecewise_constant_schedule for step decay
            boundaries_and_scales = {
                int(params['schedule_step'] * i): params['schedule_gamma'] ** i
                for i in range(1, int(self.num_epochs // params['schedule_step']) + 1)
            }
            self.schedulers = [optax.piecewise_constant_schedule(
                init_value=params['lr'],
                boundaries_and_scales=boundaries_and_scales
            ) for _ in range(self.k)]

        else:
            self.schedulers = None


        self.tree_root = create_tree(self.params) if self.params.get('cost_graph') == 'tree' else None

    def calc_exp_term(self, phis, x):
      reduced_phi = sum(phis)
      if self.cost_graph == 'full':
          diffs = jnp.expand_dims(x, axis=-1) - jnp.expand_dims(x, axis=-2)
          cost = 0.5 * jnp.linalg.norm(diffs, axis=1) ** 2
          e_term = jnp.exp((reduced_phi - cost.sum(axis=(1, 2))) / self.eps)
      elif self.cost_graph == 'circle':
          shifted_x = jnp.roll(x, shift=1, axis=-1)
          diffs = x - shifted_x
          cost = jnp.linalg.norm(diffs, axis=1) ** 2
          e_term = jnp.exp((reduced_phi - cost.sum(axis=(-1))) / self.eps)
      else:
          raise ValueError(f"Unknown cost_graph: {self.cost_graph}")
      return jnp.mean(e_term)



    @partial(jax.jit, static_argnums=(0,))  # JIT compile the loss function
    def _loss_fn(self, params_list, x):
        phi_all = []
        for i in range(self.k):
            model, _ = self.models[i]  # model instance
            phi = model.apply(params_list[i], x[:, :, i]) # (b, )
            phi_all.append(jnp.expand_dims(phi,axis=-1))

        e_term = self.calc_exp_term(phi_all, x)
        loss = -(sum([p.mean() for p in phi_all]) - self.eps * e_term)
        return loss , loss

    def train_mot_(self, X):
        # X is assumed to be a JAX array of shape (num_batches * batch_size, ...).
        self.tot_loss = []
        self.times = []
        num_batches = X.shape[0] // self.batch_size

        for epoch in range(self.num_epochs):
            epoch_losses = []
            t0 = timer()

            # Iterate over batches (here we slice the data manually, similar to torch DataLoader)
            for i in range(num_batches):
                batch = X[i * self.batch_size:(i + 1) * self.batch_size]
                # In JAX, you don't need to zero gradients because gradients are freshly computed.
                # Gather current parameters for all models.
                params_list = [params for _, params in self.models]

                # Define a loss function that computes the loss for the current batch.
                # Note: Each model in self.models is assumed to be a tuple (model, params).
                def loss_fn(params_list):
                    # Compute phi for each model on its corresponding slice.
                    phi_list = [self.models[k][0].apply(params_list[k], batch[:, :, k])
                                for k in range(self.k)]
                    # Compute the exponential term from the pairwise cost.
                    e_term = self.calc_exp_term(phi_list, batch)
                    # Compute loss as in your Torch logic:
                    #    loss = -(mean(sum(phi)) - self.eps * e_term)
                    loss = -(sum([jnp.mean(phi) for phi in phi_list]) - self.eps * e_term)
                    return loss

                # Compute loss and gradients with respect to all model parameters.
                loss_value, grads_list = jax.value_and_grad(loss_fn)(params_list)

                # Update each model's parameters.
                new_models = []
                new_opt_states = []
                for k_ind in range(self.k):
                    grad = grads_list[k_ind]
                    # Optionally clip gradients if enabled.
                    if self.params.get('clip_grads', False):
                        grad = jax.tree_util.tree_map(lambda g: jnp.clip(g, a_max=self.params['max_grad_norm']),
                                                      grad)
                    # Get updates and new optimizer state from your optimizer.
                    updates, new_opt_state = self.opts[k_ind].update(grad, self.opt_states[k_ind],
                                                                     params_list[k_ind])
                    # Apply the updates.
                    updated_params = optax.apply_updates(params_list[k_ind], updates)
                    # Save the updated model parameters.
                    model = self.models[k_ind][0]
                    new_models.append((model, updated_params))
                    new_opt_states.append(new_opt_state)

                # Replace the old models and optimizer states.
                self.models = new_models
                self.opt_states = new_opt_states

                epoch_losses.append(-loss_value+self.eps)

            # Compute average loss for the epoch.
            avg_loss = jnp.mean(jnp.array(epoch_losses))
            self.tot_loss.append(float(avg_loss))
            epoch_time = timer() - t0
            self.times.append(epoch_time)
            print(f'finished epoch {epoch}, loss={avg_loss:.5f}, took {epoch_time:.2f} seconds')


    def train_mot(self, X):
        X = jnp.array(X)  # Convert to JAX array.
        self.tot_loss = []
        self.times = []

        for epoch in range(self.num_epochs):
            t0 = timer()
            epoch_losses = []
            num_batches = X.shape[0] // self.batch_size
            for i in range(num_batches):

                batch = X[i*self.batch_size:(i+1)*self.batch_size]

                loss_and_grad_fn = value_and_grad(self._loss_fn, argnums=0, has_aux=True)
                params_list = [params for _, params in self.models]
                (total_loss,loss_value), grads_list = loss_and_grad_fn(params_list, batch)

                new_models = []
                new_opt_states = []

                for k_ind in range(self.k):
                    if self.params.get('clip_grads', False):
                        grads_list[k_ind] = tree_map(lambda g: jnp.clip(g, a_max=self.params['max_grad_norm']), grads_list[k_ind])

                    updates, new_opt_state = self.opts[k_ind].update(grads_list[k_ind], self.opt_states[k_ind],params_list[k_ind])

                    updated_params = optax.apply_updates(params_list[k_ind], updates)
                    model, _ = self.models[k_ind]  # Get the model instance
                    new_models.append((model, updated_params))
                    new_opt_states.append(new_opt_state)

                self.models = new_models
                self.opt_states = new_opt_states #Update optimizer states

                epoch_losses.append(loss_value)


            if self.schedulers:
                for sched in self.schedulers:
                    # Assuming schedulers are implemented as optax transformations or functions
                    # that adjust the optimizer state.
                    # print("Learning rate decay not implemented")
                    # print("update scheduler")
                    #Not necessary here, just need to correctly initialize.
                    pass


            l = jnp.mean(jnp.array(epoch_losses)) #epoch loss
            self.tot_loss.append(float(-l + self.eps))  # Convert to Python float for storage
            epoch_time = timer() - t0
            self.times.append(epoch_time)
            print(f'finished epoch {epoch}, loss={-l + self.eps:.5f}, took {epoch_time:.2f} seconds')

        # No need for a separate models_to_eval, just use self.models directly.
        # In JAX, models are typically pure functions, and 'eval' mode isn't a concept.

    def save_results(self, X=None):
        tot_loss = jnp.mean(jnp.array(self.tot_loss[-10:]))
        avg_time = jnp.mean(jnp.array(self.times))

        data_to_save = {
            'avg_loss': float(tot_loss),  # Ensure it's a Python float
            'avg_time': float(avg_time),
            'tot_loss': self.tot_loss,
            'times': self.times,
            'params': self.params,
            # 'plan': plan, # removed plan,
        }
        path = os.path.join(self.params['figDir'], 'results.pkl')
        with open(path, 'wb') as file:
            pickle.dump(data_to_save, file)

        if self.using_wandb:
            # wandb.log({'tot_loss': tot_loss, 'avg_time': avg_time}) # removed wandb
            pass
        else:
            print(f'Finished run, loss is {tot_loss:.5f}, average epoch time is {avg_time:.3f} seconds')

    def calc_pairwise_coupling_regularizer(self, phi, x):
        #Placeholder. Not used, as it is not in the minimal example.
        return 0


class Net_EOT_JAX:  # Consistent with previous code
    def __init__(self, dim, K=None, deeper=None):  # K and deeper are ignored for this specific NN
        self.dim = dim
        if K is None:
            K=dim
        # For a direct translation, K and deeper are not used, but kept for consistency
        self.hidden_dim = 10*K  # Hidden dimension, matching original PyTorch code

    def init(self, key):
        # Initialize parameters using Glorot (Xavier) uniform initialization.
        # This is a common and good practice.
        key1, key2 = jax.random.split(key)
        fc1_w = jax.random.uniform(key1, (self.dim, self.hidden_dim),
                               minval=-jnp.sqrt(6/(self.dim+self.hidden_dim)),
                               maxval=jnp.sqrt(6/(self.dim+self.hidden_dim)))  # Glorot/Xavier uniform
        fc1_b = jax.random.uniform(key1, (self.hidden_dim,),
                               minval=-jnp.sqrt(6/(self.dim+self.hidden_dim)),
                               maxval=jnp.sqrt(6/(self.dim+self.hidden_dim)))

        fc2_w = jax.random.uniform(key2, (self.hidden_dim, self.hidden_dim),
                               minval=-jnp.sqrt(6/(self.hidden_dim+1)),
                               maxval=jnp.sqrt(6/(self.hidden_dim+1)))
        fc2_b = jax.random.uniform(key2, (self.hidden_dim,),
                               minval=-jnp.sqrt(6/(self.hidden_dim+1)),
                               maxval=jnp.sqrt(6/(self.hidden_dim+1)))

        fc3_w = jax.random.uniform(key2, (self.hidden_dim, 1),
                               minval=-jnp.sqrt(6 / (self.hidden_dim + 1)),
                               maxval=jnp.sqrt(6 / (self.hidden_dim + 1)))
        fc3_b = jax.random.uniform(key2, (1,),
                               minval=-jnp.sqrt(6 / (self.hidden_dim + 1)),
                               maxval=jnp.sqrt(6 / (self.hidden_dim + 1)))

        # Combine parameters into a dictionary (or a nested dictionary/tuple)
        params = {
            'fc1': {'w': fc1_w, 'b': fc1_b},
            'fc2': {'w': fc2_w, 'b': fc2_b},
            'fc3': {'w': fc3_w, 'b': fc3_b}
        }
        return params

    def apply(self, params, x):
        # Forward pass
        x = relu(jnp.dot(x, params['fc1']['w']) + params['fc1']['b'])
        x = relu(jnp.dot(x, params['fc2']['w']) + params['fc2']['b'])
        x = jnp.dot(x, params['fc3']['w']) + params['fc3']['b']
        return x.squeeze(-1)  # Remove the last dimension to match the PyTorch output shape (b,)


class encoder_model_JAX:
    def __init__(self, k, dataset, out_dim=10):
        if dataset == 'mnist':
            self.angles = [-15, 15, -10, 10, -5]
            self.translations = [(-2, -2), (2, 2), (-2, 2), (2, -2), (-1, 1)]
        self.k = k
        self.out_dim = out_dim
        self.hidden_dim = 256  # You can adjust this as needed

    def gen_k_views(self, images):
        """
        generate k views of the data
        """
        transformed_images = []
        for image in images:
            transformed = []
            for i in range(self.k):
                if i % 2:
                    # rotate
                    new_img = rotate(image, self.angles[i // 2])
                else:
                    # translate
                    new_img = translate(image, self.translations[i // 2])
                transformed.append(new_img)
                transformed_img = torch.stack(transformed, dim=1)
            transformed_images.append(transformed_img)
        return torch.concatenate(transformed_images, dim=0)

    def init(self, key):
        """
        Initialize the encoder model parameters using Glorot (Xavier) uniform initialization.
        """
        key1, key2, key3 = jr.split(key, 3)
        fc1_w = jr.uniform(key1, (784, self.hidden_dim),
                           minval=-jnp.sqrt(6 / (784 + self.hidden_dim)),
                           maxval=jnp.sqrt(6 / (784 + self.hidden_dim)))  # Glorot/Xavier uniform
        fc1_b = jr.uniform(key1, (self.hidden_dim,),
                           minval=-jnp.sqrt(6 / (784 + self.hidden_dim)),
                           maxval=jnp.sqrt(6 / (784 + self.hidden_dim)))

        fc2_w = jr.uniform(key2, (self.hidden_dim, self.hidden_dim),
                           minval=-jnp.sqrt(6 / (self.hidden_dim + self.hidden_dim)),
                           maxval=jnp.sqrt(6 / (self.hidden_dim + self.hidden_dim)))
        fc2_b = jr.uniform(key2, (self.hidden_dim,),
                           minval=-jnp.sqrt(6 / (self.hidden_dim + self.hidden_dim)),
                           maxval=jnp.sqrt(6 / (self.hidden_dim + self.hidden_dim)))

        fc3_w = jr.uniform(key3, (self.hidden_dim, self.out_dim),
                           minval=-jnp.sqrt(6 / (self.hidden_dim + self.out_dim)),
                           maxval=jnp.sqrt(6 / (self.hidden_dim + self.out_dim)))
        fc3_b = jr.uniform(key3, (self.out_dim,),
                           minval=-jnp.sqrt(6 / (self.hidden_dim + self.out_dim)),
                           maxval=jnp.sqrt(6 / (self.hidden_dim + self.out_dim)))

        params = {
            'fc1': {'w': fc1_w, 'b': fc1_b},
            'fc2': {'w': fc2_w, 'b': fc2_b},
            'fc3': {'w': fc3_w, 'b': fc3_b}
        }
        return params

    def apply(self, params, x):
        """
        Forward pass of the encoder model.
        """
        x = relu(jnp.dot(x, params['fc1']['w']) + params['fc1']['b'])
        x = relu(jnp.dot(x, params['fc2']['w']) + params['fc2']['b'])
        x = jnp.dot(x, params['fc3']['w']) + params['fc3']['b']
        return x
    
    def apply(self, params, x):
        """
        Forward pass of the encoder model.
        """
        return x
    

class contrastiveLearningTrainer:
    def __init__(self, params):
        self.key = jr.PRNGKey(0)
        self.params=params
        self.k = params.k


        # Initialize the dictionary to hold models and parameters, each model is a (model,params) pair
        self.models = {
            'nemot': {
                'model': self._initialize_nemot_models(params)
            },
            'enc': {
                'model': self._initialize_enc_model(params)
            }
        }

        # Initialize optimizers and optimizer states
        self.opt = {
            'nemot': [],
            'enc': None
        }
        self.opt_states = {
            'nemot': [],
            'enc': None
        }

        # Initialize NEMOT optimizers
        for _, model_params in self.models['nemot']['model']:
            optimizer = optax.adam(learning_rate=params['lr'])
            opt_state = optimizer.init(model_params)
            self.opt['nemot'].append(optimizer)
            self.opt_states['nemot'].append(opt_state)

        # Initialize encoder optimizer
        enc_optimizer = optax.adam(learning_rate=params['lr'])
        enc_opt_state = enc_optimizer.init(self.models['enc']['model'][1])
        # enc_opt_state = enc_optimizer.init(self.models['enc']['model'][0].init(self.models['enc']['model'][1]))
        self.opt['enc'] = enc_optimizer
        self.opt_states['enc'] = enc_opt_state

        self.cost_graph = params['cost_graph']
        self.eps = params['eps']

    def _initialize_nemot_models(self, params):
        # Initialize k Net_EOT_JAX models as in the MOT_NE_alg_JAX class
        nemot_models = []
        for _ in range(params['k']):
            d = self.params['enc_dim']
            model = Net_EOT_JAX(dim=d,K=min(6 * d, 80))  # Assuming Net_EOT_JAX is defined elsewhere
            params_init = model.init(jr.PRNGKey(0))  # Initialize model parameters
            nemot_models.append((model, params_init))
        return nemot_models

    def _initialize_enc_model(self, params):
        # Initialize the encoder model from encoder_model_JAX
        model = encoder_model_JAX(params.k, params.dataset)  # Assuming encoder_model_JAX is defined elsewhere
        params_init = model.init(jr.PRNGKey(0))  # Initialize model parameters
        return (model, params_init)

    # ...existing code for other methods...

    def torch_to_jax(self, images, labels):
        """
        maps a torch batch  from a dataloader to jax format
        """
        jax_images = jnp.array(images.numpy())
        jax_labels = jnp.array(labels.numpy())
        return jax_images, jax_labels
        

    def train(self, dataloader):
        """
        models - dictionary, contains the NEMOT models (if NEMOT is trained) and contrastive encoder
        training routine for the contrastive learning framework. 
        flow (for each batch in an epoch):
        1. obtain k views of the data
        2. obtain embedding from the contrastive encoder for each view 
        3. pass embeddings through NEMOT or through Sinkhorn
        4. calculate the contrastive loss 
        5. backprop to update the training model (either NEMOT or contrastive encoder)
        """
        self.tot_loss = []
        self.times = []
        self.evaluate_linear_classifier(dataloader)
        for epoch in range(self.params.epochs):
            epoch_losses = []
            t0 = timer()
            # training_nemot = epoch ==0 or epoch%4
            # training_nemot = epoch%4==0
            training_nemot = 1

            for images, labels in dataloader:
                images = self.models['enc']['model'][0].gen_k_views(images)  # generate k predetermined views of the data
                images, labels = self.torch_to_jax(images, labels)  # map to jax format
                
                if self.params.dataset == 'mnist':
                    images = images.reshape(images.shape[0], images.shape[1], -1)  # flatten the images

                training_params = {
                    'encoder': self.models['enc']['model'][1],
                    'nemot': [params for _, params in self.models['nemot']['model']]
                }

                def loss_fn_nemot(training_params, x):
                    data = self.models['enc']['model'][0].apply(training_params['encoder'], x)  # apply encoder
                    phi_list = [self.models['nemot']['model'][k][0].apply(training_params['nemot'][k], data[:, k, :]) # Compute phi for each model on its corresponding slice.
                                for k in range(self.k)]
                    e_term = self.calc_exp_term(phi_list, data) # Compute the exponential term from the pairwise cost.
                    loss = -(sum([jnp.mean(phi) for phi in phi_list]) - self.eps * e_term)
                    return loss, loss
                def loss_fn_enc(training_params, x):
                    data = self.models['enc']['model'][0].apply(training_params['encoder'], x)  # apply encoder
                    phi_list = [self.models['nemot']['model'][k][0].apply(training_params['nemot'][k], data[:, k, :]) # Compute phi for each model on its corresponding slice.
                                for k in range(self.k)]
                    e_term = self.calc_exp_term(phi_list, data) # Compute the exponential term from the pairwise cost.
                    loss = (sum([jnp.mean(phi) for phi in phi_list]) - self.eps * e_term)
                    return loss, loss
                
                if training_nemot:
                    loss_and_grad_fn = value_and_grad(loss_fn_nemot, argnums=0, has_aux=True)
                else:
                    loss_and_grad_fn = value_and_grad(loss_fn_enc, argnums=0, has_aux=True)
                
                (total_loss,loss_value), grads_list = loss_and_grad_fn(training_params, images)

                if training_nemot:
                    # params_to_update = [params for _, params in self.models['nemot']['model']]
                    
                    new_models = []
                    new_opt_states = []

                    for k_ind in range(self.k):
                        if self.params.get('clip_grads', False):
                            grads_list['nemot'][k_ind] = tree_map(lambda g: jnp.clip(g, a_max=self.params['max_grad_norm']), grads_list['nemot'][k_ind])

                        updates, new_opt_state = self.opt['nemot'][k_ind].update(grads_list['nemot'][k_ind], self.opt_states['nemot'][k_ind],training_params['nemot'][k_ind])

                        updated_params = optax.apply_updates(training_params['nemot'][k_ind], updates)
                        model, _ = self.models['nemot']['model'][k_ind]  # Get the model instance
                        new_models.append((model, updated_params))
                        new_opt_states.append(new_opt_state)

                    self.models['nemot']['model'] = new_models
                    self.opt_states['nemot'] = new_opt_states #Update optimizer states
                else:
                    if self.params.get('clip_grads', False):
                        grads_list['enc'] = tree_map(lambda g: jnp.clip(g, a_max=self.params['max_grad_norm']), grads_list['encoder'])
                    updates, new_opt_state = self.opt['enc'].update(grads_list['encoder'], self.opt_states['enc'],training_params['encoder'])
                    updated_params = optax.apply_updates(training_params['encoder'], updates)
                    model, _ = self.models['enc']['model']  # Get the model instance
                    self.models['end'] = (model, updated_params)
                    self.opt_states['enc'] = new_opt_state

                epoch_losses.append(loss_value)
            # Compute average loss for the epoch.
            l = jnp.mean(jnp.array(epoch_losses)) #epoch loss
            self.tot_loss.append(float(-l + self.eps))
            epoch_time = timer() - t0
            self.times.append(epoch_time)
            if training_nemot:
                print(f'finished (nemot) epoch {epoch}, loss={-l + self.eps:.5f}, took {epoch_time:.2f} seconds')
            else:
                print(f'finished (enc) epoch {epoch}, loss={l + self.eps:.5f}, took {epoch_time:.2f} seconds')
                self.evaluate_linear_classifier(dataloader)

    
    def eval(self, dataloader, models):
        """
        evaluation of the contrastive learning model.
        use the model to obtain the embeddings of the data and train a linear classifier on them. (all of them?)
        """
        pass
    
    # @partial(jax.jit, static_argnums=(0,))  # JIT compile the loss function
    def _new_loss_fn(self, params, x):
        """
        loss fn operation:
        1. apply encoders
        2. apply phi models to Xs 
        3. calc nemot loss  
        """
        phi_all = []
        enc_model = self.models['enc']['model']
        for i in range(self.k):
            x_i = enc_model.apply(params['enc'], x[:,:,i])
            model, _ = self.models['nemot']['model'][i]  # model instance
            phi = model.apply(params['nemot'][i], x_i) # (b, )
            phi_all.append(jnp.expand_dims(phi,axis=-1))

        e_term = self.calc_exp_term(phi_all, x)
        loss = -(sum([p.mean() for p in phi_all]) - self.eps * e_term)
        return loss , loss

    def calc_exp_term(self, phis, x):
        reduced_phi = sum(phis)
        if self.cost_graph == 'full':
            diffs = jnp.expand_dims(x, axis=-1) - jnp.expand_dims(x, axis=-2)
            cost = 0.5 * jnp.linalg.norm(diffs, axis=1) ** 2
            e_term = jnp.exp((reduced_phi - cost.sum(axis=(1, 2))) / self.eps)
        elif self.cost_graph == 'circle':
            shifted_x = jnp.roll(x, shift=1, axis=-1)
            diffs = x - shifted_x
            cost = jnp.linalg.norm(diffs, axis=1) ** 2
            e_term = jnp.exp((reduced_phi - cost.sum(axis=(-1))) / self.eps)
        else:
            raise ValueError(f"Unknown cost_graph: {self.cost_graph}")
        return jnp.mean(e_term)
    
    def evaluate_linear_classifier(self, dataloader):
        """
        Evaluate the performance of a linear classifier on the encoded data.
        
        Args:
            dataloader: The dataloader for the dataset.
        """
        embeddings = []
        labels = []

        # Get embeddings for the entire dataset
        for images, lbls in dataloader:
            images = self.models['enc']['model'][0].gen_k_views(images)  # generate k predetermined views of the data
            images, lbls = self.torch_to_jax(images, lbls)  # map to jax format

            if self.params.dataset == 'mnist':
                images = images.reshape(images.shape[0], images.shape[1], -1)  # flatten the images

            encodings = self.models['enc']['model'][0].apply(self.models['enc']['model'][1], images)
            embeddings.append(encodings)
            labels.append(lbls)

        embeddings = jnp.concatenate(embeddings, axis=0)
        labels = jnp.concatenate(labels, axis=0)

        # k_ind = 0
        # embeddings = embeddings[:, k_ind, :]    # work with the encoding of a specific view

        # Flatten embeddings
        embeddings = embeddings.reshape(embeddings.shape[0], -1)

        # Split data into train and test sets
        split_idx = int(0.8 * len(embeddings))
        X_train, X_test = embeddings[:split_idx], embeddings[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]

        # Train a linear classifier
        classifier = LogisticRegression()
        classifier.fit(np.array(X_train), np.array(y_train))

        # Evaluate the classifier
        y_pred = classifier.predict(np.array(X_test))
        accuracy = accuracy_score(np.array(y_test), y_pred)

        print(f"Linear classifier accuracy: {accuracy:.4f}")

        

        