import numpy as np
import torch

def gen_data(params):
    if params['data_dist'] == 'uniform':
        if params['euler'] == 1:
            '''
            generate euler flow samples - n evenly spaces samples along [0,1]
            '''
            X = [np.linspace(0, 1, params['n'], dtype=np.float32).reshape(params['n'], 1).astype(np.float32)]*params['k']
        else:
            # generate k samples which are d-dimensional with n samples (from Taos's notebook)
            X = []
            for i in range(params['k']):
                X.append(np.random.uniform(-1/np.sqrt(params['dims'][i]),1/np.sqrt(params['dims'][i]),(params['n'],params['dims'][i])).astype(np.float32))


        if params['alg'] not in ('ne_gw','ne_mot'):
            X = np.stack(X, axis=-1)
            MU = [(1 / params['n']) * np.ones(params['n'])]*params['k']
            return X, MU

        X = torch.from_numpy(np.stack(X, axis=-1))

        return X


def QuadCost(data, mod='circle', root=None):
    k = data.shape[-1]
    n = data.shape[0]
    d = data.shape[1]
    if mod == 'circle':
        differences = []
        if k>2:
            if isinstance(data, np.ndarray):
                for i in range(k):
                    # x = data[:,:,i]
                    # y = data[:,:,(i + 1) % k]
                    # differences.append(np.linalg.norm(x[:,None] - y[None,:], axis=-1) ** 2)
                    ####
                    # Extract vectors for variables i and j
                    vectors_i = data[:, :, i][:, np.newaxis, :]
                    vectors_j = data[:, :, (i+1) % k][np.newaxis, :, :]
                    # Compute the norm of the vector differences
                    vector_diffs = vectors_i - vectors_j
                    norms = np.linalg.norm(vector_diffs, axis=2) ** 2  # Compute norms along the vector dimension
                    differences.append(norms)
            else:
                for i in range(k):
                    # Extract vectors for variables i and i+1
                    vectors_i = data[:, :, i].unsqueeze(1)  # Adding dimension using unsqueeze
                    vectors_j = data[:, :, (i+1)%k].unsqueeze(0)  # Adding dimension using unsqueeze

                    # Compute the norm of the vector differences
                    vector_diffs = vectors_i - vectors_j
                    norms = torch.norm(vector_diffs, dim=2) ** 2  # Compute norms along the vector dimension
                    differences.append(norms)


        else:
            x = data[:, :, 0]
            y = data[:, :, 1]
            differences.append(torch.norm(x[:, None] - y[None, :], dim=-1) ** 2)
        # differences = [torch.norm(data[:, :, i] - data[:, :, (i + 1) % k], dim=1)**2 for i in range(k)]
    elif mod == 'tree':
        differences = QuadCostTree(data, root)
    elif mod == 'full':
        # calculate all pairwise quadratic losses
        ###
        # option 1 - through broadcasting:
        # Expand 'data' to (n, d, k, k) by repeating it across new dimensions
        # data_expanded = data.unsqueeze(3).expand(-1, -1, -1, k)
        # data_t_expanded = data.unsqueeze(2).expand(-1, -1, k, -1)
        #
        # # Compute differences using broadcasting (resulting shape will be (n, d, k, k))
        # differences = data_expanded - data_t_expanded
        #
        # # Compute norms (resulting shape will be (n, k, k))
        # differences = torch.norm(differences, dim=1)
        ###
        # option 2 - via a nested loop (doesnt use tensor operations but performs half the computations)
        # pairwise_norms = torch.zeros((n, k, k))
        # for i in range(k):
        #     for j in range(i + 1, k):
        #         pairwise_norms[:, i, j] = torch.norm(data[:, :, i] - data[:, :, j], dim=1)
        # differences += pairwise_norms.transpose(1, 2)
        ###
        if isinstance(data, np.ndarray):
            # CLASSIC ALG
            differences = np.zeros([n] * k)
            for i in range(k):
                for j in range(i + 1, k):
                    # Extract vectors for variables i and j
                    vectors_i = data[:, :, i][:, np.newaxis, :]
                    vectors_j = data[:, :, j][np.newaxis, :, :]

                    # Compute the norm of the vector differences
                    vector_diffs = vectors_i - vectors_j
                    norms = np.linalg.norm(vector_diffs, axis=2)**2  # Compute norms along the vector dimension

                    # Prepare to broadcast norms into the tensor
                    # Create an array of 1s with length k for reshaping
                    broadcast_shape = [1] * k
                    broadcast_shape[i] = n
                    broadcast_shape[j] = n
                    norms_reshaped = norms.reshape(broadcast_shape)

                    # Sum the broadcasted norms into the tensor
                    differences += norms_reshaped
        else:
            # NE alg
            differences = torch.zeros([n] * k).to(data.device)
            for i in range(k):
                for j in range(i + 1, k):
                    # Extract vectors for variables i and j
                    vectors_i = data[:, :, i].unsqueeze(1)  # Adding dimension using unsqueeze
                    vectors_j = data[:, :, j].unsqueeze(0)  # Adding dimension using unsqueeze

                    # Compute the norm of the vector differences
                    vector_diffs = vectors_i - vectors_j
                    norms = torch.norm(vector_diffs, dim=2)**2  # Compute norms along the vector dimension

                    # Prepare to broadcast norms into the tensor
                    # Create an array of 1s with length k for reshaping
                    broadcast_shape = [1] * k
                    broadcast_shape[i] = n
                    broadcast_shape[j] = n
                    norms_reshaped = norms.reshape(broadcast_shape)

                    # Sum the broadcasted norms into the tensor
                    differences += norms_reshaped
    elif mod == 'euler':
        '''
        Calculate Euler Flows cost graph.
        The Euler cost is defined as:
        c(x_1...x_k) = \|\sigma(x_1) - x_k\|^2 + \sum_{i=1}^k\|x_{i+1}-x_i\|^2
        '''
        differences = []
        if isinstance(data, np.ndarray):
            pass
        else:
            for i in range(k-1):
                vectors_i = data[:, :, i].unsqueeze(1)
                vectors_j = data[:, :, (i + 1) % k].unsqueeze(0)

                # Compute the norm of the vector differences
                vector_diffs = vectors_i - vectors_j
                norms = torch.norm(vector_diffs, dim=2) ** 2  # Compute norms along the vector dimension
                differences.append(norms)

            vectors_i = data[:, :, k-1].unsqueeze(1)
            vectors_j = EulerSigma(data[:, :, 0].unsqueeze(0))
            vector_diffs = vectors_i - vectors_j
            norms = torch.norm(vector_diffs, dim=2) ** 2  # Compute norms along the vector dimension
            differences.append(norms)

    return differences


def QuadCostTree(X, root):
    """
    Calculate the (k-1) squared L2 distance matrices for each non-root node in the tree.

    Parameters:
    - X: torch.Tensor of shape (n, d, k), where n is the number of samples, d is the dimension of each vector, and k is the number of nodes.
    - root: The root of the tree, which is a Node object with children.

    Returns:
    - A list of torch.Tensor matrices, each of shape (n, n), representing the squared L2 distance for each non-root node.
    """
    n, d, k = X.shape
    matrices = [0]*k

    def traverse_and_calculate(node):
        # If the node is not the root, calculate the squared L2 distance matrix
        if not node.is_root_flag:
            parent_index = node.parent_index   # Adjust for zero-indexing in Python
            node_index = node.index   # Adjust for zero-indexing in Python

            # Efficient broadcasting-based computation for squared L2 norms
            # # Implementation where C[i,j]=x[i]-parent[j]
            # vectors_i = X[:, :, node_index].unsqueeze(1)  # (n, 1, d)
            # vectors_j = X[:, :, parent_index].unsqueeze(0)  # (1, n, d)

            # Implementation where C[i,j]=parent[i]-x[j]
            vectors_i = X[:, :, node_index].unsqueeze(0)  # (n, 1, d)
            vectors_j = X[:, :, parent_index].unsqueeze(1)  # (1, n, d)

            # Compute the squared L2 distance between all pairs using broadcasting
            vector_diffs = vectors_i - vectors_j  # (n, n, d)
            C_i = torch.norm(vector_diffs, dim=2) ** 2  # (n, n), squared L2 norms

            # Store the matrix for this node
            matrices[node_index] = C_i

        # Traverse to the children recursively
        for child in node.children:
            traverse_and_calculate(child)

    # Traverse the tree starting from the root and calculate the matrices
    traverse_and_calculate(root)

    # CURRENTLY FOR ROOT AT idx=0
    return matrices

def kronecker_product(vectors):
    for index in range(1, len(vectors)):
        if index == 1:
            out = np.tensordot(vectors[index - 1], vectors[index], axes=0)
        else:
            out = np.tensordot(out, vectors[index], axes=0)
    return out

def calc_ent(p):
    return -np.sum(p*np.log(p))


def EulerSigma(data, case=0):
    """
    Applies the Euler Flow sigma displacement function.
    :param data: assumed to be one-dimensional, so shape is (n,d,1)
    :return:
    """
    if case == 0:
        # in this case its x = (x + 0.5) mod 1
        data = (data + 0.5)%1
        return data