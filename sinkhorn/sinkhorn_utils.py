import numpy as np
from utils.data_fns import QuadCost, calc_ent
import copy
import random
from timeit import default_timer as timer
import pickle
import os
import wandb
import math


class MOT_Sinkhorn():
    def __init__(self, params, X, MU):
        self.eps = params.eps
        self.eta = 1/self.eps
        self.params = params
        self.k = params.k
        self.mus = MU
        self.n = MU[0].shape[0]
        # self.n = params.n
        self.ns = [params.n]*self.k
        self.tol = 1e-10
        if params.cost_graph == 'circle':
            self.tol = 1e-5
        self.params = params
        # params['cost_graph'] = 'full'
        self.calc_kernel(params['cost_graph'], X)
        self.using_wandb = params['using_wandb']




    def calc_kernel(self, cost_graph, X):
        """
        calculate the kernel.
        X has shape (n,d,k)
        Kernel shape depends on cost graphical structure
        """
        if self.params['alg'] == 'sinkhorn_mot':
            if cost_graph == 'circle':
                # calculate consecutive couples, Each Ki is a matrix Ki = exp(-Ci/eps)
                # Ci = cost(x_i,x_{i+1}), therefore has a shape (n x n)
                # for i in range(self.k):
                C = QuadCost(data=X, mod='circle')
                self.C = C
                K = [np.exp(-c/self.eps) for c in C]


            elif cost_graph == 'full':
                C = QuadCost(data=X, mod='full')
                if self.params.norm_by_k:
                    C = C / self.k
                self.cost = C
                K = np.exp(-C/self.eps)

            elif cost_graph == 'tree':
                # TD
                K=0

        elif self.params['alg'] == 'sinkhorn_gw':
            if cost_graph == 'circle':
                #TD!!!
                # calculate consecutive couples, Each Ki is a matrix Ki = exp(-Ci/eps)
                # Ci = cost(x_i,x_{i+1}), therefore has a shape (n x n)
                C = QuadCost(data=X, mod='circle')
                K = [np.exp(-c / self.eps) for c in C]

            elif cost_graph == 'tree':
                # TD
                K = 0

            elif cost_graph == 'full':
                C = QuadCost(data=X, mod='full')
                self.cost = C
                K = np.exp(-C / self.eps)


        self.kernel = K

    def solve_sinkhorn(self, method='cyclic'):
        """
        eta is regularization parameter
        tol is the error tolerance for stopping
        method is "greedy", "random", or "cyclic"
        """
        eta = self.eta
        tol = self.tol
        naive = self.params.cost_graph == 'full'
        self.tot_loss = []
        self.times = []

        self.phi = [np.zeros(self.n)]*self.k

        assert(method in ['greedy','random','cyclic'])
        t00 = timer()
        curriter = -1
        while curriter < 250:
            t0 = timer()
            curriter += 1
            if curriter % self.k == 0:
                """ Every k iterations check whether we should exit """
                (worsti,worsterr) = self._sinkhorn_worst_error(naive)

                if worsterr < tol:
                    break
            # if curriter % 1000 == 0:
            if curriter % 10 == 0:
                print('Iteration ',curriter)
                print('worsterr',worsterr)
                loss = self.calc_ot_cost(training=True)
                print('OT cost', loss)
                self.tot_loss = [loss]



            """ Pick scaling index """
            scalei = -1

            if method == 'greedy':
                """ Greedy selection rule """
                (worsti, worsterr) = self._sinkhorn_worst_error(eta, p, naive)
                print('worsterr',worsterr)
                if worsterr < tol:
                    continue
                scalei = worsti

            elif method == 'random':
                """ Random selection rule """
                scalei = random.randint(0,self.k-1)

            elif method == 'cyclic':
                """ Cyclic selection rule """
                scalei = curriter % self.k

            else:
                assert(False)

            """ Rescale weights on scalei """

            if naive:
                mi = self.marginalize_naive(scalei)
            else:
                mi = self.marginalize(self.eta, self.phi, scalei)

            ratio = self.mus[scalei] / mi
            self.phi[scalei] = self.phi[scalei] + np.log(ratio) / eta
            # print(f'iteration: {curriter}, took {timer()-t0:.2f} seconds')
            self.times.append(timer() - t0)

        t0 = timer()
        self.phi, self.rankone = self._round_sinkhorn(naive)
        print(f'Rounding step took, took {timer() - t0:.2f} seconds')

        print(f'Finished, n={self.n}, k={self.k}, took {timer() - t00:.2f} seconds')
        return


    def marginalize(self, eta, p, margidx):
        """
        Given weights p = [p_1,\ldots,p_k], and regularization eta > 0,
        Let K = \exp[-C].
        Let d_i = \exp[\eta p_i] for all i \in [k].
        Let P = (d_1 \otimes \dots \otimes d_k) \odot K.
        Return m_{margidx}(P).
        """

        n = self.n
        reg_cost = self.kernel
        reg_loop_cost = self.kernel[self.k-1]

        """ Compute transition matrices 1 --> margidx, margidx --> k """
        trans1 = np.eye(n)
        transk = np.eye(n)

        # trans1[i,j] is the mass of cost of transitioning from i at time 1 to j at time margidx
        # include the potentials in [1,margidx)
        for i in range(margidx):
            currcolscaling = np.diag(np.exp(eta * p[i]))
            trans1 = trans1 @ currcolscaling
            trans1 = trans1 @ reg_cost[i]

        # transk[i,j] is the mass of the cost of transitioning from i at time margidx to j at time K
        # include the potentials in (margidx,k]
        for i in range(margidx+1, self.k):
            transk = transk @ reg_cost[i-1]
            currcolscaling = np.diag(np.exp(eta * p[i]))
            transk = transk @ currcolscaling

        # print('trans1',trans1)
        # print('transk',transk)

        # transk1[i,j] is the mass of the cost of transitioning from i at time margidx to j at time 1 (going through time k and looping back to time 1)
        # includes the potentials in (margidx,k]
        transk1 = transk @ reg_loop_cost

        # For each fixed l, compute trans1[:,l] \dot transk1[l,:].
        # This marginalizes over time 1, but still does not compute the potentials at margidx.
        notscaled = np.diag(transk1 @ trans1)


        scaled = notscaled * np.exp(eta * p[margidx])
        # comparison = self.marginalize_naive(eta, p, margidx)
        # assert(np.all(np.isclose(scaled, comparison)))

        return scaled

    def marginalize_(self, margidx):
        """
        Given weights p = [p_1,\ldots,p_k], and regularization eta > 0,
        Let K = \exp[-C].
        Let d_i = \exp[\eta p_i] for all i \in [k].
        Let P = (d_1 \otimes \dots \otimes d_k) \odot K.
        Return m_{margidx}(P).
        """
        eta = self.eta
        p = self.phi

        n = self.n
        reg_cost = self.kernel
        reg_loop_cost = self.kernel[self.k-1]

        """ Compute transition matrices 1 --> margidx, margidx --> k """
        trans1 = np.eye(n)
        transk = np.eye(n)

        # trans1[i,j] is the mass of cost of transitioning from i at time 1 to j at time margidx
        # include the potentials in [1,margidx)
        for i in range(margidx):
            currcolscaling = np.diag(np.exp(eta * p[i]))
            trans1 = trans1 @ currcolscaling
            trans1 = trans1 @ reg_cost[i]

        # transk[i,j] is the mass of the cost of transitioning from i at time margidx to j at time K
        # include the potentials in (margidx,k]
        for i in range(margidx+1, self.k):
            transk = transk @ reg_cost[i-1]
            currcolscaling = np.diag(np.exp(eta * p[i]))
            transk = transk @ currcolscaling

        # print('trans1',trans1)
        # print('transk',transk)

        # transk1[i,j] is the mass of the cost of transitioning from i at time margidx to j at time 1 (going through time k and looping back to time 1)
        # includes the potentials in (margidx,k]
        transk1 = transk @ reg_loop_cost

        # For each fixed l, compute trans1[:,l] \dot transk1[l,:].
        # This marginalizes over time 1, but still does not compute the potentials at margidx.
        notscaled = np.diag(transk1 @ trans1)


        scaled = notscaled * np.exp(eta * p[margidx])
        # comparison = self.marginalize_naive(eta, p, margidx)
        # assert(np.all(np.isclose(scaled, comparison)))

        return scaled


    def marginalize_naive(self, i, outplan=False):
        # assert(False)
        """
        NAIVE, BRUTE FORCE IMPLEMENTATION OF THE FOLLOWING:
        Given weights p = [p_1,\ldots,p_k], and regularization eta > 0,
        Let K = \exp[-\eta C].
        Let d_i = \exp[\eta p_i].
        Let P = (d_1 \otimes \dots \otimes d_k) \odot K.
        Return m_i(P).
        """
        eta = self.eta
        scaled_cost_tensor = copy.deepcopy(self.kernel)

        for scale_axis in range(self.k):
            dim_array = np.ones((1,scaled_cost_tensor.ndim),int).ravel()
            dim_array[scale_axis] = -1
            p_scale_reshaped = self.phi[scale_axis].reshape(dim_array)
            p_scaling = np.exp(eta * p_scale_reshaped)
            # print(p_scaling.shape)
            scaled_cost_tensor = scaled_cost_tensor * p_scaling
            # print(scaled_cost_tensor)
        if outplan:
            return scaled_cost_tensor
        mi = np.apply_over_axes(np.sum, scaled_cost_tensor, [j for j in range(self.k) if j != i])
        mi = mi.flatten()
        return mi


    def _round_sinkhorn(self, naive=False):
        """ Round sinkhorn solution with a rank-one perturbation """
        p = copy.deepcopy(self.phi)
        eta = self.eta
        for i in range(self.k):
            if naive:
                mi = self.marginalize_naive(i)
            else:
                # mi = self.marginalize_circle(i)
                mi = self.marginalize(self.eta, self.phi, i)
            badratio = self.mus[i] / mi
            minbadratio = np.minimum(1, badratio)
            p[i] = p[i] + np.log(minbadratio) / eta

        rankone = []
        for i in range(self.k):
            if naive:
                mi = self.marginalize_naive(i)
            else:
                # mi = self.marginalize_circle(i)
                mi = self.marginalize(self.eta, self.phi, i)
            erri = self.mus[i] - mi
            # COMMENTED THIS ASSERT!
            # assert(np.all(erri >= -1e-10))
            erri = np.maximum(0,erri)
            if i > 0 and np.sum(np.abs(erri)) > 1e-8:
                rankone.append(erri /  np.sum(np.abs(erri)))
            else:
                rankone.append(erri)

        return p, rankone

    def _sinkhorn_worst_error(self,naive=False):
        """
        Compute the worst error of any marginal. Used for the termination
        condition of solve_sinkhorn.
        """
        worsti = -1
        worsterr = -1
        for i in range(self.k):
            if naive:
                mi = self.marginalize_naive(i)
            else:
                mi = self.marginalize(self.eta, self.phi, i)
            erri = np.sum(np.abs(mi - self.mus[i]))
            if erri > worsterr:
                worsti = i
                worsterr = erri
        return (worsti, worsterr)

    def calc_ot_cost(self, training=False):
        """
        currently implemeneted for full cost structures
        :param training:
        :return:
        """
        if self.params.cost_graph == 'circle':
            # Unregularized OT cost for Neural MOT plan.
            if training:
                return 0
            P = self.calc_plan(training=False)
            cost = sum([np.sum(c*p) for (c, p) in zip(self.C, P)])
            return cost
        # FULL ENTROPIC OT COST
        P = self.calc_plan(training)
        KL = sum([calc_ent(mu) for mu in self.mus]) - calc_ent(P)
        return np.sum(P*self.cost) + self.eps*KL

    def calc_plan(self, training):
        if self.params.cost_graph == 'circle':
            P = self.circle_plan()
            return P
        else:
            P = self.marginalize_naive(0, outplan=True)
            if training:
                return P
            for index in  range(1,len(self.rankone)):
                if index == 1:
                    rankone = np.tensordot(self.rankone[index-1], self.rankone[index], axes=0)
                else:
                    rankone = np.tensordot(rankone, self.rankone[index], axes=0)
            return P + rankone
        #     #### OLD:
        #     # Create a shape of length k with 1s except at the index position
        #     shape = [1] * self.k
        #     shape[index] = -1
        #     reshapred_rankone.append(self.rankone[index].reshape(shape))
        #     P = P * (vec.reshape(shape))
        # R = sum(reshapred_rankone)
        # return P*np.exp(-self.kernel/self.eps)+R

    def circle_plan(self):
        otmaps = []
        for i in range(0, self.k):
            print(f'Extracting map {i}')
            otmaps.append(self.get_pairwise_marginal(self.eta, self.phi, self.rankone, i, (i + 1)%self.k ))
            print('Map extracted')
        return otmaps


    def get_pairwise_marginal_(self, i1, i2):
        pcopy = copy.deepcopy(self.phi)
        marg2 = np.zeros((self.ns[i1],self.ns[i2]))
        # print(np.sum(m1))
        # print(np.prod([np.sum(rankone[i]) for i in range(self.k)]))
        # assert(False)
        for j1 in range(self.ns[i1]):
            pcopy[i1] = np.ones(self.ns[i1]) * -math.inf
            pcopy[i1][j1] = self.phi[i1][j1]
            m2 = self.marginalize(i2)
            marg2[j1,:] = m2
        rankonefactor = 1
        for i in range(self.k):
            if i not in [i1, i2]:
                rankonefactor *= np.sum(self.rankone[i])
        for j1 in range(self.ns[i1]):
            for j2 in range(self.ns[i2]):
                marg2[j1,j2] += self.rankone[i1][j1] * self.rankone[i2][j2] * rankonefactor
        return marg2

    def get_pairwise_marginal_old(self, eta, p, rankone, i1, i2):
        pcopy = copy.deepcopy(p)
        marg2 = np.zeros((self.ns[i1],self.ns[i2]))
        # print(np.sum(m1))
        # print(np.prod([np.sum(rankone[i]) for i in range(self.k)]))
        # assert(False)
        for j1 in range(self.ns[i1]):
            pcopy[i1] = np.ones(self.ns[i1]) * -math.inf
            pcopy[i1][j1] = p[i1][j1]
            m2 = self.marginalize(eta, pcopy, i2)
            marg2[j1,:] = m2
        rankonefactor = 1
        for i in range(self.k):
            if i not in [i1, i2]:
                rankonefactor *= np.sum(rankone[i])
        for j1 in range(self.ns[i1]):
            for j2 in range(self.ns[i2]):
                marg2[j1,j2] += rankone[i1][j1] * rankone[i2][j2] * rankonefactor
        return marg2

    def get_pairwise_marginal(self, eta, p, rankone, i1, i2):
        marg2 = np.zeros((self.ns[i1], self.ns[i2]))
        pcopy = np.array(p, copy=True)
        for j1 in range(self.ns[i1]):
            pcopy[i1, :] = -np.inf
            pcopy[i1, j1] = p[i1][j1]
            m2 = self.marginalize(eta, pcopy, i2)
            marg2[j1, :] = m2

        rankonefactor = np.prod([np.sum(rankone[i]) for i in range(self.k) if i not in [i1, i2]])

        marg2 += np.outer(rankone[i1], rankone[i2]) * rankonefactor

        return marg2

    def save_results(self, X=None):
        tot_loss = np.mean(self.tot_loss[-10:])
        avg_time = np.mean(self.times)
        data_to_save = {
            'avg_loss': tot_loss,
            'avg_time': avg_time,
            'tot_loss': self.tot_loss,
            'times': self.times,
            'params': self.params
            }
        if self.params.calc_ot_cost:
            ot_cost = self.calc_ot_cost()
        else:
            ot_cost=0
        # Save path
        path = os.path.join(self.params.figDir, 'results.pkl')

        # Saving the data using pickle
        with open(path, 'wb') as file:
            pickle.dump(data_to_save, file)

        if self.using_wandb:
            wandb.log({'tot_loss': tot_loss,
                       'avg_time': avg_time,
                       'ot_cost': ot_cost
                       })

        txt_path = os.path.join(self.params.figDir, 'results_summary.txt')
        with open(txt_path, 'w') as txt_file:
            txt_file.write(f"Total Loss: {tot_loss}\n")
            txt_file.write(f"Average Time: {avg_time}\n")
        print(f'Finished run, loss is {tot_loss:.5f}, average epoch time is {avg_time:.3f} seconds, OT cost is {ot_cost:.5f}')


    def calc_emot_from_phi(self):
        phi = self.phi
        c = self.C
        n = phi[0].shape[0]
        k = len(phi)
        eps = self.eps
        e_term = np.eye(n)

        for i in range(k):
            L = np.exp((0.5 * (phi[i] + phi[(i + 1) % k].T) - c[i]) / eps)
            e_term = (e_term @ L) * (1 / n)

        exp_term = np.trace(e_term)

        emot = sum(phi).mean() - eps*exp_term + eps