import numpy as np
import argparse
import random
import os, pickle
from Environment import LinearTransitionModel, RewardModel, MF_MDP
from Algorithm import Evaluate_Maximal_Model_Diff


'''
This file is to prepare data, including:
(1) model construction
(2) solve NE of each model
(3) store results for further usage
'''

def policy_diff(pi, pi_):
    ret = 0
    for (x, y) in zip(pi, pi_):
        ret += np.sum(np.abs(x - y))

    return ret

def solve_NE_BR_mixture(M):
    min_gap = 10
    pi = []
    for h in range(M.H):
        pi_h = np.random.uniform(size=[M.S, M.A])
        pi_h = pi_h / np.sum(pi_h, axis=1, keepdims=True)
        pi.append(pi_h)
    alpha = 0.02
    for k in range(2000):
        gap, pi_BR = M.compute_NE_gap(pi)

        for h in range(M.H):
            pi[h] = pi[h] * (1 - alpha) + alpha * pi_BR[h]

        min_gap = min(min_gap, gap)

        if gap < 5e-4:
           return pi, gap
    
    return None, min_gap

### Preparation. (1) Compute a NE for each model (2) create a table record the d(M,M'|pi^NE_M) for any (M,M')
def prepare():
    args = get_parser()
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)

    # initial phi_sa and mu_feature
    LinearT = LinearTransitionModel(args.S, args.A, args.H, args.d_phi, args.d_psi, args.scale)
    # all the model share the representation phi_sa and mu_feature
    # but may not share the psi_next_s

    data = {
        'phi_sa': LinearT.phi_sa,
        'mu_feature': LinearT.mu_feature
    }

    data['psi_next_s'] = LinearT.psi_next_s

    # construct shared initial distribution and reward function
    mu_1 = np.abs(np.random.randint(args.S, size=[args.S]))
    mu_1 = mu_1 / np.sum(mu_1)

    R = RewardModel(args.S, args.A, args.H, args.d_phi, LinearT.phi_sa)

    # firstly, generate models by generating random features
    Model_Class = []
    random_generated_size = 1
    # assert args.M_size % random_generated_size == 0

    print('Generating random models')
    count = 0
    while True:
        LinearT_ = LinearTransitionModel(args.S, args.A, args.H, args.d_phi, args.d_psi, args.scale, data)
        M = MF_MDP(args.S, args.A, args.H, mu_1, LinearT_, R)
        
        # solve NE of M
        est_pi_NE, gap = solve_NE_BR_mixture(M)

        if est_pi_NE:
            M.set_NE(est_pi_NE)
            Model_Class.append(M)
        else:
            count += 1
            print('Not converge ', count, gap)
            print('Converged ', len(Model_Class))
        # stop generating model once enough
        if len(Model_Class) == random_generated_size:
            break
    
    print('Generating perturb models')
    # secondly, generate models by perturbing models generated by step 1
    for random_M in Model_Class:
        if len(Model_Class) == args.M_size:
            break
        data = {
            'phi_sa': LinearT.phi_sa,
            'mu_feature': LinearT.mu_feature,
            'psi_next_s': random_M.P.psi_next_s,
        }
        perturb_generated_Model_Class = []
        while True:
            LinearT_ = LinearTransitionModel(args.S, args.A, args.H, args.d_phi, args.d_psi, args.scale, data)
            M = MF_MDP(args.S, args.A, args.H, mu_1, LinearT_, R)

            # solve NE of M
            est_pi_NE, gap = solve_NE_BR_mixture(M)
            
            if est_pi_NE:
                M.set_NE(est_pi_NE)
                perturb_generated_Model_Class.append(M)
            else:
                count += 1
                print('Not converge ', count, gap)
            print('Converged ', len(perturb_generated_Model_Class))

            # stop generating model once enough
            if len(perturb_generated_Model_Class) == (args.M_size - random_generated_size) // random_generated_size:
                break
        Model_Class = Model_Class + perturb_generated_Model_Class

    assert len(Model_Class) == args.M_size

    # Record[i,j,k] = d(M_j, M_k|pi_i)
    Record = np.zeros([args.M_size, args.M_size, args.M_size])
    for i in range(args.M_size):
        pi_ref = Model_Class[i].pi_NE
        for j in range(args.M_size):
            Mj = Model_Class[j]
            Mj.compute_density_and_set_model(pi_ref)
        for j in range(args.M_size):
            print('Compute Policy, Mj ', (i,j))
            Mj = Model_Class[j]
            for k in range(j):
                Mk = Model_Class[k]
                _, d_Mj_Mk_pi_ref, _, d_Mk_Mj_pi_red = Evaluate_Maximal_Model_Diff(Mj, Mk)
                Record[i][j][k] = d_Mj_Mk_pi_ref
                Record[i][k][j] = d_Mk_Mj_pi_red

                if j == 1:
                    print(d_Mj_Mk_pi_ref, d_Mk_Mj_pi_red)

    prepara_data = {
        'Record': Record,
        'Model_Class': Model_Class,
    }

    with open('Preparation_S{}_A{}_H{}_dphi{}_dpsi{}_M{}_ModelSeed{}_Scale{}.pickle'.format(args.S, args.A, args.H, args.d_phi, args.d_psi, args.M_size, args.model_seed, args.scale), 'wb') as f:
        pickle.dump(prepara_data, file=f)



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', type = int, default = 5, help='number of states')
    parser.add_argument('-A', type = int, default = 5, help='number of actions')
    parser.add_argument('-H', type = int, default = 3, help='H')
    parser.add_argument('--d-phi', type = int, default = 5, help='linear dimension for state-action feature phi')
    parser.add_argument('--d-psi', type = int, default = 5, help='linear dimension for next state feature psi')
    parser.add_argument('--M-size', type = int, default = 60, help='size of function class')
    parser.add_argument('--scale', type = float, default = 0.5, help='scale for random perturbation')

    parser.add_argument('--model-seed', type = int, default = 1000, help='seed')
    parser.add_argument('--debug', default = False, action='store_true', help='whether to debug')

    args = parser.parse_args()
 
    return args
 
if __name__ == '__main__':
    prepare()