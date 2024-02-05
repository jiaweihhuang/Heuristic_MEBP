import numpy as np
import pickle, os
import argparse
import random
from Algorithm import ModelElimination
from multiprocessing import Pool
from copy import deepcopy


def get_percentile(data):
    ret = []
    for i in range(10):
        ret.append(
            np.percentile(data, i * 10 + 5)
        )
    return ret

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    file_name = 'Preparation_S{}_A{}_H{}_dphi{}_dpsi{}_M{}_ModelSeed{}_Scale{}.pickle'.format(args.S, args.A, args.H, args.d_phi, args.d_psi, args.M_size, args.model_seed, args.scale)
    
    if not os.path.exists(file_name):
        assert 0 == 1, 'Should run preparartion first'

    with open(file_name, 'rb') as f:
        data = pickle.load(f)

    record = data['Record']
    Model_Class = data['Model_Class']

    record_feasible = record
    M_feasible = Model_Class

    record_percentil = get_percentile(record)

    for i in range(len(Model_Class)):
        Model_Class[i].set_index(i)

    ### Step 1: Construction of Model Class (randomly pick one of them as the true model)
    M_star_index = np.random.randint(len(Model_Class))
    M_star = Model_Class[M_star_index]
    print('M_star_index is ', M_star_index)

    epsilon = args.epsilon
    model_err_threshold = args.epsilon / 10

    log = {
        'iter': [],
        '#traj': [],
        'maximal_gap': [],
        'ratio': [],
        'M_hat_gap': [],
        'M_hat_ratio': [],
        'M_size': [],
        'Branch': [],
    }

    ### Step 2: Main Algorithm
    finish = False
    M_hat = None
    for k in range(args.K):
        ### evaluation
        # randomly pick a model from remaining feasible models and evaluate the gap of its NE policy

        threshold = np.log(args.H * len(Model_Class) * args.T * (k + 1) / args.delta)
        max_gap = 0.0
        gap_list = []
        count = 0
        
        for M in M_feasible:
            gap, _ = M_star.compute_NE_gap(M.pi_NE)
            max_gap = max(gap, max_gap)
            gap_list.append(gap)
            count += 1

        log['iter'].append(k)
        log['#traj'].append((k+1) * args.T * args.H * 2)
        if k == 0:
            baseline = max_gap
            ratio = 1.0
            
            M_hat_gap = max_gap
            M_hat_ratio = 1.0
        else:
            if max_gap <= epsilon:
                ratio = 0
            else:
                ratio = max_gap / baseline

            M_hat_gap = gap_list[M_hat]
            if M_hat_gap <= epsilon:
                M_hat_ratio = 0
            else:
                M_hat_ratio = M_hat_gap / baseline

        log['maximal_gap'].append(max_gap)
        log['ratio'].append(ratio)

        log['M_hat_gap'].append(M_hat_gap)
        log['M_hat_ratio'].append(M_hat_ratio)
        log['M_size'].append(len(M_feasible))

        print('ratio ', log['ratio'])
        print('M_hat_ratio ', log['M_hat_ratio'])
        print('M_size ', log['M_size'])
        print('M_hat_gap ', log['M_hat_gap'])
        print('maximal_gap ', log['maximal_gap'])
        print('record_percentil', record_percentil)
        print('Branches', log['Branch'])

        if len(M_feasible) == 1:
            break

        ### Step 2 - 1: if-branch in algorithm, search if there are policy such that models are scattered
        find = False
        num_M = record_feasible.shape[0]

        num_neighbor = np.sum(record_feasible < model_err_threshold, axis=1)
        num_neighbor_CM = np.max(num_neighbor, axis=1)
        scatter_index = np.where(num_neighbor_CM < num_M / 2)[0]

        print('maximal number of neigbors', np.max(num_neighbor_CM))

        # if len(scatter_index) > 0, (If Branch in Alg in paper), randomly pick a policy for elimination
        # otherwise, (Else-Branch in Alg in paper) search the model having the maximal number of neigborhoods conditioning on its NE policy
        if len(scatter_index) > 0:
            find = True
            pi_ref = M_feasible[np.random.choice(scatter_index)].pi_NE

            # log['Branch'].append('If')
        else:
            find = False
            num_neighbor = []
            for i in range(num_M):
                num_neighbor.append(np.sum(record_feasible[i][i] < model_err_threshold))

            M_index = np.argmax(num_neighbor)
            pi_ref = M_feasible[M_index].pi_NE
            # log['Branch'].append('Else')

        ### Step 3: run the elimination algorithm
        # update feasible models with ModelElimination
        print('threshold', threshold)
        M_feasible, M_hat = ModelElimination(M_star, M_feasible, pi_ref, args.T, threshold, epsilon)
        print(M_hat)
        M_feasible_index = [M.index for M in M_feasible]
        record_feasible = record[M_feasible_index,:,:][:,M_feasible_index,:][:,:,M_feasible_index]

        # if "not find" and M is not eliminated, then NE is the policy we recommend;
        # otherwise, continue with the model class after elimination
        if not find and M_index in M_feasible_index:
            gap, _ = M_star.compute_NE_gap(pi_ref)
            print('M_index ', M_index, 'M_star_index ', M_star_index)
            print('Gap ', gap)
            finish = True
            
        if finish:
            print('Finished!!!')

    print('comparison between model identification ', M_star.index, M_feasible[0].index)

    log_dir = 'Preparation_S{}_A{}_H{}_dphi{}_dpsi{}_M{}_Scale{}_T{}'.format(args.S, args.A, args.H, args.d_phi, args.d_psi, args.M_size, args.scale, args.T, args.model_seed)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'ModelSeed{}_Seed{}.pickle'.format(args.model_seed, args.seed)), 'wb') as f:
        pickle.dump(log, f)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-K', type = int, default = 1000, help='training iteration')
    parser.add_argument('-S', type = int, default = 100, help='number of states')
    parser.add_argument('-A', type = int, default = 50, help='number of actions')
    parser.add_argument('-H', type = int, default = 3, help='H')
    parser.add_argument('--d-phi', type = int, default = 5, help='linear dimension for state-action feature phi')
    parser.add_argument('--d-psi', type = int, default = 5, help='linear dimension for next state feature psi')
    parser.add_argument('--M-size', type = int, default = 200, help='size of function class')

    parser.add_argument('-T', type = int, default = 50, help='number iteration in model elimination algorithm')

    parser.add_argument('--seed', type = int, nargs='+', default = [1000], help='seed')
    parser.add_argument('--model-seed', type = int, nargs='+', default = [1000], help='seed')
    parser.add_argument('--debug', default = False, action='store_true', help='whether to debug')
    
    parser.add_argument('--delta', type = float, default = 0.001, help='confidence level')

    parser.add_argument('--scale', type = float, default = 0.1, help='scale for random perturbation')
    parser.add_argument('--epsilon', type = float, default = 0.001, help='Quality of NE')

    args = parser.parse_args()
 
    return args
 
if __name__ == '__main__':
    args = get_parser()
    args_list = []
    for model_seed in args.model_seed:
            for seed in args.seed:
                args_copy = deepcopy(args)
                args_copy.model_seed = model_seed
                args_copy.seed = seed
                args_list.append(args_copy)

    with Pool(processes=len(args_list), maxtasksperchild=1) as p:
            p.map(main, args_list, chunksize=1)