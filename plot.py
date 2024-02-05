import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def main():
    args = get_parser()
    
    dirs = ['Preparation_S{}_A{}_H{}_dphi{}_dpsi{}_M{}_Scale{}_T{}'.format(args.S, args.A, args.H, args.d_phi, args.d_psi, args.M_size, args.scale, args.T)]

    fig = plt.figure(figsize=(16, 6))

    for d in dirs:
        os.chdir(d)
        files = os.listdir()
        M_size_all_data = []
        NE_Gap_all_data = []
        max_len = 0

        for name in files:
            with open(name, 'rb') as f:
                data = pickle.load(f)
            M_size_all_data.append(data['M_size'])
            NE_Gap_all_data.append(data['ratio'])

            max_len = max(max_len, len(data['M_size']))

        os.chdir('..')
        
    for i in range(len(M_size_all_data)):
        M_size_all_data[i] += [M_size_all_data[i][-1] for _ in range(max_len - len(M_size_all_data[i]))]
        NE_Gap_all_data[i] += [NE_Gap_all_data[i][-1] for _ in range(max_len - len(NE_Gap_all_data[i]))]
        
    M_size_all_data = np.array(M_size_all_data)
    NE_Gap_all_data = np.array(NE_Gap_all_data)
    M_size_avg = np.mean(M_size_all_data, axis=0)
    M_size_std = np.std(M_size_all_data, axis=0) / np.sqrt(M_size_all_data.shape[0] - 1)
    NE_Gap_size_avg = np.mean(NE_Gap_all_data, axis=0)
    NE_Gap_size_std = np.std(NE_Gap_all_data, axis=0) / np.sqrt(NE_Gap_all_data.shape[0] - 1)

    iteration = [i * args.T * 2 for i in range(max_len)]

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)


    ax1.plot(iteration, M_size_avg)
    ax1.fill_between(iteration, M_size_avg - 2 * M_size_std, M_size_avg + 2 * M_size_std, alpha=0.1)
    ax1.set_xlabel('Number of Trajectories', fontsize=20)
    ax1.set_ylabel('Number of Model Candidates', fontsize=20)
    # ax1.set_xticks(fontsize=20)
    ax1.set_xlim(0, np.max(iteration))
    ax1.set_ylim([1, args.M_size * 1.05])
    ax1.set_xticks(iteration[::4], iteration[::4], fontsize=15)
    for label in ax1.get_yticklabels(): 
        label.set_fontsize(15)

    ax2.plot(iteration, NE_Gap_size_avg)
    ax2.fill_between(iteration, NE_Gap_size_avg - 2 * NE_Gap_size_std, NE_Gap_size_avg + 2 * NE_Gap_size_std, alpha=0.1)
    ax2.set_xlabel('Number of Trajectories', fontsize=20)
    ax2.set_ylabel('NE Gap (Normalized)', fontsize=20)
    ax2.set_xlim(0, np.max(iteration))
    ax2.set_ylim(0, 1.0)
    # ax2.set_yticks(iteration[::4], iteration[::4], fontsize=15)
    ax2.set_xticks(iteration[::4], iteration[::4], fontsize=15)
    for label in ax2.get_yticklabels(): 
        label.set_fontsize(15)

    plt.savefig('./Exp.pdf', bbox_inches='tight')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', type = int, default = 100, help='number of states')
    parser.add_argument('-A', type = int, default = 50, help='number of actions')
    parser.add_argument('-H', type = int, default = 3, help='H')
    parser.add_argument('--d-phi', type = int, default = 5, help='linear dimension for state-action feature phi')
    parser.add_argument('--d-psi', type = int, default = 5, help='linear dimension for next state feature psi')
    parser.add_argument('--M-size', type = int, default = 200, help='size of function class')

    parser.add_argument('-T', type = int, default = 50, help='number iteration in model elimination algorithm')

    parser.add_argument('--model-seed', type = int, nargs='+', default = [1000, 2000, 3000, 4000, 5000], help='seed')
    
    parser.add_argument('--scale', type = float, default = 0.1, help='scale for random perturbation')

    args = parser.parse_args()
 
    return args

if __name__ == '__main__':
    main()