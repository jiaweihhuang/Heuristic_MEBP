import numpy as np


'''
Compute and find the pi_exe maximizing E_{pi_exe, M1(pi_ref)}[\| M1(pi_ref) - M2(pi_ref) \|_1].
Basic idea, use model diff as reward, and find the optimal policy.
'''
def Evaluate_Maximal_Model_Diff_given_RefPolicy(M1, M2, pi_ref):
    M1_mu_list, M1_trans = M1.compute_density(pi_ref)
    M2_mu_list, M2_trans = M2.compute_density(pi_ref)

    reward = []
    for h in range(M1.H):
        # compute the current density
        if h == 0:
            mu = M1.mu_1
        else:
            mu = M1.compute_next_density(mu, pi_ref, h - 1)

        diff = np.sum(np.abs(M1_trans[h] - M2_trans[h]), axis=1, keepdims=True)      # vector [SA,1]

        reward.append(diff)

    # Compute the adversarial policy maximizing the expected discrepancy (over M1)
    pi_adv_M1_M2 = []
    Q = np.zeros([M1.S, M1.A])
    V = np.zeros([M1.S, 1])
    for h in reversed(range(M1.H)):
        Q = (reward[h] + M1_trans[h]@V).reshape([M1.S, M1.A])

        V = np.max(Q, axis=1, keepdims=True)
        pi_adv_h = np.argmax(Q, axis=1).astype(np.int32)

        pi_adv_M1_M2.insert(0, pi_adv_h)

        assert V.shape[0] == M1.S
    d_M1_M2_pi_ref = np.sum(M1.mu_1 * V.squeeze())


    # Compute the adversarial policy maximizing the expected discrepancy (over M2)
    pi_adv_M2_M1 = []
    Q = np.zeros([M2.S, M2.A])
    V = np.zeros([M2.S, 1])
    for h in reversed(range(M2.H)):
        Q = (reward[h] + M2_trans[h]@V).reshape([M2.S, M2.A])

        V = np.max(Q, axis=1, keepdims=True)
        pi_adv_h = np.argmax(Q, axis=1).astype(np.int32)

        pi_adv_M2_M1.insert(0, pi_adv_h)
        assert V.shape[0] == M2.S
    d_M2_M1_pi_ref = np.sum(M2.mu_1 * V.squeeze())

    return pi_adv_M1_M2, d_M1_M2_pi_ref, pi_adv_M2_M1, d_M2_M1_pi_ref


def Evaluate_Maximal_Model_Diff(M1, M2):
    M1_mu_list, M1_trans = M1.mu_list, M1.trans_list
    M2_mu_list, M2_trans = M2.mu_list, M2.trans_list

    reward = []
    for h in range(M1.H):
        diff = np.sum(np.abs(M1_trans[h] - M2_trans[h]), axis=1, keepdims=True)      # vector [SA,1]

        reward.append(diff)

    # Compute the adversarial policy maximizing the expected discrepancy (over M1)
    Q = np.zeros([M1.S, M1.A])
    V = np.zeros([M1.S, 1])
    for h in reversed(range(M1.H)):
        Q = (reward[h] + M1_trans[h]@V).reshape([M1.S, M1.A])

        V = np.max(Q, axis=1, keepdims=True)

        assert V.shape[0] == M1.S
    d_M1_M2_pi_ref = np.sum(M1.mu_1 * V.squeeze())


    # Compute the adversarial policy maximizing the expected discrepancy (over M2)
    Q = np.zeros([M2.S, M2.A])
    V = np.zeros([M2.S, 1])
    for h in reversed(range(M2.H)):
        Q = (reward[h] + M2_trans[h]@V).reshape([M2.S, M2.A])

        V = np.max(Q, axis=1, keepdims=True)

        assert V.shape[0] == M2.S
    d_M2_M1_pi_ref = np.sum(M2.mu_1 * V.squeeze())

    return None, d_M1_M2_pi_ref, None, d_M2_M1_pi_ref


def compute_maximal_distance(M_class, pi_ref, epsilon):
    # once find a pi, M, M' with error larger than epsilon, then return.
    M_size = len(M_class)
    for j in range(M_size):
        Mj = M_class[j]
        for k in range(j):
            Mk = M_class[k]
            pi_adv_Mj_Mk, d_Mj_Mk_pi_ref, pi_adv_Mk_Mj, d_Mk_Mj_pi_red = Evaluate_Maximal_Model_Diff_given_RefPolicy(Mj, Mk, pi_ref)

            if d_Mj_Mk_pi_ref > epsilon:
                return pi_adv_Mj_Mk, d_Mj_Mk_pi_ref
            if d_Mk_Mj_pi_red > epsilon:
                return pi_adv_Mk_Mj, d_Mk_Mj_pi_red
            
    # otherwise, all the models are agree with each other.
    return M_class[0].pi_NE, epsilon

def ModelElimination(M_star, M_class, pi_ref, T, threshold, epsilon):
    M_feasible = M_class

    data = [[] for h in range(M_star.H)]

    for M in M_feasible:
        M.compute_density_and_set_model(pi_ref)


    for t in range(T):
        pi_adv, error = compute_maximal_distance(M_feasible, pi_ref, epsilon)

        if error <= epsilon:
            print(len(M_feasible), error, [M.index for M in M_feasible])
            break
        
        # Generate Data
        for h in range(M_star.H):
            s_h, a_h, ns_h, r_h = M_star.generate_traj(pi_adv, pi_ref)[h]
            s_h_, a_h_, ns_h_, r_h_ = M_star.generate_traj(pi_ref, pi_ref)[h]

            data[h] = [(s_h, a_h, ns_h), (s_h_, a_h_, ns_h_)]

        # # Compute log-likelihood
        LLE = np.array(
            [M_feasible[i].compute_accum_log_likelihood(data, pi_ref, use_trans=True) for i in range(len(M_feasible))]
        )
        
        M_hat = np.argmax(LLE)
        MLE = LLE[M_hat]

        M_feasible = [M_feasible[i] for i in np.where(LLE > MLE - threshold)[0]]
        LLE = [LLE[i] for i in np.where(LLE > MLE - threshold)[0]]

    print('gap is ', np.max(LLE) - np.min(LLE))
    
    return M_feasible, np.argmax(LLE)