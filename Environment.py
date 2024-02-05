import numpy as np
from copy import deepcopy

'''
P: S x A x D(S) -> D(S)
R: S x A x D(S) -> [1, 1/H]
'''
high_val = 10
class TransitionModel:
    def __init__(self, S, A, H):
        self.S = S
        self.A = A
        self.H = H

    def sample(self, state, action, mu, h):
        pass

    def evolve(self, mu, policy, h):
        pass

    def compute_trans(self, state, action, mu, h):
        pass

    def compute_all_trans(self, mu, h):
        # mu x h -> matrix of P_h(s'|s,a,mu)
        pass

class RewardModel:
    def __init__(self, S, A, H, d_phi, phi_sa):
        self.S = S
        self.A = A
        self.H = H

        self.R_sa = []
        for h in range(self.H):
            R_sa = np.random.randint(5, size=[self.S, self.A]) / 5 / self.H
            self.R_sa.append(R_sa)

        self.d_phi = d_phi
        self.phi_sa = phi_sa

        self.S_space = np.array([i for i in range(self.S)])

    def compute_all_rewards(self, mu, h):
        R = -np.log(1e-11) -np.log(mu + 1e-11).reshape([-1, 1])

        R = np.ones([self.S, self.A]) * R
        R = R.reshape([-1, 1])

        assert len(R.shape) == 2
        return R
    
    def compute_reward(self, state, action, mu, h):
        R = self.compute_all_rewards(mu, h).reshape([self.S, self.A])
        return R[state][action].squeeze()
    

class LinearTransitionModel(TransitionModel):
    def __init__(self, S, A, H, d_phi, d_psi, scale, data={'phi_sa': None, 'mu_feature': None, 'psi_next_s': None}):
        super().__init__(S, A, H)
        self.d_phi = d_phi
        self.d_psi = d_psi
        self.scale = scale

        self.construct_sa_feature(data['phi_sa'])
        self.construct_mu_feature(data['mu_feature'])
        self.construct_next_s_feature(data['psi_next_s'])

    def construct_sa_feature(self, data=None):
        if data is not None:
            self.phi_sa = deepcopy(data)
        else:
            self.phi_sa = []
            for h in range(self.H):
                phi_sa_h = np.random.randn(self.S * self.A, self.d_phi)
                self.phi_sa.append(phi_sa_h.reshape([self.S, self.A, self.d_phi]))
        
        return self.phi_sa

    def construct_mu_feature(self, data=None):
        if data is not None:
            self.mu_feature = deepcopy(data)
        else:
            self.mu_feature = []
            for h in range(self.H):
                mu_feature_h = np.random.randn(self.S, self.d_phi, self.d_psi)
                self.mu_feature.append(mu_feature_h.reshape(self.S, -1))
        
        return self.mu_feature

    def construct_next_s_feature(self, data=None):
        if data is not None:
            self.psi_next_s = deepcopy(data)
            assert self.scale > 0
            for h in range(self.H):
                scale = np.random.uniform(low=0.0, high=self.scale)

                perturb_psi_next_s_h = np.random.randn(self.d_psi, self.S) + 1e-8
                self.psi_next_s[h] = self.psi_next_s[h] * (1 - scale) + perturb_psi_next_s_h * scale
        else:
            self.psi_next_s = []
            for h in range(self.H):
                psi_next_s_h = np.random.randn(self.d_psi, self.S) + 1e-8
                self.psi_next_s.append(psi_next_s_h)

        return self.psi_next_s

    def sample(self, state, action, mu, h):
        U = mu.reshape([1, -1]) @ self.mu_feature[h]
        
        P_sa_mu = self.phi_sa[h][state][action].reshape([1, -1]) @ U.reshape([self.d_phi, self.d_psi]) @ self.psi_next_s[h]
        P_sa_mu = np.abs(P_sa_mu)
        P_sa_mu = P_sa_mu / np.sum(P_sa_mu)
        return np.random.choice(np.arange(self.S), p = P_sa_mu.squeeze())

    def evolve(self, mu, pi, h):
        assert h < self.H
        U = mu.reshape([1, -1]) @ self.mu_feature[h]
        P_mu = self.phi_sa[h].reshape([-1, self.d_phi]) @ U.reshape([self.d_phi, self.d_psi]) @ self.psi_next_s[h] # P_mu: SAxS
        
        P_mu = np.abs(P_mu)
        P_mu = P_mu / np.sum(P_mu, axis=1, keepdims=True)

        if len(pi[h].shape) == 2:
            mu_sa = mu.reshape([-1, 1]) * pi[h]
        else:
            mu_sa = np.zeros([self.S, self.A])
            mu_sa[np.arange(self.S), pi[h]] = mu

        next_mu = mu_sa.reshape([1, -1]) @ P_mu

        assert np.abs(np.sum(next_mu) - 1.0) < 1e-6, np.sum(next_mu)
        next_mu = next_mu / np.sum(next_mu)

        return P_mu, next_mu.squeeze()


    # return a matrix SA x S
    def compute_all_trans(self, mu, h):
        U = mu.reshape([1, -1]) @ self.mu_feature[h]
        P_mu = self.phi_sa[h].reshape([-1, self.d_phi]) @ U.reshape([self.d_phi, self.d_psi]) @ self.psi_next_s[h] # SxAxS
        P_mu = np.abs(P_mu)
        P_mu = P_mu / np.sum(P_mu, axis=1, keepdims=True)
        return P_mu
    

class MF_MDP:
    def __init__(self, S, A, H, mu_1, P, R):
        self.S = S
        self.A = A
        self.H = H
        self.mu_1 = mu_1       # vector with shape [self.S]

        # P and R should be a list of TransitionModel and RewardModel classes
        self.P = P
        self.R = R

        self.LLE = 0.0

    def set_reference_policy(self, pi_ref):
        '''
        reference policy determine the density in transition
        '''
        self.pi_ref = pi_ref

    def reset(self,):
        self.state = np.random.choice(np.arange(self.S), p=self.mu_1)
        self.mu = self.mu_1
        return self.state

    def step(self, action, h, use_trans=False):
        next_state = self.P.sample(self.state, action, self.mu, h)
        r = self.R.compute_reward(self.state, action, self.mu, h)
        _, self.mu = self.P.evolve(self.mu, self.pi_ref, h)

        self.state = next_state

        return next_state, r

    def generate_traj(self, pi_exe, pi_ref, use_trans=True):
        '''
        pi_exe: the policy for execution
        pi_ref: the reference policy
        '''
        self.set_reference_policy(pi_ref)

        trajectory = []

        state = self.reset()

        for h in range(self.H):
            if len(pi_exe[h].shape) == 1:
                action = pi_exe[h][state]
            else:
                action = np.random.choice(np.arange(self.A), p=pi_exe[h][state])
            # print(action, pi_exe[h][state])
            next_state, r = self.step(action, h, use_trans=use_trans)
            trajectory.append((state, action, next_state, r))
            state = next_state

        return trajectory

    # pi: list of SxA matrix
    def compute_density(self, pi):
        mu = self.mu_1
        mu_list = []
        trans_list = []

        for h in range(self.H):
            mu_list.append(mu)
            trans, next_mu = self.P.evolve(mu, pi, h)
            trans_list.append(trans)

            mu = next_mu

        return mu_list, trans_list

    def compute_density_and_set_model(self, pi):
        self.mu_list, self.trans_list = self.compute_density(pi)
        

    
    def compute_next_density(self, mu, pi, h):
        _, next_mu = self.P.evolve(mu, pi, h)
        return next_mu
    
    '''
    pi_adv is None: compute BR policy and return it and its value
    pi_adv is not None: compute policy value
    '''
    def compute_value(self, mu_list, pi=None):
        if pi is None:
            pi_BR = []
            
        Q = np.zeros([self.S, self.A])
        V = np.zeros([self.S, 1])
        for h in reversed(range(self.H)):
            mu = mu_list[h]
            R = self.R.compute_all_rewards(mu, h)
            P = self.P.compute_all_trans(mu, h)
            Q = (R + P@V).reshape([self.S, self.A])
            if pi is None:
                V = np.max(Q, axis=1, keepdims=True)

                pi_BR_h_index = np.argmax(Q, axis=1).astype(np.int32)
                pi_BR_h = np.zeros([self.S, self.A])
                pi_BR_h[np.arange(self.S), pi_BR_h_index] = 1.0
                pi_BR.insert(0, pi_BR_h)
            else:
                if len(pi[h].shape) == 1:
                    V = Q[np.arange(self.S), pi[h]].reshape([-1, 1])
                else:
                    V = np.sum(Q * pi[h], axis=1).reshape([-1, 1])

            assert V.shape[0] == self.S
        V = np.sum(self.mu_1 * V.squeeze())

        if pi is None:
            return V, pi_BR
        else:
            return V
    
    def compute_NE_gap(self, pi):
        mu_list, _ = self.compute_density(pi)

        # Compute policy value of the adversarial policy
        V_adv, pi_BR = self.compute_value(mu_list, None)

        # Compute the policy value of pi
        V_pi = self.compute_value(mu_list, pi)

        return V_adv - V_pi, pi_BR
    

    def compute_log_likelihood(self, data, pi_ref):
        mu_list, trans_list = self.compute_density(pi_ref)
        LLE = 0.0
        for h in range(self.H):
            trans = trans_list[h]
            data_h = data[h]

            for (s,a,ns) in data_h:
                LLE += np.log(trans[s * self.A + a][ns] + 1e-8)

        return LLE


    def compute_accum_log_likelihood(self, data, pi_ref, use_trans=False):
        if use_trans:
            trans_list = self.trans_list
        else:
            mu_list, trans_list = self.compute_density(pi_ref)

        LLE = 0.0
        for h in range(self.H):
            trans = trans_list[h]
            data_h = data[h]

            for (s,a,ns) in data_h:
                LLE += np.log(trans[s * self.A + a][ns] + 1e-8)
        self.LLE += LLE
        return self.LLE

    # set the NE policy
    def set_NE(self, pi_NE):
        self.pi_NE = pi_NE

    def set_index(self, index):
        self.index = index

    def set_LLE(self, LLE):
        self.LLE += LLE