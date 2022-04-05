""" 
Solve Nash equilibrium with Mulitplicative Weights Update (gradient-free) in zero-sum games. 

References: 
https://arxiv.org/abs/1807.04252
https://dl.acm.org/doi/abs/10.1145/3219166.3219235?casa_token=4Vu0L6uBun8AAAAA:X2HOzU0od-wd7BhvZMyPl-duY0EZtsWzrJOBqiuE9RS-Son-u2S_pKBWz94kzXFIzu8mtlwt7UM

"""
import numpy as np
import time, copy
import sys

def get_payoff_vector(payoff_matrix, opponent_policy):
    payoff = opponent_policy @ payoff_matrix
    return payoff

def NashEquilibriumMWUSolver(A, Itr=300, verbose=False, adaptive_lr_rate=-1/3):
    """ Solve Nash equilibrium with multiplicative weights udpate."""
    # discount = 0.9
    EPS = 1e-7 # prevent numerical problem
    std = np.std(A)
    A = A/(std+EPS)  # normalization: MWU is sensitive to the scale of matrix value, smaller value should have larger learning rate

    row_action_num = A.shape[0]
    col_action_num = A.shape[1]
    learning_rate = np.sqrt(np.log(row_action_num)/Itr)  # sqrt(log |A| / T)

    row_policy = np.ones(row_action_num)/row_action_num
    col_policy = np.ones(col_action_num)/col_action_num
    policies = np.array([row_policy, col_policy])
    final_policy = copy.deepcopy(policies)

    for i in range(Itr):
        if adaptive_lr_rate and (i+1) % 20 == 0:
            learning_rate = np.power(i, adaptive_lr_rate)
        policies_ = copy.deepcopy(policies)  # track old value before update (update is inplace)
        # for row player, maximizer
        payoff_vec = policies_[1] @ A.T
        policies[0] = policies[0] * np.exp(learning_rate*payoff_vec)

        # for col player, minimizer
        # payoff_vec = get_payoff_vector(A, policies[0])
        payoff_vec = policies_[0] @ A
        policies[1] = policies[1] * np.exp(-learning_rate*payoff_vec)

        # above is unnormalized, normalize it to be distribution
        # for k in range(len(policies)):
        #     abs_policy = np.abs(policies[k])
        #     policies[k] = abs_policy/np.sum(abs_policy)
        policies = policies/np.expand_dims(EPS+np.sum(policies, axis=-1), -1)

        # MWU is average-iterate coverging, so accumulate polices
        final_policy += policies

        # if (i+1) % 100 == 0:
        #     learning_rate *= discount

    final_policy = final_policy / (Itr+1)

    if verbose:
        print(f'For row player, strategy is {final_policy[0]}')
        print(f'For column player, strategy is {final_policy[1]}')
        print(learning_rate)

    nash_value = (final_policy[0] @ A @ final_policy[1].T)*(std+EPS)  # unnormalize to get original values
    return final_policy, nash_value


def NashEquilibriumParallelMWUSolver(A, Itr=300, verbose=False, adaptive_lr_rate=-1/3):
    """ Solve mulitple Nash equilibrium with multiplicative weights udpate."""
    EPS = 1e-7 # prevent numerical problem
    A = np.array(A)
    std = np.std(A)
    A = A/(std+EPS)  # normalization: MWU is sensitive to the scale of matrix value, smaller value should have larger learning rate

    matrix_num = A.shape[0]
    row_action_num = A.shape[1]
    col_action_num = A.shape[2]
    learning_rate = 100*np.sqrt(np.log(row_action_num)/Itr)  # sqrt(log |A| / T)

    row_policy = np.ones(row_action_num)/row_action_num
    col_policy = np.ones(col_action_num)/col_action_num
    policies = np.array(matrix_num*[[row_policy, col_policy]])
    final_policy = copy.deepcopy(policies)

    for i in range(Itr):
        if adaptive_lr_rate and (i+1) % 20 == 0:
            learning_rate = np.power(i, adaptive_lr_rate)
        policies_ = copy.deepcopy(policies)  # track old value before update (update is inplace)
        # for row player, maximizer
        payoff_vec = np.einsum('nb,nab->na', policies_[:, 1], A) 
        policies[:, 0] = policies[:, 0] * np.exp(learning_rate*payoff_vec)

        # for col player, minimizer
        payoff_vec = np.einsum('na,nab->nb', policies_[:, 0], A) 
        policies[:, 1] = policies[:, 1] * np.exp(-learning_rate*payoff_vec)


        # above is unnormalized, normalize it to be distribution
        policies = policies/np.expand_dims(EPS+np.sum(policies, axis=-1), -1)
        # MWU is average-iterate coverging, so accumulate polices
        final_policy += policies

    final_policy = final_policy / (Itr+1)

    if verbose:
        print(f'For row player, strategy is {final_policy[:, 0]}')
        print(f'For column player, strategy is {final_policy[:, 1]}')
        print(learning_rate)
    
    nash_value = np.einsum('nb,nb->n', np.einsum('na,nab->nb', policies[:, 0], A), final_policy[:, 1])*(std+EPS)  # unnormalize to get original values

    return final_policy, nash_value

if __name__ == "__main__":
    ###   TEST LP NASH SOLVER ###
    A = np.array([[0, 2, -1], [-1, 0, 1], [1, -1, 0]])
    As = np.array(4*[[[0, 2, -1], [-1, 0, 1], [1, -1, 0]]])  # multiple game matrix solved at the same time

    # A=np.array([[ 0.594,  0.554,  0.552,  0.555,  0.567,  0.591],
    # [ 0.575,  0.579,  0.564,  0.568,  0.574,  0.619],
    # [-0.036,  0.28,   0.53,   0.571,  0.57,  -0.292],
    # [ 0.079, -0.141, -0.2,    0.592,  0.525, -0.575],
    # [ 0.545,  0.583,  0.585,  0.562,  0.537,  0.606],
    # [ 0.548,  0.576,  0.58,   0.574,  0.563,  0.564]])

    A=np.array([[ 0.001,  0.001,  0.00,     0.00,     0.005,  0.01, ],
    [ 0.033,  0.166,  0.086,  0.002, -0.109,  0.3,  ],
    [ 0.001,  0.003,  0.023,  0.019, -0.061, -0.131,],
    [-0.156, -0.039,  0.051,  0.016, -0.028, -0.287,],
    [ 0.007,  0.029,  0.004,  0.005,  0.003, -0.012],
    [ 0.014,  0.018, -0.001,  0.008, -0.009,  0.007]])

    t0=time.time()
    ne = NashEquilibriumMWUSolver(A, verbose=True)
    print(ne)
    t1=time.time()
    print(t1-t0)

    # print(As, As.shape)
    # ne = NashEquilibriumParallelMWUSolver(As, verbose=True)
    # print(ne)
    # t1=time.time()
    # print(t1-t0)
