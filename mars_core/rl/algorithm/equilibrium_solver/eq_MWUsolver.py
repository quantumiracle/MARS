""" 
Solve Nash equilibrium with Mulitplicative Weights Update (gradient-free) in zero-sum games. 

References: 
https://arxiv.org/abs/1807.04252
https://dl.acm.org/doi/abs/10.1145/3219166.3219235?casa_token=4Vu0L6uBun8AAAAA:X2HOzU0od-wd7BhvZMyPl-duY0EZtsWzrJOBqiuE9RS-Son-u2S_pKBWz94kzXFIzu8mtlwt7UM

"""
import numpy as np
import time, copy

def NashEquilibriumMWUSolver(A, Itr=500, learning_rate=0.5, verbose=False):
    discount = 0.9
    row_action_num = A.shape[0]
    col_action_num = A.shape[1]

    row_policy = np.ones(row_action_num)/row_action_num
    col_policy = np.ones(col_action_num)/col_action_num
    policies = [row_policy, col_policy]
    final_policy = copy.deepcopy(policies)

    def get_payoff_vector(payoff_matrix, opponent_policy):
        payoff = opponent_policy @ payoff_matrix
        return payoff

    for i in range(Itr):
        # for row player, maximizer
        payoff_vec = get_payoff_vector(A.T, policies[1])
        for j in range(policies[0].shape[0]): # iterate over all actions
            denom = np.sum(policies[0] * np.exp(learning_rate*payoff_vec))
            policies[0][j] = policies[0][j]*np.exp(learning_rate*payoff_vec[j])/denom

        # for col player, minimizer
        payoff_vec = get_payoff_vector(A, policies[0])
        for j in range(policies[1].shape[0]): # iterate over all actions
            denom = np.sum(policies[1] * np.exp(-learning_rate*payoff_vec))
            policies[1][j] = policies[1][j]*np.exp(-learning_rate*payoff_vec[j])/denom

        # above is unnormalized, normalize it to be distribution
        for k in range(len(policies)):
            abs_policy = np.abs(policies[k])
            policies[k] = abs_policy/np.sum(abs_policy)

        # MWU is average-iterate coverging, so accumulate polices
        final_policy[0] += policies[0]
        final_policy[1] += policies[1]

        if (i+1) % 100 == 0:
            learning_rate *= discount

    final_policy[0] = final_policy[0]/(Itr+1)  # initial value is not zero
    final_policy[1] = final_policy[1]/(Itr+1)

    if verbose:
        print(f'For row player, strategy is {final_policy[0]}')
        print(f'For column player, strategy is {final_policy[1]}')
        print(learning_rate)

    return final_policy

if __name__ == "__main__":
    ###   TEST LP NASH SOLVER ###
    # A = np.array([[0, -1, 1], [2, 0, -1], [-1, 1, 0]])
    A=np.array([[ 0.594,  0.554,  0.552,  0.555,  0.567,  0.591],
    [ 0.575,  0.579,  0.564,  0.568,  0.574,  0.619],
    [-0.036,  0.28,   0.53,   0.571,  0.57,  -0.292],
    [ 0.079, -0.141, -0.2,    0.592,  0.525, -0.575],
    [ 0.545,  0.583,  0.585,  0.562,  0.537,  0.606],
    [ 0.548,  0.576,  0.58,   0.574,  0.563,  0.564]])

    # A=np.array([[ 0.001,  0.001,  0.00,     0.00,     0.005,  0.01, ],
    # [ 0.033,  0.166,  0.086,  0.002, -0.109,  0.3,  ],
    # [ 0.001,  0.003,  0.023,  0.019, -0.061, -0.131,],
    # [-0.156, -0.039,  0.051,  0.016, -0.028, -0.287,],
    # [ 0.007,  0.029,  0.004,  0.005,  0.003, -0.012],
    # [ 0.014,  0.018, -0.001,  0.008, -0.009,  0.007]])

    t0=time.time()
    ne = NashEquilibriumMWUSolver(A, verbose=True)
    print(ne)
    t1=time.time()
    print(t1-t0)