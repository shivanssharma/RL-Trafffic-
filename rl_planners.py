import numpy as np

class ValueIterationPlanner:
    """
    Planner that uses the Value Iteration algorithm to determine the optimal policy for a given environment.

    Args:
        env (gym.Env): The traffic environment.
        gamma (float): Discount factor for future rewards.
        theta (float): Threshold for stopping value iteration (determines convergence).
    """
    def __init__(self, env, gamma=0.1, theta=1e-2):
        self.env = env
        # print('This is the environment -> ' + str(env))
        self.gamma = gamma
        self.theta = theta
        sizes = [self.env.max_cars_dir + 1, self.env.max_cars_dir + 1, 2]
        self.all_states = np.indices(sizes).reshape(len(sizes), -1).T
        self.state_to_index = {tuple(state): index for index, state in enumerate(self.all_states)}
        self.policy = np.random.choice([0, 1], size=np.array(sizes).prod())  # initialize policy arbitrarily
        self.value_function = np.zeros(np.array(sizes).prod())
        self.value_iteration()

    def value_iteration(self):
        """
        Perform value iteration to compute the optimal value function and policy.
        """
        while True:
            delta = 0
            Q = np.zeros((len(self.all_states), self.env.action_space.n), dtype=np.float64)
            # iterate over all states
            for state_index, state_array in enumerate(self.all_states):
                state = tuple(state_array)
                # iterate over all actions
                for a in range(self.env.action_space.n):
                    # calculate the expected value of taking action `a` in state `state`
                    for prob, next_state, reward, done in self.env.P[state][a]:
                        next_state_index = self.state_to_index[tuple(next_state)]
                        # print('Reward = ' + str(reward))
                        Q[state_index, a] += prob * (reward + self.gamma * self.value_function[next_state_index] * (not done))
                # update the value function with the best action value
                best_action_value = np.max(Q[state_index])
                delta = max(delta, np.abs(self.value_function[state_index] - best_action_value))
                self.value_function[state_index] = best_action_value
            # check for convergence
            if delta < self.theta:
                break
        # extract the policy from the Q-table
        for state_index in range(len(self.all_states)):
            self.policy[state_index] = np.argmax(Q[state_index])

        print(self.policy)

    def choose_action(self, state):
        """
        Select the action based on the learned policy.

        Args:
            state (tuple): The current state of the environment.

        Returns:
            int: The action chosen by the policy.
        """
        state_index = self.state_to_index[tuple(state)]
        return self.policy[state_index]


class PolicyIterationPlanner:
    """
    Planner that uses the Policy Iteration algorithm to determine the optimal policy for a given environment.

    Args:
        env (gym.Env): The traffic environment.
        gamma (float): Discount factor for future rewards.
        theta (float): Threshold for stopping policy evaluation.
    """

    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        sizes = [self.env.max_cars_dir + 1, self.env.max_cars_dir + 1, 2]
        self.all_states = np.indices(sizes).reshape(len(sizes), -1).T
        self.state_to_index = {tuple(state): index for index, state in enumerate(self.all_states)}
        self.policy = np.random.choice([0, 1], size=np.array(sizes).prod())  # initialize policy arbitrarily
        self.value_function = np.zeros(np.array(sizes).prod())
        self.policy_iteration()

    def evaluate_policy(self):
        """
        Evaluate the current policy by computing the value function for all states.

        Returns:
            None
        """
        pass

    def improve_policy(self):
        """
        Improve the current policy by making it greedy with respect to the current value function.

        Returns:
            bool: True if the policy is stable (i.e., no changes were made), False otherwise.
        """
        pass

    def policy_iteration(self):
        """
        Perform policy iteration by alternately evaluating and improving the policy.

        Returns:
            np.ndarray: The final policy after convergence.
        """
        return self.policy

    def choose_action(self, state):
        """
        Select the action based on the learned policy.

        Args:
            state (tuple): The current state of the environment.

        Returns:
            int: The action chosen by the policy.
        """
        pass
