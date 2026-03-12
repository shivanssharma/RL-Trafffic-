import sys
import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces

from traffic_simulator import TrafficSim
from traffic_simulator import TrafficRenderer

# constants for traffic light actions
RED, GREEN = 0, 1

class TrafficEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, max_cars_dir=20, max_cars_total=30, lambda_ns=2, lambda_ew=2, cars_leaving=5, rewards=None, max_steps=1000): # THEY GET
        """
        Initialize the environment with specified parameters.

        Args:
            max_cars_dir (int): Maximum number of cars allowed in a single direction (north-south or east-west).
            max_cars_total (int): Maximum number of cars allowed in total across both directions.
            lambda_ns (int): Poisson rate parameter for car arrivals in the north-south direction.
            lambda_ew (int): Poisson rate parameter for car arrivals in the east-west direction.
            cars_leaving (int): Number of cars leaving the intersection per timestep.
            rewards (dict): Reward values for different traffic states.
            max_steps (int): Maximum number of steps per episode.
        """

        # set the main parameters
        self.max_cars_dir = max_cars_dir
        self.max_cars_total = max_cars_total
        self.lambda_ns = lambda_ns
        self.lambda_ew = lambda_ew

        # set the rewards function
        self.rewards = rewards

        # setting max number of steps per episode and keeping track
        self.max_steps = max_steps
        self.current_step = 0

        # two states for each direction (N/S and E/W) and one for the traffic light
        self.nS = (self.max_cars_dir + 1) ** 2 * 2  # number of states
        self.nA = 2  # number of actions (0: keep, 1: switch)

        # define action and observation spaces
        self.action_space = spaces.Discrete(self.nA)
        sizes = [self.max_cars_dir + 1, self.max_cars_dir + 1, 2]
        self.observation_space = spaces.MultiDiscrete(sizes)

        # initial state distribution
        self.isd = np.indices(sizes).reshape(len(sizes), -1).T

        # initial state
        """
        random_index = np.random.choice(self.isd.shape[0])
        self.s = tuple(self.isd[random_index]) # (ns,ew,light)
        """
        self.s = (0,0,1)

        # initialize simulator with environment parameters
        self.sim = TrafficSim(max_cars_dir, lambda_ns, lambda_ew, cars_leaving, self.s[0], self.s[1], self.s[2])
        # initialize renderer in human mode
        self.renderer = TrafficRenderer(self.sim, "human")

        # determine the transition probability matrix
        print("Building transition matrix...")
        self.P = self._build_transition_prob_matrix()
        print("Transition matrix built.")

    def _build_transition_prob_matrix(self):
        """Build the transition probability matrix."""
        P = {}
        for ns in range(self.max_cars_dir + 1):
            for ew in range(self.max_cars_dir + 1):
                for light in [RED, GREEN]:
                    state = (ns, ew, light)
                    P[state] = {action: [] for action in range(self.nA)}
                    for action in range(self.nA):
                        transitions = []
                        for appr_ns in range(8):
                            for appr_ew in range(8):
                                # determine the next state based on action
                                next_light = abs(light - action)
                                next_ns, next_ew, prob_next_state = self.sim.get_updated_wait_cars(ns, ew, next_light, appr_ns, appr_ew)
                                # get reward
                                reward = self.get_rewards(next_ns, next_ew, next_light)
                                done = self.is_terminal(next_ns, next_ew)
                                next_state = (next_ns, next_ew, next_light)
                                # collect all transitions for normalization
                                transitions += [(prob_next_state, next_state, reward, done)]
                        # normalize the probabilities to ensure they sum to 1
                        total_prob = sum([t[0] for t in transitions])
                        transitions = [(p / total_prob, s, r, d) for (p, s, r, d) in transitions]
                        # assign the normalized transitions to the state-action pair
                        P[state][action] = transitions
        return P

    def get_rewards(self, ns, ew, light):
        """
        Calculate the reward for a given state.

        Args:
            ns (int): Number of cars in the north-south direction.
            ew (int): Number of cars in the east-west direction.
            light (int): The current state of the traffic light in the north-south direction (0 for red, 1 for green).

        Returns:
            float: The calculated reward based on the given state.
        """
        cfg = self.rewards or {}
        ns = ns
        ew = ew
        total_cars = ns + ew
        is_clear = (total_cars == 0)
        is_over_total = (total_cars >= self.max_cars_total)
        is_over_dir = (ns >= self.max_cars_dir) or (ew >= self.max_cars_dir)
        is_violation = is_over_total or is_over_dir
        clear_reward = cfg.get("clear_reward", 1.0)
        under_bonus = cfg.get("under_bonus", 0.0)
        default_queue_penalty = (1.0 / float(self.max_cars_total)) if self.max_cars_total else 0.0
        queue_penalty = cfg.get("queue_penalty", default_queue_penalty)
        violation_penalty = cfg.get("violation_penalty", 1.0)
        if is_clear:
            return clear_reward
        r = -queue_penalty * total_cars
        if not is_violation:
            r += under_bonus
        else:
            r -= violation_penalty
        return r
        # return 1.0
        # pass

    def is_terminal(self, ns, ew):
        """
        Check if the state is terminal.

        Args:
            ns (int): Number of cars in the north-south direction.
            ew (int): Number of cars in the east-west direction.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        cfg = self.rewards or {}

        ns = ns
        ew = ew
        total_cars = ns + ew

        term_on_clear = cfg.get("terminal_on_clear", False)
        term_on_jam = cfg.get("terminal_on_jam", False)

        if term_on_clear and total_cars == 0:
            return True

        if term_on_jam:
            is_over_total = (total_cars >= self.max_cars_total)
            is_over_dir = (ns >= self.max_cars_dir) or (ew >= self.max_cars_dir)
            if is_over_total or is_over_dir:
                return True

        return False
        pass

    def is_truncated(self):
        """
        Check if the maximum number of steps has been reached.

        Returns:
            bool: True if the maximum number of steps has been reached, False otherwise.
        """
        return self.current_step >= self.max_steps
        pass


    def step(self, action):
        """
        Take a step in the environment based on the action.

        Args:
            action (int): The action to take in the environment.

        Returns:
            tuple: A tuple containing:
                - obs (np.ndarray): The new state after taking the action.
                - r (float): The reward received for taking the action.
                - done (bool): Whether the episode has ended.
                - truncated (bool): Whether the episode was truncated due to reaching the max number of steps.
                - dict: Additional information, such as the probability of the transition.
        """
        self.sim.advance(action)
        ns, ew, light, prob = self.sim.get_world_state()
        self.s = (ns, ew, light)

        self.current_step += 1

        r = self.get_rewards(self.s[0], self.s[1], self.s[2])
        terminated = self.is_terminal(self.s[0], self.s[1])
        truncated = (not terminated) and self.is_truncated()

        obs = np.array(self.s, dtype=np.int64)
        info = {"prob": prob}
        # print(f'prob : {prob}')

        return obs, r, terminated, truncated, info
        pass

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        # randomly select the number of cars in the NS and EW directions and the traffic light state
        random_index = np.random.choice(self.isd.shape[0])
        # set the initial state
        s = self.isd[random_index]
        self.s = tuple(s) # (ns,ew,light)
        # reset simulator object
        self.sim.reset(*self.s)
        self.current_step = 0
        if return_info:
            return s, {}
        return s

    def render(self, close=False):
        """Render the environment."""
        if close and self.renderer:
            if self.renderer:
                self.renderer.close()
            return

        if self.renderer:
            return self.renderer.render(*self.s)
