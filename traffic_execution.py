import time
import gymnasium as gym
from traffic_environment import TrafficEnv
import rl_planners
import numpy as np
import matplotlib.pyplot as plt

# define rewards function
rewards = {"state": 0}

# initialize the environment with rewards and max_steps
env = TrafficEnv(rewards = rewards, max_steps=1000)

# set the RL algorithm to plan or train an agent
rl_algo = "Value Iteration"

# initialize the agent and train it
if rl_algo == "Value Iteration":
    agent = rl_planners.ValueIterationPlanner(env)
elif rl_algo == "Policy Iteration":
    agent = rl_planners.PolicyIterationPlanner(env)


# TODO: Initialize variables to track performance metrics
# Metrics to include:
# 1. Count of instances where car count exceeds critical thresholds (N total cars or M in any direction)
# 2. Average number of cars waiting at the intersection in all directions during a time period
# 3. Maximum continuous time where car count remains below critical thresholds


# reset the environment and get the initial observation
observation, info = env.reset(seed=42), {}
np.random.seed(42)
env.action_space.seed(42)

# TODO: Initialize variables to track environment metrics
# Example: cumulative rewards, episode duration, etc.

# ── tracking variables (added) ──────────────────────────────────────────
ns_history     = []
ew_history     = []
reward_history = []
# ────────────────────────────────────────────────────────────────────────

# set light state variables
RED, GREEN = 0, 1

# run the environment until terminated or truncated
terminated, truncated = False, False

try:
    while (not terminated and not truncated):
        # use the agent's policy to choose an action
        action = agent.choose_action(observation)
        # step through the environment with the chosen action
        observation, reward, terminated, truncated, info = env.step(action)

        # TODO: Update variables to calculate performance and environment metrics based on the new observation

        # ── collect data each step (added) ──────────────────────────────────
        ns_history.append(int(observation[0]))
        ew_history.append(int(observation[1]))
        reward_history.append(float(reward))
        # ────────────────────────────────────────────────────────────────────

        # unpack the state to get the number of cars and traffic light state
        ns, ew, light = tuple(observation)
        light_color = "GREEN" if light == GREEN else "RED"
        # print the current state
        print(f"Step: x, NS Cars: {ns}, EW Cars: {ew}, Light NS: {light_color}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        # print(f"Step: {_}, NS Cars: {ns}, EW Cars: {ew}, Light NS: {light_color}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        # render the environment at each step
        env.render()
        # add a delay to slow down the rendering for better visualization
        time.sleep(0.1)

        # reset the environment if terminated or truncated
        if terminated or truncated:
            print("\nTERMINATED OR TRUNCATED, RESETTING...\n")

            # TODO: Update metrics for completed episode

            observation, info = env.reset(), {}

            # TODO: Reset tracking variables for the new episode

            terminated, truncated = False, False

except KeyboardInterrupt:
    print("\nInterrupted — saving charts...")

finally:
    # close the environment
    env.render(close=True)

    # TODO: Evaluate performance based on high-level metrics

    print("\n=== PERFORMANCE EVALUATION ===")
    # TODO: Print performance metrics

    # ════════════════════════════════════════════════════════════════════
    #  CHARTS & TABLES  (added below — original code above is unchanged)
    # ════════════════════════════════════════════════════════════════════

    if len(ns_history) == 0:
        print("No data collected — charts skipped.")
    else:
        SAVE_DIR   = r"D:\msc4sem\reinforcement lab\gymTraffic-templates_SSSIHL\gymTraffic-templates"
        steps      = np.arange(1, len(ns_history) + 1)
        total_cars = [n + e for n, e in zip(ns_history, ew_history)]
        CRIT_TOTAL = env.max_cars_total
        CRIT_DIR   = env.max_cars_dir

        
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(steps, reward_history, color="#C44E52", lw=1.0, alpha=0.7, label="Reward")
        if len(reward_history) >= 20:
            w  = max(10, len(reward_history) // 50)
            ma = np.convolve(reward_history, np.ones(w) / w, mode="valid")
            ax.plot(np.arange(w, len(reward_history) + 1), ma, color="#2d2d2d", lw=2.0, label=f"Rolling mean ({w}-step)")
        ax.set_title("Reward Over Time", fontsize=13, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(SAVE_DIR + r"\chart_reward_over_time.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("Saved: chart_reward_over_time.png")

        # ── Figure 3: Average NS / EW cars bar chart ─────────────────────────
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ["NS Cars", "EW Cars", "Total Cars"]
        means  = [np.mean(ns_history), np.mean(ew_history), np.mean(total_cars)]
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        bars   = ax.bar(labels, means, color=colors, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title("Average Cars Waiting (All Steps)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Avg Cars")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(SAVE_DIR + r"\chart_avg_queue_bar.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("Saved: chart_avg_queue_bar.png")


        print(f"\nAll charts saved to {SAVE_DIR}")
