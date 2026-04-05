import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrafficEnv(gym.Env):
    """
    Traffic Signal Control Environment — v2
    ========================================
    Grid   : 2x2 (4 intersections)
    Layout :
        (0,0) ---E/W--- (0,1)
          |                |
         N/S              N/S
          |                |
        (1,0) ---E/W--- (1,1)

    Each intersection has 4 queues: North, South, East, West
    Directions are indexed: N=0, S=1, E=2, W=3

    ACTION
    ------
    For each intersection the agent picks ONE direction to give a green light.
    Green = cars in that direction can move through the intersection.
    Action space: MultiDiscrete([4, 4, 4, 4])
      → one value per intersection, each value is 0/1/2/3 (N/S/E/W)

    CAR FLOW
    ---------
    When direction D gets a green light at intersection I:
      - Up to `max_pass` cars are cleared from queue I[D]
      - Those cars travel to the NEIGHBORING intersection's opposite queue
        e.g. East green at (0,0) → cars arrive at West queue of (0,1)
      - If the destination is the grid boundary (no neighbor), cars exit the grid

    Neighbor map:
      N from (r,c) → arrives at S of (r-1, c)   [exits if r==0]
      S from (r,c) → arrives at N of (r+1, c)   [exits if r==1]
      E from (r,c) → arrives at W of (r, c+1)   [exits if c==1]
      W from (r,c) → arrives at E of (r, c-1)   [exits if c==0]

    OBSERVATION
    -----------
    Car counts at every direction of every intersection.
    Shape: (4 intersections × 4 directions,) = 16 values
    All values are in [0, max_cars]

    REWARD
    ------
    Negative total waiting cars across all queues.
    The agent learns to minimize congestion.

    EPISODE
    -------
    Runs for max_steps steps, then truncates.
    """

    # Direction indices
    N, S, E, W = 0, 1, 2, 3

    # Opposite direction (cars arriving from direction D come in from the opposite side)
    OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}  # N↔S, E↔W

    # Neighbor intersection for each direction from grid position (row, col)
    # Returns None if it's a boundary (car exits grid)
    def _neighbor(self, row, col, direction):
        if direction == self.N:
            return (row - 1, col) if row > 0 else None
        if direction == self.S:
            return (row + 1, col) if row < 1 else None
        if direction == self.E:
            return (row, col + 1) if col < 1 else None
        if direction == self.W:
            return (row, col - 1) if col > 0 else None

    def __init__(self, max_cars=20, max_pass=3, max_steps=100):
        super().__init__()

        self.n_intersections = 4
        self.n_directions = 4
        self.max_cars = max_cars      # max queue length per direction
        self.max_pass = max_pass      # cars that pass per green light per step
        self.max_steps = max_steps

        # Action: pick 1 of 4 directions to go green, for each intersection
        self.action_space = spaces.MultiDiscrete([4, 4, 4, 4])

        # Observation: car counts, shape (16,)
        self.observation_space = spaces.Box(
            low=0,
            high=self.max_cars,
            shape=(self.n_intersections * self.n_directions,),
            dtype=np.int32
        )

        # State: shape (4, 4) → [intersection_idx, direction]
        # intersection index = row * 2 + col
        self.state = None
        self.current_step = 0

    def _idx(self, row, col):
        """Convert (row, col) to flat intersection index."""
        return row * 2 + col

    def _rowcol(self, idx):
        """Convert flat intersection index to (row, col)."""
        return divmod(idx, 2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start each queue with a small random number of waiting cars
        self.state = self.np_random.integers(
            low=0, high=5,
            size=(self.n_intersections, self.n_directions),
            dtype=np.int32
        )
        self.current_step = 0
        return self.state.flatten(), {}

    def step(self, action):
        """
        action: array of 4 ints, one per intersection
                each int is 0=N, 1=S, 2=E, 3=W
        """
        new_state = self.state.copy()

        # --- 1. Process green lights: move cars through intersections ---
        for i in range(self.n_intersections):
            row, col = self._rowcol(i)
            green_dir = int(action[i])

            # How many cars actually pass (can't pass more than are waiting)
            cars_passing = min(self.state[i, green_dir], self.max_pass)

            if cars_passing == 0:
                continue

            # Remove cars from the green queue
            new_state[i, green_dir] -= cars_passing

            # Send those cars to the neighboring intersection (if it exists)
            neighbor = self._neighbor(row, col, green_dir)
            if neighbor is not None:
                n_row, n_col = neighbor
                n_idx = self._idx(n_row, n_col)
                arriving_dir = self.OPPOSITE[green_dir]
                # Clip so we don't exceed max_cars at the neighbor
                new_state[n_idx, arriving_dir] = min(
                    new_state[n_idx, arriving_dir] + cars_passing,
                    self.max_cars
                )
            # else: cars exit the grid (boundary) — just disappear

        # --- 2. Add new cars arriving randomly at each queue ---
        arrivals = self.np_random.integers(
            low=0, high=4,
            size=(self.n_intersections, self.n_directions),
            dtype=np.int32
        )
        new_state = np.clip(new_state + arrivals, 0, self.max_cars)
        self.state = new_state

        # --- 3. Compute reward: negative total waiting cars ---
        reward = -float(self.state.sum())

        # --- 4. Check termination ---
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        return self.state.flatten(), reward, terminated, truncated, {}

    def render(self):
        """Print a readable view of the current grid state."""
        dir_names = ["N", "S", "E", "W"]
        print(f"\n{'='*50}")
        print(f"Step: {self.current_step}   Total waiting: {self.state.sum()}")
        print(f"{'='*50}")
        for row in range(2):
            for col in range(2):
                idx = self._idx(row, col)
                queues = "  ".join(
                    f"{dir_names[d]}:{self.state[idx, d]}"
                    for d in range(self.n_directions)
                )
                print(f"  Intersection ({row},{col}): {queues}")
            if row == 0:
                print()


# -------------------------------------------------------
# Quick validation — run this file directly
# -------------------------------------------------------
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    print("Running check_env...")
    env = TrafficEnv()
    check_env(env)
    print("✅ check_env passed!\n")

    print("Running 10-step random episode...\n")
    obs, info = env.reset(seed=42)
    env.render()

    dir_names = ["N", "S", "E", "W"]

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        action_str = ", ".join(
            f"({r},{c})→{dir_names[action[env._idx(r,c)]]}"
            for r in range(2) for c in range(2)
        )
        print(f"\nAction: {action_str}")
        print(f"Reward: {reward:.1f}")
        env.render()

        if terminated or truncated:
            print("\nEpisode finished.")
            break
