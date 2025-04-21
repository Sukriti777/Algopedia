import numpy as np
import random
import time
import matplotlib.pyplot as plt
import streamlit as st
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Sorting Environment for Reinforcement Learning
class SortingEnvironment:
    def __init__(self, array_size=10):
        self.array_size = array_size
        self.state = np.random.permutation(self.array_size)
        self.total_steps = 0
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.action_space = [(i, j) for i in range(self.array_size) for j in range(i + 1, self.array_size)]
        self.action_size = len(self.action_space)

    def reset(self, initial_state=None):
        if initial_state is None:
            self.state = np.random.permutation(self.array_size)
        else:
            self.state = np.array(initial_state)
        self.total_steps = 0
        return self._normalize_state(self.state.copy())

    def _normalize_state(self, state):
        return self.scaler.fit_transform(state.reshape(-1, 1)).flatten()

    def is_sorted(self):
        return np.all(self.state[:-1] <= self.state[1:])

    def step(self, action):
        i, j = action
        self.total_steps += 1
        self.state[i], self.state[j] = self.state[j], self.state[i]
        
        # Calculate reward
        if self.is_sorted():
            reward = 100 - self.total_steps
            done = True
        else:
            # Reward based on how many elements are in correct position
            correct_positions = np.sum(np.diff(self.state) >= 0)
            reward = correct_positions / (self.array_size - 1) - 0.5  # Range [-0.5, 0.5]
            done = False
            
        return self._normalize_state(self.state.copy()), reward, done

# DQN Agent for Sorting
class DQNAgent:
    def __init__(self, env, learning_rate=0.001, discount_factor=0.95,
                 exploration_rate=1.0, min_exploration_rate=0.01,
                 exploration_decay_rate=0.995, memory_size=10000, batch_size=32):
        
        self.env = env
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # Exploration parameters
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.discount_factor = discount_factor
        
        # Build neural network
        self.model = self._build_model(env.array_size, env.action_size, learning_rate)
        self.target_model = self._build_model(env.array_size, env.action_size, learning_rate)
        self.update_target_model()

    def _build_model(self, state_size, action_size, learning_rate):
        model = Sequential()
        model.add(Dense(64, input_dim=state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        action_idx = self.env.action_space.index(action)
        self.memory.append((state, action_idx, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.choice(self.env.action_space)
        
        state = np.reshape(state, [1, self.env.array_size])
        act_values = self.model.predict(state, verbose=0)
        action_idx = np.argmax(act_values[0])
        return self.env.action_space[action_idx]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(next_q_values[i])

        self.model.fit(states, targets, epochs=1, verbose=0)

    def train(self, episodes=1000, update_target_every=50):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.replay()
            
            # Decay exploration rate
            self.exploration_rate = max(self.min_exploration_rate, 
                                      self.exploration_rate * self.exploration_decay_rate)
            
            # Update target network periodically
            if episode % update_target_every == 0:
                self.update_target_model()

    def sort(self, initial_state):
        state = self.env.reset(initial_state)
        done = False
        steps = 0
        sorting_process = [state.copy()]
        
        while not done:
            action = self.choose_action(state)
            next_state, _, done = self.env.step(action)
            state = next_state
            steps += 1
            sorting_process.append(state.copy())
            
        return state, steps, sorting_process

# Traditional Sorting Algorithms with Metrics
class SortMetrics:
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
        self.time = 0

def quicksort(arr, metrics=None, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot_index = partition(arr, metrics, low, high)
        quicksort(arr, metrics, low, pivot_index - 1)
        quicksort(arr, metrics, pivot_index + 1, high)
    return arr

def partition(arr, metrics, low, high):
    pivot = arr[high]
    i = low
    for j in range(low, high):
        if metrics:
            metrics.comparisons += 1
        if arr[j] < pivot:
            if metrics:
                metrics.swaps += 1
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    if metrics:
        metrics.swaps += 1
    arr[i], arr[high] = arr[high], arr[i]
    return i

def merge_sort(arr, metrics=None):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        
        merge_sort(left, metrics)
        merge_sort(right, metrics)
        
        i = j = k = 0
        while i < len(left) and j < len(right):
            if metrics:
                metrics.comparisons += 1
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
                if metrics:
                    metrics.swaps += 1
            k += 1
        
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1
    return arr

def builtin_sort(arr):
    return sorted(arr)

# Performance Testing Function
def test_sorting_performance(sort_function, arr, metrics=None):
    arr_copy = arr.copy()
    start_time = time.time()
    
    if metrics:
        metrics.swaps = 0
        metrics.comparisons = 0
        sorted_arr = sort_function(arr_copy, metrics)
    else:
        sorted_arr = sort_function(arr_copy)
    
    end_time = time.time()
    if metrics:
        metrics.time = end_time - start_time
    
    return sorted_arr

# Streamlit App
def main():
    st.title("Sorting Algorithm Visualizer with RL")
    st.write("Compare traditional sorting algorithms with Reinforcement Learning approach")
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    array_size = st.sidebar.slider("Array Size", 5, 15, 10)
    train_episodes = st.sidebar.slider("Training Episodes", 100, 2000, 500)
    
    # Initialize environment and agent
    env = SortingEnvironment(array_size)
    agent = DQNAgent(env)
    
    # Training section
    if st.sidebar.button("Train RL Agent"):
        with st.spinner(f"Training RL Agent with {train_episodes} episodes..."):
            agent.train(episodes=train_episodes)
        st.sidebar.success("Training completed!")
    
    # Array input
    st.subheader("Input Array")
    input_method = st.radio("Input method:", ["Random", "Manual Input", "Preset Patterns"])
    
    if input_method == "Random":
        arr = np.random.permutation(array_size)
    elif input_method == "Manual Input":
        arr = []
        cols = st.columns(array_size)
        for i in range(array_size):
            with cols[i]:
                arr.append(st.number_input(f"Element {i}", value=i, key=f"num_{i}"))
        arr = np.array(arr)
    else:  # Preset Patterns
        pattern = st.selectbox("Select pattern:", 
                              ["Random", "Sorted", "Reverse Sorted", "Nearly Sorted", "Few Unique"])
        if pattern == "Random":
            arr = np.random.permutation(array_size)
        elif pattern == "Sorted":
            arr = np.arange(array_size)
        elif pattern == "Reverse Sorted":
            arr = np.arange(array_size)[::-1]
        elif pattern == "Nearly Sorted":
            arr = np.arange(array_size)
            # Swap a few elements
            for _ in range(max(1, array_size // 5)):
                i, j = random.sample(range(array_size), 2)
                arr[i], arr[j] = arr[j], arr[i]
        else:  # Few Unique
            arr = np.random.choice([1, 5, 10], size=array_size)
    
    st.write("Input array:", arr)
    
    # Sort buttons
    st.subheader("Sorting Algorithms")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("QuickSort"):
            metrics = SortMetrics()
            start_time = time.time()
            sorted_arr = test_sorting_performance(quicksort, arr, metrics)
            end_time = time.time()
            
            st.write("**QuickSort Results**")
            st.write("Sorted array:", sorted_arr)
            st.write(f"Time: {(end_time - start_time):.6f} seconds")
            st.write(f"Comparisons: {metrics.comparisons}")
            st.write(f"Swaps: {metrics.swaps}")
    
    with col2:
        if st.button("MergeSort"):
            metrics = SortMetrics()
            start_time = time.time()
            sorted_arr = test_sorting_performance(merge_sort, arr, metrics)
            end_time = time.time()
            
            st.write("**MergeSort Results**")
            st.write("Sorted array:", sorted_arr)
            st.write(f"Time: {(end_time - start_time):.6f} seconds")
            st.write(f"Comparisons: {metrics.comparisons}")
            st.write(f"Swaps: {metrics.swaps}")
    
    with col3:
        if st.button("Built-in Sort"):
            start_time = time.time()
            sorted_arr = test_sorting_performance(builtin_sort, arr)
            end_time = time.time()
            
            st.write("**Built-in Sort Results**")
            st.write("Sorted array:", sorted_arr)
            st.write(f"Time: {(end_time - start_time):.6f} seconds")
    
    with col4:
        if st.button("RL Sort"):
            start_time = time.time()
            sorted_arr, steps, sorting_process = agent.sort(arr)
            end_time = time.time()
            
            st.write("**RL-Based Sort Results**")
            st.write("Sorted array:", sorted_arr)
            st.write(f"Time: {(end_time - start_time):.6f} seconds")
            st.write(f"Steps: {steps}")
            
            # Visualization of sorting process
            st.subheader("RL Sorting Process")
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, state in enumerate(sorting_process):
                ax.plot(state, label=f'Step {i}')
            ax.set_title("Array State During RL Sorting")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
    
    # Performance comparison
    if st.button("Compare All Algorithms"):
        results = []
        algorithms = [
            ("QuickSort", lambda x: test_sorting_performance(quicksort, x, SortMetrics())),
            ("MergeSort", lambda x: test_sorting_performance(merge_sort, x, SortMetrics())),
            ("Built-in Sort", lambda x: test_sorting_performance(builtin_sort, x)),
            ("RL Sort", lambda x: agent.sort(x)[:2])
        ]
        
        metrics_data = []
        for name, func in algorithms:
            start_time = time.time()
            if name in ["QuickSort", "MergeSort"]:
                metrics = SortMetrics()
                sorted_arr = func(arr.copy())
                end_time = time.time()
                metrics_data.append({
                    "Algorithm": name,
                    "Time (s)": end_time - start_time,
                    "Comparisons": metrics.comparisons,
                    "Swaps": metrics.swaps
                })
            elif name == "RL Sort":
                sorted_arr, steps = func(arr.copy())
                end_time = time.time()
                metrics_data.append({
                    "Algorithm": name,
                    "Time (s)": end_time - start_time,
                    "Steps": steps,
                    "Comparisons": "N/A",
                    "Swaps": "N/A"
                })
            else:
                sorted_arr = func(arr.copy())
                end_time = time.time()
                metrics_data.append({
                    "Algorithm": name,
                    "Time (s)": end_time - start_time,
                    "Comparisons": "N/A",
                    "Swaps": "N/A"
                })
        
        # Display metrics table
        st.subheader("Performance Comparison")
        st.table(metrics_data)
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        algorithms = [x["Algorithm"] for x in metrics_data]
        times = [x["Time (s)"] for x in metrics_data]
        
        ax.bar(algorithms, times)
        ax.set_title("Sorting Algorithm Performance Comparison")
        ax.set_ylabel("Time (seconds)")
        ax.set_xlabel("Algorithm")
        st.pyplot(fig)

if __name__ == "__main__":
    main()