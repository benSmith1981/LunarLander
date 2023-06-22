import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
gym.logger.set_level(40)  # errors only
path_to_my_model = 'cartpole-model.h5'

class Agent:
    """
    Encapsulates a primitive, 2 hidden layer neural network
    """
    def __init__(self, observation_space, action_space):
        OBSERVATION_SPACE_DIM = observation_space
        ACTION_SPACE_DIM = action_space
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.Input(shape=(OBSERVATION_SPACE_DIM,)))
        self.model.add(tf.keras.layers.Dense(24, activation='relu'))
        self.model.add(tf.keras.layers.Dense(24, activation='relu'))
        self.model.add(tf.keras.layers.Dense(ACTION_SPACE_DIM,activation='linear'))
        self.model.compile(loss=tf.keras.losses.mse, optimizer=Adam())
        self.model.summary()

    def load(self, model_path):
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print("Loaded model from disk")
        else:
            print("No model found. Starting training from scratch.")

    def train(self, x: list, y: list):
        """
        Trains NN on a sample of [state] -> [action_values] data
        :param x: list of observations in a roll-out
        :param y: list of rewards for every action in each observation
        :return:
        """
        x_arr = np.array(x)
        y_arr = np.array(y)

        self.model.fit(x_arr, y_arr, batch_size=50, epochs=5, verbose=0)

    def predict(self, x):
        x_arr = np.expand_dims(np.array(x), 0)
        return self.model.predict(x_arr, verbose=0)

def main():
    env = gym.make("CartPole-v1",
                   render_mode='human')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    n_epochs = 800
    env.reset()

    # Create a `Sequential` model and add a Dense layer as the first layer.
    agent = Agent(state_size,action_size)
    agent.load(path_to_my_model)  # Load existing model if it exists

    # Hyper-parameters
    epsilon = 0.3  # epsilon greedy
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.6

    # start the training
    episode_rewards = []

    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.draw()
    replay_buffer_len = 10e3
    replay_buffer = list()
    rewards = list()
    for episode in range(n_epochs):
        ax.clear()

        episode_reward = 0
        state, info = env.reset()
        # if episode % 10 == 0:
        #     env.render()
        while True:
            y_pred = agent.predict(state)[0]
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(y_pred)

            state, reward, done, truncated, info = env.step(action)
            y_pred[action] = reward
            # y_pred[action] = reward if done else reward + gamma * np.max(agent.model.predict(state)[0])

            replay_buffer.append(state)
            rewards.append(y_pred)
            episode_reward += reward
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            if done or truncated:
                episode_rewards.append(episode_reward)
                if len(replay_buffer) > replay_buffer_len:
                    print("Training the agent")
                    agent.train(replay_buffer, rewards)
                    replay_buffer = list()
                    rewards = list()
                print(f"Episode {episode} reward: {episode_reward}")
                break

        im = ax.plot(range(len(episode_rewards)), episode_rewards)
        fig.canvas.flush_events()
        # Save the modelx
        agent.model.save(path_to_my_model)

    plt.ioff()
    plt.show()

    env.close()


# Create a function to preprocess the state
def preprocess_state(state):
    return np.expand_dims(state, axis=0)

def run_trained_model():
    env = gym.make("CartPole-v1",
                render_mode='human')

    # Load the trained model
    loaded_model = tf.keras.models.load_model(path_to_my_model)

    state, _ = env.reset()

    done = False
    while not done:
        env.render()
        # Preprocess the state
        processed_state = preprocess_state(state)

        # Use the model to predict the action
        action = np.argmax(loaded_model.predict(processed_state)[0])

        state, reward, done, truncated, info = env.step(action)

    env.close()

if __name__ == '__main__':
    main()
    # run_trained_model()


