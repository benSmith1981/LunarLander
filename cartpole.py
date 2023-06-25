import gym
from stable_baselines3 import A2C

# Create the CartPole environment
env = gym.make('CartPole-v1',
               render_mode="human")

# Create and train the A2C agent
agent = A2C('MlpPolicy', env)
agent.learn(total_timesteps=10000)

# Evaluate the trained agent
mean_reward, _ = evaluate_policy(agent, env, n_eval_episodes=10)

# Print the mean reward achieved by the agent
print("Mean reward:", mean_reward)


# import gym
# import random
# import numpy as np
# import tensorflow as tf

# from collections import deque
# print("Gym:", gym.__version__)
# print("Tensorflow:", tf.__version__)

# env_name = "CartPole-v0"
# env = gym.make(env_name,
#             render_mode="human")
# print("Observation space:", env.observation_space)
# print("Action space:", env.action_space)

# from tensorflow.keras import layers

# class QNetwork():
#     def __init__(self, state_dim, action_size):
#         self.state_in = tf.keras.Input(shape=state_dim)
#         self.action_in = tf.keras.Input(shape=(), dtype=tf.int32)
#         self.q_target_in = tf.keras.Input(shape=())
#         action_one_hot = tf.one_hot(self.action_in, depth=action_size)
        
#         self.hidden1 = layers.Dense(100, activation=tf.nn.relu)(self.state_in)
#         self.q_state = layers.Dense(action_size, activation=None)(self.hidden1)
#         self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1)
        
#         self.loss = tf.reduce_mean(tf.square(self.q_state_action - self.q_target_in))
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        

#     def update_model(self, session, state, action, q_target):
#         feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target}
#         session.run(self.optimizer, feed_dict=feed)
        
#     def get_q_state(self, session, state):
#         q_state = session.run(self.q_state, feed_dict={self.state_in: state})
#         return q_state
    

# class ReplayBuffer():
#     def __init__(self, maxlen):
#         self.buffer = deque(maxlen=maxlen)
        
#     def add(self, experience):
#         self.buffer.append(experience)
        
#     def sample(self, batch_size):
#         sample_size = min(len(self.buffer), batch_size)
#         samples = random.choices(self.buffer, k=sample_size)
#         return map(list, zip(*samples))
    
# class DQNAgent():
#     def __init__(self, env):
#         self.state_dim = env.observation_space.shape
#         self.action_size = env.action_space.n
#         self.q_network = QNetwork(self.state_dim, self.action_size)
#         self.replay_buffer = ReplayBuffer(maxlen=10000)
#         self.gamma = 0.97
#         self.eps = 1.0
        
#         # self.sess = tf.Session()
#         self.sess = tf.compat.v1.Session()
#         self.sess.run(tf.global_variables_initializer())
        
#     def get_action(self, state):
#         q_state = self.q_network.get_q_state(self.sess, [state])
#         action_greedy = np.argmax(q_state)
#         action_random = np.random.randint(self.action_size)
#         action = action_random if random.random() < self.eps else action_greedy
#         return action
    
#     def train(self, state, action, next_state, reward, done):
#         self.replay_buffer.add((state, action, next_state, reward, done))
#         states, actions, next_states, rewards, dones = self.replay_buffer.sample(50)
#         q_next_states = self.q_network.get_q_state(self.sess, next_states)
#         q_next_states[dones] = np.zeros([self.action_size])
#         q_targets = rewards + self.gamma * np.max(q_next_states, axis=1)
#         self.q_network.update_model(self.sess, states, actions, q_targets)
        
#         if done: self.eps = max(0.1, 0.99*self.eps)
    
#     def __del__(self):
#         # self.sess.close()
#         tf.compat.v1.reset_default_graph()



# agent = DQNAgent(env)
# num_episodes = 400

# for ep in range(num_episodes):
#     state = env.reset()
#     total_reward = 0
#     done = False
#     while not done:
#         action = agent.get_action(state)
#         next_state, reward, done, info = env.step(action)
#         agent.train(state, action, next_state, reward, done)
#         env.render()
#         total_reward += reward
#         state = next_state
        
#     print("Episode: {}, total_reward: {:.2f}".format(ep, total_reward))