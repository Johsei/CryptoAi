import os
import pandas as pd
import numpy as np
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

class TradingEnvironment:
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.portfolio = {'balance': 10000, 'crypto_held': 0}
        self.position = 0
        self.orders = []
        self.episode_orders = []

    def reset(self):
        self.portfolio = {'balance': 10000, 'crypto_held': 0}
        self.position = 0
        self.orders = []
        self.episode_orders = []
        return self._next_observation()

    def _next_observation(self):
        end = self.position + self.window_size
        obs = np.array(self.data.iloc[self.position:end][['open', 'high', 'low', 'close', 'volume']]).reshape(1, self.window_size, 5)
        portfolio_info = np.array([[self.portfolio['balance'], self.portfolio['crypto_held'], 0, 0, 0]]).reshape(1, 1, 5)
        obs = np.append(obs, portfolio_info, axis=1)
        return obs

    def step(self, action):
        current_price = self.data.iloc[self.position]['close']
        crypto_amount = 0
        reward = 0
        
        if action == 0:  # Halten
            pass
        elif action == 1:  # Kaufen
            crypto_amount = self.portfolio['balance'] / current_price
            self.portfolio['balance'] -= crypto_amount * current_price
            self.portfolio['crypto_held'] += crypto_amount
        elif action == 2:  # Verkaufen
            crypto_amount = self.portfolio['crypto_held']
            self.portfolio['balance'] += crypto_amount * current_price
            self.portfolio['crypto_held'] = 0
        
        self.orders.append((action, current_price, crypto_amount))
        self.episode_orders.append(self.orders.copy())
        
        reward = self.portfolio['balance'] + self.portfolio['crypto_held'] * current_price
        self.position += 1
        
        done = self.position >= len(self.data) - 1
        next_obs = self._next_observation()
        
        return next_obs, reward, done

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.state_size, 5)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

""" 

import os
import pandas as pd
import numpy as np
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

class TradingEnvironment:
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.portfolio = {'balance': 10000, 'crypto_held': 0}
        self.position = 0
        self.orders = []
        self.episode_orders = []

    def reset(self):
        self.portfolio = {'balance': 10000, 'crypto_held': 0}
        self.position = 0
        self.orders = []
        self.episode_orders = []
        return self._next_observation()

    #def _next_observation(self):
    #    end = self.position + self.window_size
    #    #obs = np.array(self.data[self.position:end][['open', 'high', 'low', 'close', 'volume']])
    #    obs = np.array(self.data[self.position:end][['open', 'high', 'low', 'close', 'volume']]).reshape(1, self.window_size, 5)
    #    obs = np.append(obs, [self.portfolio['balance'], self.portfolio['crypto_held']])
    #   return obs

    def _next_observation(self):
        end = self.position + self.window_size
        obs = np.array(self.data[self.position:end][['open', 'high', 'low', 'close', 'volume']]).reshape(1, self.window_size, 5)
        obs = np.append(obs, [[self.portfolio['balance'], self.portfolio['crypto_held']]], axis=1)
        return obs

    def step(self, action):
        current_price = self.data.loc[self.position, 'Close']
        crypto_amount = 0
        reward = 0
        
        if action == 0:  # Halten
            pass
        elif action == 1:  # Kaufen
            crypto_amount = self.portfolio['balance'] / current_price
            self.portfolio['balance'] -= crypto_amount * current_price
            self.portfolio['crypto_held'] += crypto_amount
        elif action == 2:  # Verkaufen
            crypto_amount = self.portfolio['crypto_held']
            self.portfolio['balance'] += crypto_amount * current_price
            self.portfolio['crypto_held'] = 0
        
        self.orders.append((action, current_price, crypto_amount))
        self.episode_orders.append(self.orders.copy())
        
        reward = self.portfolio['balance'] + self.portfolio['crypto_held'] * current_price
        self.position += 1
        
        done = self.position >= len(self.data) - 1
        next_obs = self._next_observation()
        
        return next_obs, reward, done

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        #model.add(LSTM(64, input_shape=(self.state_size,)))
        model.add(LSTM(64, input_shape=(self.state_size, 5)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0])
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
 """