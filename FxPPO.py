# wgledbetter's modifications of ppo examples for the forex application.

# ==============================================================================
## Imports
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
# from plotly import tools as py_tools # For subplotting
import datetime

from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam

import numba as nb

from FxEnv import FxEnv


# ==============================================================================
## Global Variables
# Define
PAIRS = []
PAIRS.append('EUR_USD')
#PAIRS.append('USD_JPY')
PAIRS.append('GBP_USD')
#PAIRS.append('AUD_USD')
#PAIRS.append('USD_CHF')
#PAIRS.append('USD_CAD')
#PAIRS.append('EUR_JPY')
PAIRS.append('EUR_GBP')

MODE = 'Local'
FREQ = 'M5'
DATA_FOLDER_NAME = 'histdata_v3'
TEST_FOLDER_NAME = 'testdata_v3'
MARKET_STATE_HISTORY = 6
DEPOSIT = 1000
LEV = 10
LOT_SIZE = 10

LOSS_CLIPPING = 0.2
EPISODES = 1000
EPOCHS = 10
GAMMA = 0.99
BATCH_SIZE = 256
LR = 1e-4

FX_ACTIVATION = K.softmax
ACTOR_ACTIV = 'tanh'

NUM_LAYERS = 2
#WHERE_LSTM = 0
HIDDEN_SIZE = 200
HIDDEN_ACTIV = 'relu'

# ------------------------------------------------------------------------------
# Calculate
NPAIRS = len(PAIRS)
NUM_ACTIONS = 3*NPAIRS


# ==============================================================================
## Global Functions
def fx_activation(activation=FX_ACTIVATION):
    # The point of this was to try and apply softmax to groups of 3, but that's
        # throwing errors, so I'll just bypass this completely with tanh or
        # something
    def fx_activation_function(x):
        outList = []
        for i in range(NPAIRS):
            vec = x[3*i:3*(i+1)]
            outList.append(activation(vec))

        out = K.stack(outList)
        return out

    return fx_activation_function


# ______________________________________________________________________________
@nb.jit
def fx_prob2act(prob):
    act = np.array([])
    for i, _ in enumerate(PAIRS):
        cmd = np.argmax(prob[3*i:3*(i+1)])
        vec = np.zeros(3)
        vec[cmd] = 1
        act = np.append(act, vec)

    return act


# ______________________________________________________________________________
@nb.jit
def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


# ______________________________________________________________________________
def proximal_policy_optimization_loss(actual_value, predicted_value,
                                      old_prediction):
    advantage = actual_value - predicted_value

    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred)
        old_prob = K.sum(y_true * old_prediction)
        r = prob/(old_prob + 1e-10)

        return -K.log(prob + 1e-10) * K.mean(K.minimum(r * advantage,
                                                       K.clip(r,
                                                              min_value=0.8,
                                                              max_value=1.2
                                                              ) * advantage
                                                       )
                                             )
    return loss


# ==============================================================================
## Agent class definition
class Agent:

    def __init__(self):
        self.env = FxEnv(mode=MODE, pairs=PAIRS, freq=FREQ, deposit=DEPOSIT,
                         data_folder_name=DATA_FOLDER_NAME, lev=LEV,
                         market_state_hist=MARKET_STATE_HISTORY,
                         lot_size=LOT_SIZE
                         )
        self.state_size = self.env.stateSize()
        self.action_size = self.env.actionSize()

        self.critic = self.build_critic()
        self.actor = self.build_actor()

        print(self.action_size, 'action_size', self.state_size, 'state_size')
        self.episode = 0
        self.observation = self.env.reset()
        self.reward = []
        self.reward_over_time = []

        self.dummy_action = np.zeros((1, self.action_size))
        self.dummy_value = np.zeros((1, 1))

    # --------------------------------------------------------------------------
    # Initialization of Networks
    def build_actor(self):
        state_input = Input(shape=(self.state_size,))
        actual_value = Input(shape=(1,))
        predicted_value = Input(shape=(1,))
        old_prediction = Input(shape=(self.action_size,))

        x = Dense(HIDDEN_SIZE, activation=HIDDEN_ACTIV)(state_input)
        for l in range(NUM_LAYERS-1):
            x = Dense(HIDDEN_SIZE, activation=HIDDEN_ACTIV)(x)

        out_actions = Dense(self.action_size, activation=ACTOR_ACTIV,
                            name='output')(x)

        model = Model(inputs=[state_input, actual_value,
                              predicted_value, old_prediction],
                      outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss(
                              actual_value=actual_value,
                              old_prediction=old_prediction,
                              predicted_value=predicted_value)])
        model.summary()

        return model

    # __________________________________________________________________________
    def build_critic(self):
        state_input = Input(shape=(self.state_size,))

        x = Dense(HIDDEN_SIZE, activation=HIDDEN_ACTIV)(state_input)
        for l in range(NUM_LAYERS-1):
            x = Dense(HIDDEN_SIZE, activation=HIDDEN_ACTIV)(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LR), loss='mse')

        return model

    # --------------------------------------------------------------------------
    # Functions for Training
    @nb.jit
    def reset_env(self):
        self.episode += 1
        self.observation = self.env.reset()
        self.reward = []

    # __________________________________________________________________________
    def get_action(self):
        p = self.actor.predict([self.observation.reshape(1, self.state_size),
                                self.dummy_value, self.dummy_value,
                                self.dummy_action])
        action_matrix = fx_prob2act(p[0])
        return action_matrix, p

    # __________________________________________________________________________
    def transform_reward(self):
        self.reward_over_time.append(np.array(self.reward).sum())
        for j in range(len(self.reward)):
            reward = self.reward[j]
            for k in range(j+1, len(self.reward)):
                reward += self.reward[k] * GAMMA**k  # Discounted Reward Func
            self.reward[j] = reward

    # __________________________________________________________________________
    def get_batch(self, batch_size=BATCH_SIZE):
        batch = [[], [], [], []]
        tmp_batch = [[], [], []]
        while len(batch[0]) < batch_size:
            action_matrix, predicted_action = self.get_action()
            observation, reward, done, info = self.env.act(action_matrix)
            self.reward.append(reward)

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                for i in range(len(tmp_batch[0])):
                    obs = tmp_batch[0][i]
                    action_mat = tmp_batch[1][i]
                    pred = tmp_batch[2][i]
                    r = self.reward[i]
                    batch[0].append(obs)
                    batch[1].append(action_mat)
                    batch[2].append(pred)
                    batch[3].append(r)
                tmp_batch = [[], [], []]
                self.reset_env()

        obs = np.array(batch[0])
        action_mat = np.array(batch[1])
        pred = np.array(batch[2])
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        reward = np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        return obs, action_mat, pred, reward

    # __________________________________________________________________________
    def train(self, episodes=EPISODES, batch_size=BATCH_SIZE):
        self.episode = 0
        if MODE == 'Local':
            self.env.acct.mrkt.data_folder_name = DATA_FOLDER_NAME
        self.reset_env()
        while self.episode < episodes:
            print('*******************************')
            print('Episode {}'.format(self.episode))
            obs, action, pred, reward = self.get_batch(batch_size=batch_size)
            old_pred = pred
            pred_values = self.critic.predict(obs)

            for _ in range(EPOCHS):
                self.actor.train_on_batch([obs, reward, pred_values, old_pred],
                                          [action])
            for _ in range(EPOCHS):
                self.critic.train_on_batch([obs], [reward])

    # __________________________________________________________________________
    def test(self, nGames=1, test_folder_name=TEST_FOLDER_NAME):
        # Run the actor on environment and see results
        # Desired outputs for analysis:
            # Market data for that game
            # Trades made
            # Account Valuation over Time
            # Final Open Positions
            # Rewards

        output = []
        if MODE == 'Local':
            # Switch to 2018 dataset
            self.env.acct.mrkt.data_folder_name = test_folder_name

        for g in range(nGames):
            temp_output = {}
            self.reset_env()
            done = False
            while not done:
                action_matrix, predicted_action = self.get_action()
                observation, reward, done, info = self.env.act(action_matrix)
                self.reward.append(reward)
                self.observation = observation

            mkt_data = self.env.acct.mrkt.pair
            trades = self.env.acct.trades
            positions = self.env.acct.positions

            temp_output['data'] = mkt_data
            temp_output['trades'] = trades
            temp_output['value'] = self.env.acct.valuation
            temp_output['positions'] = positions
            temp_output['reward'] = self.reward

            output.append(temp_output)

        return output

    # __________________________________________________________________________
    def load_actor(self, fname):
        # Load a previous actor model
        # Be careful that the environment sizing is the same.
        # ACTUALLY: Doesn't seem to work because of custom loss function
        self.actor = load_model(fname)

    # __________________________________________________________________________
    def load_critic(self, fname):
        # Load a previous critic model
        # Be careful that the environment sizing is the same.
        # ACTUALLY: Doesn't seem to work because of custom loss function
        self.critic = load_model(fname)

    # __________________________________________________________________________
    def test_start(self, test_folder_name=TEST_FOLDER_NAME):
        # Begins new game
        if MODE == 'Local':
            self.env.acct.mrkt.data_folder_name = test_folder_name

        self.reset_env()
        self.test_done = False

    # __________________________________________________________________________
    def test_step(self, nSteps=1):
        if not self.test_done:
            for s in range(nSteps):
                actmat, predact = self.get_action()
                obs, rew, done, info = self.env.act(actmat)
                self.reward.append(rew)
                self.observation = obs

        return actmat, predact

    # __________________________________________________________________________
    def plot_test(self, testOut):
        # Notes:
            # Trade entry and exit overlay on candle data

        time = datetime.datetime.now()
        nGames = len(testOut)
        testDF = pd.DataFrame(testOut)

        for g in range(nGames):
            plotData = []
            gameData = testDF.iloc[g]
            for p in PAIRS:
                pairData = gameData['data'][p]
                dates = pairData['gmt']
                candleTrace = go.Ohlc(x=dates,
                                      open=pairData['open'],
                                      high=pairData['high'],
                                      low=pairData['low'],
                                      close=pairData['close'])
                profTrades = []
                lossTrades = []
                for t in gameData['trades']:
                    if t['Pair'] == p:
                        if t['PL'] > 0:
                            profTrades.append(t)
                        else:
                            lossTrades.append(t)

                # Now plot the trades on top of the candles
                xPt = []
                yPt = []
                profCol = '#00FF00'
                for t in profTrades:
                    tStart = dates.iloc[t['Start']+1]
                    tStop = dates.iloc[t['Stop']+1]
                    xPt += [tStart, tStop, None]
                    pStart = t['Open']
                    pStop = t['Close']
                    yPt += [pStart, pStop, None]

                if profTrades == []:
                    profTradeTrace = []
                else:
                    profTradeTrace = go.Scatter(x=xPt, y=yPt,
                                                mode='lines+markers',
                                                line=dict(color=profCol)
                                                )

                xLt = []
                yLt = []
                lossCol = '#FF0000'
                for t in lossTrades:
                    tStart = dates.iloc[t['Start']+1]
                    tStop = dates.iloc[t['Stop']+1]
                    xLt += [tStart, tStop, None]
                    pStart = t['Open']
                    pStop = t['Close']
                    yLt += [pStart, pStop, None]

                if lossTrades == []:
                    lossTradeTrace = []
                else:
                    lossTradeTrace = go.Scatter(x=xLt, y=yLt,
                                                mode='lines+markers',
                                                line=dict(color=lossCol))

                # Valuation and reward in subplots? Later...

                plotData = [candleTrace, profTradeTrace, lossTradeTrace]
                layout = go.Layout(xaxis=dict(fixedrange=False),
                                   yaxis=dict(fixedrange=False))
                fig = go.Figure(data=plotData, layout=layout)
                py.plot(fig,
                        filename='./plots/{}-{}-{}--{}:{}--testGame_{}--{}'
                        .format(time.year, time.month, time.day,
                                time.hour, time.minute, g, p))
