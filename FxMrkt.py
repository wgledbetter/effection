import numpy as np
import pandas as pd
import random
import os


class FxMrkt:
    def __init__(self, mode='Local', pairs=['EUR_USD', 'USD_JPY', 'GBP_USD',
                                            'AUD_USD', 'USD_CHF', 'USD_CAD',
                                            'EUR_JPY', 'EUR_GBP'],
                       freq='M5', data_folder_name='histdata_v3'):

        self.mode = mode
        self.pairs = pairs  # Names of pairs to use
        self.freq = freq
        self.i = 0  # Number of steps in this session

        if mode == 'Local':
            self.initLocal(data_folder_name=data_folder_name)
        elif mode == 'Practice':
            self.initPractice()
        elif mode == 'Live':
            self.initLive()


#===============================================================================
    # Initialization
    def initLocal(self, data_folder_name):
        # Uses historical data from local files
        # Parameters that need to be defined for local:
            # Time
        self.pair = {}  # Dictionary that stores actual market data
        self.data_folder_name = data_folder_name
        self.loadGame()
        self.session_length = len( self.pair[self.pairs[0]]['gmt'] )

#-------------------------------------------------------------------------------
    def initPractice(self):
        # Uses oanda practice account
        1+2

#-------------------------------------------------------------------------------
    def initLive(self):
        # Uses oanda live account
        1+2

#===============================================================================
    # External Interface Functions
    def nextSession(self):
        self.i = 0
        if self.mode == 'Local':
            self.loadGame()

        elif self.mode == 'Practice':
            # Wait a weekend?
            1+2

        elif self.mode == 'Live':
            # Wait a weekend?
            1+2

#-------------------------------------------------------------------------------
    def getDicState(self):
        # Return the current state as a dictionary.
        # Better for use when talking to Account
        ret = {}
        if self.mode == 'Local':
            for p in self.pair:
                ret[p] = self.pair[p].iloc[self.i].to_dict()

        elif self.mode == 'Practice':
            1+1

        elif self.mode == 'Live':
            1+1

        return ret

#-------------------------------------------------------------------------------
    def getVecState(self):
        # Return the current state as a np array.
        # Better for use when talking to Agent
        ret = np.array([])
        if self.mode == 'Local':
            for p in self.pairs:
                a = self.pair[p].iloc[self.i].values
                ret = np.append(ret, a[1:])  # Cut the timestamp

        elif self.mode == 'Practice':
            1+1

        elif self.mode == 'Live':
            1+3

        return ret

#-------------------------------------------------------------------------------
    def step(self):
        self.i += 1
        if self.mode == 'Local':
            1+1
            if self.i >= self.session_length:
                print('END OF GAME')

        elif self.mode == 'Practice':
            1+1

        elif self.mode == 'Live':
            1+1

#===============================================================================
    # Local Functions
    def loadGame(self):
        game_dir = random.choice( os.listdir('./{}/'.format(self.data_folder_name)))
        game_path = './{}/{}/'.format(self.data_folder_name, game_dir)
        for f in os.listdir(game_path):
            try:
                if (f[8:] == 'pkl') and (f[0:7] in self.pairs):
                    data = pd.read_pickle('{}{}'.format(game_path, f))
                    self.pair[f[0:7]] = data
            except:
                # Probably means its a shorter filename
                1+2
        self.session_length = len( self.pair[self.pairs[0]]['gmt'] )

#-------------------------------------------------------------------------------
    def getPrice(self, p):
        if self.mode == 'Local':
            # Some sort of distribution between high and low of next state
            # For now lets make it uniform.
            try:
                l = self.pair[p]['low'].iloc[self.i+1]
                h = self.pair[p]['high'].iloc[self.i+1]
                return np.random.uniform(l,h)
            except:
                print('END OF GAME')
                return 0

        elif self.mode == 'Practice':
            1+1

        elif self.mode == 'Live':
            1+2

#-------------------------------------------------------------------------------
    def getLastClose(self, p):
        if self.mode == 'Local':
            try:
                c = self.pair[p]['close'].iloc[self.i]
                return c
            except:
                return 999999999999