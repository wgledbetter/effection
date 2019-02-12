import numpy as np

from FxAcct import FxAcct


class FxEnv:
    def __init__(self, mode='Local', pairs=['EUR_USD', 'USD_JPY', 'GBP_USD',
                                            'AUD_USD', 'USD_CHF', 'USD_CAD',
                                            'EUR_JPY', 'EUR_GBP'],
                 freq='M5', deposit=1000, lev=1,
                 data_folder_name='histdata_v3', market_state_hist=4,
                 lot_size=100, lstm=False):

        self.mode = mode
        self.pairs = pairs
        self.nPairs = len(pairs)
        self.freq = freq
        self.market_state_hist = market_state_hist
        self.lot_size = lot_size
        self.lstm = lstm
        if self.lstm:
            self.market_state_hist = 1

        self.i = 0

        if mode == 'Local':
            self.initLocal(deposit=deposit, lev=lev,
                           data_folder_name=data_folder_name)

        elif mode == 'Practice':
            self.initPractice()

        elif mode == 'Live':
            self.initLive()


# ==============================================================================
    # Initialization
    def initLocal(self, deposit, lev, data_folder_name):
        self.acct = FxAcct(mode=self.mode, pairs=self.pairs, freq=self.freq,
                           deposit=deposit, lev=lev)
        self.mktHist = []
        if self.lstm:
            self.stateHist_LSTM = []


# ==============================================================================
    # Primary Interface
    def state(self):
        d = self.acct.mrkt.session_length/5
        h = d/24
        t1 = int(self.i/d)
        t2 = int((self.i-t1*d)/h)
        t3 = (self.i - t1*d - t2*h)/h
        time = np.array([t1, t2, t3])
        acctState = self.acct.getVecState()
        mrktState = self.acct.mrkt.getVecState()
        histMktState = np.array([])
        # If self.lstm, then this for loop doesn't execute
        for i in range(self.market_state_hist-1):
            try:
                hms = self.mktHist[-(i+1)]
            except:
                # For the first few states, we won't have a full history
                hms = np.zeros(5*self.nPairs)
                histMktState = np.append(histMktState, hms)
        st = np.append(time, np.append(acctState, np.append(mrktState,
                                                            histMktState)))
        if (self.i != 0) and (self.lstm):
            st = np.vstack([self.stateHist_LSTM, st])
        return st

# ------------------------------------------------------------------------------
    def LSTM_1state(self):
        d = self.acct.mrkt.session_length/5
        h = d/24
        t1 = int(self.i/d)
        t2 = int((self.i-t1*d)/h)
        t3 = (self.i - t1*d - t2*h)/h
        time = np.array([t1, t2, t3])
        acctState = self.acct.getVecState()
        mrktState = self.acct.mrkt.getVecState()
        st = np.append(time, np.append(acctState, mrktState))
        return st

# ------------------------------------------------------------------------------
    def state_LSTM(self):
        # Return all states from start to now
        # Each state a row
        if self.i == 0:
            return self.state()
        else:
            LSTM_hist = np.array(self.stateHist_LSTM)
            LSTM_state = np.vstack([LSTM_hist, self.state()])
            return LSTM_state

# ------------------------------------------------------------------------------
    def reward(self, mode=2):
        if mode == 1:
            val_im1 = self.acct.valuation[-2]
            val_i = self.acct.value()
            return 100*(val_i - val_im1)/val_im1
        elif mode == 2:
            W = 2  # Weight of real vs unreal
            real_im1 = self.acct.realHist[-2]
            real_i = self.acct.realVal()
            unreal_im1 = self.acct.unrealHist[-2]
            unreal_i = self.acct.unrealVal()
            # Weight real value 2x
            u_P_Wr_i = W*real_i + unreal_i
            u_P_Wr_im1 = W*real_im1 + unreal_im1
            return 100*(u_P_Wr_i - u_P_Wr_im1)/u_P_Wr_im1
        elif mode == 3:
            # Create some penalty for long trades?
            5*8

# ------------------------------------------------------------------------------
    def act(self, action):
        # Parse and Execute the buy/sell/hold actions
        # action is a 3*nPairs vector.
        for i, pair in enumerate(self.pairs):
            cmd = np.argmax(action[3*i:3*(i+1)])
            if cmd == 0:
                # Buy
                self.acct.buy(pair, self.lot_size)

            elif cmd == 1:
                # Close
                self.acct.close(pair)

            elif cmd == 2:
                # Sell
                self.acct.sell(pair, self.lot_size)

        self.step()
        # Need to return: observation(state), reward, done, info
        done = (self.i == self.acct.mrkt.session_length-2)
        if self.lstm:
            st = self.LSTM_1state()
        else:
            st = self.state()
        return st, self.reward(), done, {}

# ------------------------------------------------------------------------------
    def step(self):
        if self.mode == 'Local':
            self.mktHist.append(self.acct.mrkt.getVecState())
        if self.lstm:
            self.stateHist_LSTM.append(self.LSTM_1state())
        self.acct.step()
        self.i += 1

# ------------------------------------------------------------------------------
    def reset(self):
        self.acct.reset()
        self.i = 0
        self.mktHist = []
        self.stateHist_LSTM = []
        return self.state()


# ==============================================================================
    # Get info about the environment
    def stateSize(self):
        # time size + account state size + market state size + market hist size
        # Time = 3
        # Account = 2 per pair
        # Market State = 5 per pair
        # Market Hist = (nMktHis-1)t*(5 per pair)
        size = 3 + (2*self.nPairs) + (self.market_state_hist) * (
                                                                 5*self.nPairs)
        return size

# ------------------------------------------------------------------------------
    def actionSize(self):
        # 3 per pair
        size = 3*self.nPairs
        return size
