import numpy as np

from FxMrkt import FxMrkt


class FxAcct:
    def __init__(self, mode='Local', pairs=['EUR_USD', 'USD_JPY', 'GBP_USD',
                                            'AUD_USD', 'USD_CHF', 'USD_CAD',
                                            'EUR_JPY', 'EUR_GBP'],
                 freq='M5', deposit=1000, lev=1,
                 data_folder_name='histdata_v3'):
        self.mode = mode
        self.pairs = pairs
        self.freq = freq
        self.deposit = deposit
        self.cash = deposit
        self.lev = lev

        self.i = 0

        if mode == 'Local':
            self.initLocal(data_folder_name)

        elif mode == 'Practice':
            self.initPractice()

        elif mode == 'Live':
            self.initLive()


# ==============================================================================
    # Initialization
    def initLocal(self, data_folder_name):
        self.mrkt = FxMrkt(mode='Local', pairs=self.pairs, freq=self.freq,
                           data_folder_name=data_folder_name)
        # Need to setup machinery for managing account
        self.positions = {}
        self.trades = []
        self.valuation = [self.deposit]
        self.realHist = [self.deposit]
        self.unrealHist = [0]


# ==============================================================================
    # Control
    # Stuff like resetting balances


# ==============================================================================
    # Trading
    def buy(self, pair, size):
        if self.mode == 'Local':
            # See if there are open positions for this pair
            exist = pair in self.positions

            if exist and (self.positions[pair]['Size'] < 0):
                # If we had sold some of 'pair', close those before buying
                self.close(pair)
                exist = False

            elif exist:
                # A buy position is already open
                # Do we add to the position or leave it?
                # Let's just leave it for now.
                return

            if not exist:
                self.positions[pair] = {'Size': size,
                                        'Price': self.mrkt.getPrice(pair),
                                        'Time': self.i}
                self.cash -= size * self.positions[pair]['Price'] / self.lev
                if self.cash < 0:
                    print('OUT OF CASH')
                return

        elif self.mode == 'Practice':
            # aoeu
            1+1

        elif self.mode == 'Live':
            # aoeu
            1+1

# ------------------------------------------------------------------------------
    def sell(self, pair, size):
        if self.mode == 'Local':
            # See if there are open positions for this pair
            exist = pair in self.positions

            if exist and (self.positions[pair]['Size'] > 0):
                self.close(pair)
                exist = False

            elif exist:
                # A sell position is already open
                # Leave it alone
                return

            if not exist:
                self.positions[pair] = {'Size': -size,
                                        'Price': self.mrkt.getPrice(pair),
                                        'Time': self.i}
                self.cash -= size * self.positions[pair]['Price'] / self.lev
                return

        elif self.mode == 'Practice':
            # aoeu
            1+1

        elif self.mode == 'Live':
            # aoeu
            1+1

# ------------------------------------------------------------------------------
    def close(self, pair):
        if self.mode == 'Local':
            # Close the trade, log the prices, calculate the profit/loss
            exist = pair in self.positions
            if exist:
                t = {}
                t['Pair'] = pair
                s = self.positions[pair]['Size']
                t['Size'] = s
                t['Open'] = self.positions[pair]['Price']
                t['Close'] = self.mrkt.getPrice(pair)
                t['PL'] = s * (t['Close'] - t['Open'])
                t['Pct'] = (s/abs(s)) * (self.lev/t['Open']
                                         ) * (t['Close'] - t['Open']) * 100
                t['Start'] = self.positions[pair]['Time']
                t['Stop'] = self.i

                del self.positions[pair]
                self.cash += abs(s)*t['Open']/self.lev + t['PL']
                self.trades.append(t)

        elif self.mode == 'Practice':
            # aoeu
            1+1

        elif self.mode == 'Live':
            # aoeu
            1+1


# ==============================================================================
    # External Interface Functions
    def value(self):
        # Calculates total account value (realized + unrealized)
        val = self.cash
        lev = self.lev
        for pair in self.positions:
            pos = self.positions[pair]
            s = pos['Size']
            p1 = pos['Price']
            p2 = self.mrkt.getLastClose(pair)
            v = (abs(s)*p1/lev) + s*(p2-p1)
            val += v

        return val

# ------------------------------------------------------------------------------
    def realVal(self):
        return self.cash

# ------------------------------------------------------------------------------
    def unrealVal(self):
        val = 0
        lev = self.lev
        for pair in self.positions:
            pos = self.positions[pair]
            s = pos['Size']
            p1 = pos['Price']
            p2 = self.mrkt.getLastClose(pair)
            v = (abs(s)*p1/lev) + s*(p2-p1)
            val += v

        return val

# ------------------------------------------------------------------------------
    def getDicState(self):
        d = {}
        d['Cash'] = self.cash
        d['Value'] = self.value
        d['Positions'] = self.positions
        return d

# ------------------------------------------------------------------------------
    def getVecState(self):
        # Format:
            # Each pair is either +1(bought) 0(none) or -1(sold)
            # [1 0.997 0 0 -1 1.342 -1 136.448]
            # MEANS:
            # Bought at 0.997, no positions, sold at 1.342, sold at 136.448
        v = np.array([])
        for pr in self.pairs:
            exist = pr in self.positions
            if exist:
                bhs = self.positions[pr]['Size']/abs(
                                                    self.positions[pr]['Size'])
                p = self.positions[pr]['Price']
            else:
                bhs = 0
                p = 0
            v = np.append(v, bhs)
            v = np.append(v, p)
        return v

# ------------------------------------------------------------------------------
    def step(self):
        self.realHist.append(self.cash)
        self.mrkt.step()
        self.valuation.append(self.value())
        self.unrealHist.append(self.unrealVal())
        self.i += 1

# ------------------------------------------------------------------------------
    def reset(self):
        self.positions = {}
        self.trades = []
        self.valuation = [self.deposit]
        self.cash = self.deposit
        self.realHist = [self.deposit]
        self.unrealHist = [0]
        self.i = 0
        self.mrkt.nextSession()
