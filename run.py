from FxPPO import Agent

ag = Agent()
ag.test_start()
t = ag.test()
ag.plot_test(t)

ag.train(episodes=1000, batch_size=256)

ag.test_start()
tf = ag.test()
ag.plot_test(tf)
