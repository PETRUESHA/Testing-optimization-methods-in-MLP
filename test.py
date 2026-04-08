# import numpy as np
from mlip_4 import FitErrors, ETN, Cfg, RadialBasisCinf, MTP, CalcCfg, LossCfg, LossFunction, PairPot, QRd

# sim = CalcCfg(Cfg([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [
#               0.5, 0.5, 0.5]], [41, 41]), forces=True, stress=True, charges=False)
# func = LossFunction()
# func.add(LossCfg(sim))
# rb = RadialBasisCinf(size=6, min_dist=2.0, cutoff=5.0)
# pot0 = PairPot(rb, [41, 45])

# func.attach_pot(pot0)
# print(func.calc_errors().epa())
# print(func.calc_errors())
# print(help(FitErrors))
# print(help(ETN))
print(help(LossFunction))
