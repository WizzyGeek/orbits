
import typing as tp

import numpy as np
# import numpy.typing as npt

FOUR = 4
RK4COEFS = (0, 1/2, 1/2, 1)
RK4AGG = (1/6, 2/6, 2/6, 1/1)

class GeneralisedRungeKutta4:
    # slopes: npt.NDArray[np.float64]
    vars: list[tp.Any] # could be used for anything following a ring
    epoch: int

    def __init__(self, num_vars: int, init_vars: list[tp.Any], funcs: list[tp.Callable], t0: float = 0, step: float = 1/60) -> None:
        self.k = list(map(lambda _: [0] * FOUR, range(num_vars)))
        self.funcs = funcs
        self.vars = init_vars
        self._old = [None] * num_vars
        self.t = t0
        self.h = step
        self.epoch = 0
        self.n = num_vars

        if len(init_vars) != num_vars or num_vars != len(funcs):
            raise ValueError("Inconsistent data")

        return None

    def do_step(self, h: float | None = None):
        h = h or self.h
        slopes = self.k

        for i in range(0, FOUR):
            for j in range(self.n):
                slopes[j][i] = self.funcs[j](self.t, *map(lambda k: k[1] + (i > 0 and h * slopes[k[0]][i - 1] * RK4COEFS[i]), enumerate(self.vars)))

        for i, v in enumerate(self.vars):
            self.vars[i] = v + h * np.dot(RK4AGG, slopes[i])

        self.t += h
        self.epoch += 1

# l = RungeKutta(2, [])
# tp.reveal_type(l.slopes[0][0])
