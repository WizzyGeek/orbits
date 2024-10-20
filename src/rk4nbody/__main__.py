
import numpy as np
import cmath
from time import time

from .rk4 import GeneralisedRungeKutta4
from .tbody import vel, acc

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib


class Animation:
    def __init__(self):
        self.then = time()
        self.GRK4 = GeneralisedRungeKutta4(2, [np.array([cmath.exp(0) -0.3 - 0.3j, cmath.exp(2j * cmath.pi / 3) + 0.4j, cmath.exp(-2j * cmath.pi / 3) - 0.5j], dtype=np.complex256) * 9, np.zeros(3, dtype=np.complex256)], [vel, acc], step=1/(40 * 60))
        self.thent = self.GRK4.t - 1
        self.thene = 0
        self.BATCH = 40
        self.SKIP = 4
        self.rspss = 1
        self.f = 0
        self.create_animation()

    def update(self, frame):
        BATCH = self.BATCH
        GRK4 = self.GRK4
        self.f += 1

        now = time()
        for i in range(BATCH * int(min((now - self.then) / (GRK4.t - self.thent), 10) + 1)):
            pos = GRK4.vars[0]
            pr = min(min(abs(pos[1] - pos[0]), abs(pos[2] - pos[1]), abs(pos[2] - pos[0])) / 300, BATCH / 2)
            GRK4.do_step(h=pr/(BATCH))

        dots = np.array([[dot.real, dot.imag] for dot in GRK4.vars[0]])
        self.scatter.set_offsets(dots)

        if (GRK4.epoch - self.thene) > self.BATCH * 8:
            m = np.min(dots, axis=0)
            ma = np.max(dots, axis=0)
            now = time()
            rspss = min((now - self.then) / (GRK4.t - self.thent), 100)
            # EPSS: Epochs per simulated second
            # EPRS: Epochs per real second
            # FPRS: Frames Per real second
            # RSPSS: Real second per simulation second
            print("Current Stats | Time:", GRK4.t, "| EPSS: ", (GRK4.epoch - self.thene) / (GRK4.t - self.thent), "| EPRS:", (GRK4.epoch - self.thene) / (now - self.then), "| FPRS: ", self.f / (now - self.then), "| RSPSS:", rspss)
            self.f = 0
            self.then = time()
            self.thent = GRK4.t
            self.thene = GRK4.epoch

            # Set new limits with some padding
            self.ax.set_xlim(min(m[0] - 0.4 * abs(m[0]), 5), max(ma[0] + 0.4 * abs(ma[0]), 5))
            self.ax.set_ylim(min(m[1] - 0.4 * abs(m[1]), -5), max(ma[1] + 0.4 * abs(ma[0]), 5))

        return self.scatter,

    def create_animation(self):
        matplotlib.use('TkAgg')

        fig, self.ax = plt.subplots()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')

        self.scatter = self.ax.scatter([], [], s=100)

        self.ani = FuncAnimation(fig, self.update, frames=10000, interval=-1)

        plt.title('2D 3 body problem, Generalised RK4 adaptive')
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.grid()


def main():
    anim = Animation()
    plt.show()

if __name__ == "__main__": main()