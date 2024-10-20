
import numpy as np

def acc(t, pos, vel):
    r12 = pos[1] - pos[0]
    r23 = pos[2] - pos[1]
    r13 = pos[2] - pos[0]
    f12 = r12/(abs(r12) ** 3)
    f23 = r23/(abs(r23) ** 3)
    f13 = r13/(abs(r13) ** 3)
    if f12 != f12:
        print(pos, vel, t)
        raise SystemExit
    return np.array([f12 + f13, f23 - f12, -f13 - f23])

def vel(t, pos, vel):
    return vel
