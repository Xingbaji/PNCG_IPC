from time import time

stack = []
index = dict()
flags = []
levels = []
timings = []


class Timer(object):
    def __init__(self, description):
        self.description = description

    def __enter__(self):
        stack.append(self.description)
        if self.description not in index:
            index[self.description] = len(flags)
            flags.append(self.description)
            levels.append(len(stack))
            timings.append(0.0)
        id = index[self.description]
        timings[id] -= time()

    def __exit__(self, type, value, traceback):
        stack.pop()
        id = index[self.description]
        timings[id] += time()


def Timer_Print():
    total = 0.0
    for l, t in zip(levels, timings):
        if l == 1:
            total += t
    Timer_Print.cnt += 1
    if Timer_Print.cnt == 2:
        Timer_Print.cmp -= total

    for f, l, t in zip(flags, levels, timings):
        print('  ' * l, end='')
        avg_t = t / max(1, Timer_Print.cnt - 1)
        avg_p = t / total
        tot_t = t
        print('{0:s}: \033[92m{1:.4f}s, {2:.1%}\033[0m  \033[33m(All: {3:.2f}s)\033[0m'.format(f, avg_t, avg_p, tot_t))
    print('  Compile Time: \033[92m{0:.2f}s\033[0m'.format(Timer_Print.cmp))
    print('')

    if Timer_Print.cnt == 1:
        Timer_Print.cmp = total
        for i in range(len(timings)):
            timings[i] = 0.0


Timer_Print.cnt = 0
Timer_Print.cmp = 0.0
