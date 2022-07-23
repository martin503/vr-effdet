import time
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, scale_difference=1):
        self.scale_difference = scale_difference

    def plot(self, title, xlabel, ylabel, x, y1, y2, smoothness):
        assert len(y1) == len(y2)
        plt.title(title)
        y1 = y1[self.scale_difference:]
        y2 = y2[self.scale_difference:]
        plt.plot(x, self.__smooth_curve(y1, factor=smoothness), 'r')
        plt.plot(x, self.__smooth_curve(y2, factor=smoothness), 'b')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim((1, len(x)))
        plt.savefig('./plots/' + title + '_' + self.__plot_timestamp())
        plt.show()

    def print_min(self, name, tr, val, fac):
        self.__print_best(name, tr, val, np.min, np.argmin, fac)

    def print_max(self, name, tr, val, fac):
        self.__print_best(name, tr, val, np.max, np.argmax, fac)

    def __print_best(self, name, tr, val, fun, arg_fun, fac):
        print('Best training {} {:.2f} in epoch {}'.format(name, fun(tr), fac * (arg_fun(tr) + 1)))
        print('Best validation {} {:.2f} in epoch {}'.format(name, fun(val), fac * (arg_fun(val) + 1)))

    def __plot_timestamp(self):
        current_time = time.localtime()
        timestamp = time.strftime('%d_%b_%Y-%H:%M', current_time)
        return timestamp

    def __smooth_curve(self, points, factor=0.7):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points
