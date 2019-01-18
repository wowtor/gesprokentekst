#!/usr/bin/env python3

import argparse
import logging

import numpy as np

import liar


DEFAULT_LOGLEVEL = logging.WARNING

LOG = logging.getLogger(__file__)


class AlcoholBreathAnalyser:
    """
    Example from paper:
        Peter Vergeer, Andrew van Es, Arent de Jongh, Ivo Alberink and Reinoud
        Stoel, Numerical likelihood ratios outputted by LR systems are often
        based on extrapolation: When to stop extrapolating? In: Science and
        Justice 56 (2016) 482–491.
    """
    def __init__(self, ill_calibrated=False):
        self.ill_calibrated = ill_calibrated

    def sample_lrs(self):
        positive_lr = 1000 if self.ill_calibrated else 90
        lrs = np.concatenate([np.ones(990) * 0.101, np.ones(10) * positive_lr, np.ones(90) * positive_lr, np.ones(10) * .101])
        y = np.concatenate([np.zeros(1000), np.ones(100)])
        return lrs, y


class Data:
    def __init__(self, lrs=None, y=None):
        self.lrs = lrs
        self.y = y

    def generate_lrs(self, n):
        gen = liar.generators.NormalGenerator(0., 1., 1., 1.)
        #gen = liar.generators.RandomFlipper(gen, .01)
        self.lrs, self.y = gen.sample_lrs(n//2, n//2)

    def breath_lrs(self):
        self.lrs, self.y = AlcoholBreathAnalyser(ill_calibrated=True).sample_lrs()

    def load_lrs(self, path):
        raise ValueError('not implemented')

    def plot_isotonic(self):
        liar.pav.plot(self.lrs, self.y, on_screen=True)

    def plot_ece(self):
        liar.ece.plot(self.lrs, self.y, on_screen=True)

    def plot_nbe(self):
        add_misleading = 1
        print('ELUB', *liar.bayeserror.elub(self.lrs, self.y, add_misleading=add_misleading))
        liar.bayeserror.plot(self.lrs, self.y, add_misleading=add_misleading, on_screen=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LR operations & plotting')
    plotting = parser.add_argument_group('plotting')
    plotting.add_argument('--plot-isotonic', help='generate an Isotonic Regression plot', action='store_true')
    plotting.add_argument('--plot-ece', help='generate an ECE plot (empirical cross entropy)', action='store_true')
    plotting.add_argument('--plot-nbe', help='generate an NBE plot (normalized bayes error rate)', action='store_true')

    etl = parser.add_argument_group('data')
    etl.add_argument('--load-lrs', metavar='FILE', help='read LRs from FILE')
    etl.add_argument('--generate-lrs', metavar='N', type=int, help='draw N LRs from two score distributions')
    etl.add_argument('--breath-lrs', action='store_true', help='use the breath analyser toy data set')

    parser.add_argument('-v', help='increases verbosity', action='count', default=0)
    parser.add_argument('-q', help='decreases verbosity', action='count', default=0)
    args = parser.parse_args()

    data = Data()
    if args.load_lrs:
        data.load_lrs(args.load_lrs)
    if args.generate_lrs:
        data.generate_lrs(args.generate_lrs)
    if args.breath_lrs:
        data.breath_lrs()

    if args.plot_isotonic:
        data.plot_isotonic()
    if args.plot_ece:
        data.plot_ece()
    if args.plot_nbe:
        data.plot_nbe()
