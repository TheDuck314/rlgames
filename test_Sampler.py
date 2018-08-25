#!/usr/bin/python

from Sampler import Sampler
import numpy as np

def main():
    s = Sampler()
    s.add_examples({
        "input":  np.array([[1, 2], [3, 4], [5, 6]]),
        "output": np.array([10, 30, 50]),
    })
    s.add_examples({
        "input":  np.array([[7, 8], [9, 10], [11, 12], [13, 14]]),
        "output": np.array([70, 90, 110, 130]),
    })
    for epoch in range(3):
        print
        print "epoch {}".format(epoch)
        for i, minibatch_fd in enumerate(s.get_minibatches(minibatch_size=2)):
            print "minibatch {}:\n{}".format(i, minibatch_fd)


if __name__ == "__main__":
    main()
