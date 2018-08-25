import numpy as np
import random

class Sampler:
    """ For training algorithms that are of the form, "generate a bunch of examples",
    then do one or more epochs of gradient descent, Sampler makes storing the examples
    more convenient. Generate the feed dict corresponding to each episode's examples
    right away and stick the feed dict in the Sampler. Then once you have enough
    examples, call get_minibatches() to get one epoch's worth of examples. """
    def __init__(self):
        self._many_feed_dicts = []
        self._giant_feed_dict = None
        self._keys = None
        self.num_examples = 0

    def _check_equal_lengths(self, feed_dict):
        """ Assert that all the tensors in feed_dict have the same number of 
        examples. """
        length = feed_dict.values()[0].shape[0]
        assert all(tens.shape[0] == length for tens in feed_dict.values())

    def add_examples(self, feed_dict):
        assert not self._giant_feed_dict, "Can't call add_examples() after get_minibatches()"
        self._check_equal_lengths(feed_dict)
        # check that all the feed_dict's we get have the same keys
        if self._keys is None:
            self._keys = set(feed_dict.keys())
        else:
            assert self._keys == set(feed_dict.keys())

        self._many_feed_dicts.append(feed_dict)
        self.num_examples += feed_dict.values()[0].shape[0]

    def _prepare_giant_feed_dict(self):
        """ smush a bunch of feed dicts into one by concatenating the tensors """
        self._giant_feed_dict = {key:np.concatenate([fd[key] for fd in self._many_feed_dicts])
                                for key in self._keys}
        self._check_equal_lengths(self._giant_feed_dict)

    def _permute_giant_feed_dict(self):
        """ randomly permute our examples """
        ordering = range(self.num_examples)
        random.shuffle(ordering)
        for key in self._keys:
            self._giant_feed_dict[key] = self._giant_feed_dict[key][ordering, ...]

    def get_minibatches(self, minibatch_size):
        """ Returns a generator of feed dicts. Calling this function invalidates
        the generator for this Sampler returned by any previous call to get_minibatches() """
        if self._giant_feed_dict is None:
            self._prepare_giant_feed_dict()
        self._permute_giant_feed_dict()
        for i in range(0, self.num_examples - minibatch_size + 1, minibatch_size):
            yield {key: self._giant_feed_dict[key][i:i+minibatch_size, ...]
                   for key in self._keys}

