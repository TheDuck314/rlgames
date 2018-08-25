#!/usr/bin/python

import sys
import os
import time

from Replayer import Replayer


def wait_for_new_replay(replay_dir, last_replay_fn=None):
    last_mtime = 0.0
    if last_replay_fn and os.path.isfile(last_replay_fn):
        last_mtime = os.path.getmtime(last_replay_fn)

    while True:
        best_fn = None
        best_age = 1e99
        min_age = 1.0  # to avoid reading while a file's still being written
        max_age = 20.0
    
        for basename in os.listdir(replay_dir):
            if not basename.endswith(".pkl"):
                continue
            fn = os.path.join(replay_dir, basename)
            mtime = os.path.getmtime(fn)
            age = time.time() - mtime
            if mtime <= last_mtime:
                continue  # don't show anything older than the last replay
            if not (min_age < age < max_age):
                continue  # must satisfy age requirements
            if age >= best_age:
                continue  # look for most recent
            best_fn = fn
            best_age = age
    
        if best_fn:
            print "found new replay {} which is {} seconds old".format(best_fn, best_age)
            return best_fn

        # didn't find any files satisfying our requirements. chill for a bit
        print "sleeping while we wait for new replays..."
        time.sleep(1.0)



def continuously_show_replays(replay_dir):
    replay_fn = None
    while True:
        replay_fn = wait_for_new_replay(replay_dir, last_replay_fn=replay_fn)
        Replayer.load_and_show(replay_fn)


def main():
    if len(sys.argv) != 2:
        print "usage: ReplayWatcher.py <replay file or directory>"
        sys.exit(1)

    arg = sys.argv[1]
    if os.path.isdir(arg):
        continuously_show_replays(replay_dir=arg)
    elif os.path.isfile(arg):
        Replayer.load_and_show(fn=arg)
    else:
        print "Couldn't find {}".format(arg)
        sys.exit(1)


if __name__ == "__main__":
    main()
