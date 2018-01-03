from train import load_data
from tracker import Tracker

def p5_init(root=None):
    load_data(root)
    tracker = Tracker()
    return tracker