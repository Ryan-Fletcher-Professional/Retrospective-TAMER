from main.GLOBALS import *
import Network


class PolicyNetwork(Network):
    def __init__(self, mode=DEFAULT_MODE, sizes=None):
        super().__init__(sizes)
