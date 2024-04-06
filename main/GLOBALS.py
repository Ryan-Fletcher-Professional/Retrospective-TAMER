TTT_SIZE = 5
SNAKE_GRID_DIMS = (15, 15)

MOUNTAIN_CAR_MODE = "MountainCar-v0"
MOUNTAIN_CAR_STATE_SIZE = 2
MOUNTAIN_CAR_ACTION_SIZE = 3

MOUNTAIN_CAR_GAMMA = {"alpha": 2, "loc": 0.2, "scale": 0.1}
GAMMA_CREDIT_CUTOFF = 0.01

TTT_MODE = "TicTacToe-v0"  # TODO : Verify
TTT_STATE_SIZE = TTT_SIZE ** 2
TTT_ACTION_SIZE = TTT_SIZE ** 2
SNAKE_MODE = "Snake-v0"  # TODO : Verify
SNAKE_STATE_SIZE = (SNAKE_GRID_DIMS[0] * SNAKE_GRID_DIMS[1]) + 2  # One for each tile + apple coords
SNAKE_ACTION_SIZE = 4

MODES = [MOUNTAIN_CAR_MODE, TTT_MODE, SNAKE_MODE]
DEFAULT_MODE = MODES[0]

_state_sizes = [MOUNTAIN_CAR_STATE_SIZE, TTT_STATE_SIZE, SNAKE_STATE_SIZE]
_action_sizes = [MOUNTAIN_CAR_ACTION_SIZE, TTT_ACTION_SIZE, SNAKE_ACTION_SIZE]
MODE_STATE_SIZES = {mode: size for mode, size in zip(MODES, _state_sizes)}
MODE_ACTION_SIZES = {mode: size for mode, size in zip(MODES, _state_sizes)}

def _layer_safe(sizes: [int or float]):  # So we never accidentally make a layer have 0 neurons
    return [max(1, int(size)) for size in sizes]
DEFAULT_POLICY_SIZES = {mode: _layer_safe([MODE_STATE_SIZES[mode], 32, MODE_ACTION_SIZES[mode]]) for mode in MODES}
DEFAULT_POLICY_SIZES[MOUNTAIN_CAR_MODE] = _layer_safe([MOUNTAIN_CAR_STATE_SIZE, 8, MOUNTAIN_CAR_ACTION_SIZE])
DEFAULT_POLICY_SIZES[TTT_MODE] = _layer_safe([TTT_STATE_SIZE, 32, TTT_ACTION_SIZE])
_snake_bulge_layer_size = _layer_safe([SNAKE_STATE_SIZE * 1.5])[0]
DEFAULT_POLICY_SIZES[SNAKE_MODE] = _layer_safe([SNAKE_STATE_SIZE,
                                                _snake_bulge_layer_size, (_snake_bulge_layer_size + SNAKE_ACTION_SIZE) / 2,
                                                SNAKE_ACTION_SIZE])
