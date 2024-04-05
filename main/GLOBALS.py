TTT_SIZE = 3
SNAKE_GRID_DIMS = (15, 15)

MOUNTAIN_CAR_MODE = "MountainCar-v0"
MOUNTAIN_CAR_STATE_SIZE = 2
MOUNTAIN_CAR_ACTION_SIZE = 3
TTT_MODE = "tictactoe-v0"
TTT_STATE_SIZE = TTT_SIZE ** 2
TTT_ACTION_SIZE = TTT_SIZE ** 2
SNAKE_MODE = "snake-v0"  # TODO : Get working
SNAKE_STATE_SIZE = (SNAKE_GRID_DIMS[0] * SNAKE_GRID_DIMS[1]) + 2  # One for each tile + apple coords
SNAKE_ACTION_SIZE = 4

MODES = [MOUNTAIN_CAR_MODE, TTT_MODE, SNAKE_MODE]
DEFAULT_MODE = MODES[0]

_output_sizes = [MOUNTAIN_CAR_ACTION_SIZE, TTT_ACTION_SIZE, SNAKE_ACTION_SIZE]
_input_sizes = [MOUNTAIN_CAR_STATE_SIZE, TTT_STATE_SIZE, SNAKE_STATE_SIZE]
for i in range(len(_input_sizes)):
    _input_sizes[i] += _output_sizes[i]
MODE_INPUT_SIZES = {mode: size for mode, size in zip(MODES, _input_sizes)}
MODE_OUTPUT_SIZES = {mode: size for mode, size in zip(MODES, _input_sizes)}


def _layer_safe(sizes: [int or float]):  # So we never accidentally make a layer have 0 neurons
    return [max(1, int(size)) for size in sizes]


DEFAULT_POLICY_SIZES = {mode: _layer_safe([MODE_INPUT_SIZES[mode], 32, MODE_OUTPUT_SIZES[mode]]) for mode in MODES}
DEFAULT_POLICY_SIZES[MOUNTAIN_CAR_MODE] = _layer_safe([MODE_INPUT_SIZES[MOUNTAIN_CAR_MODE], 16, 16, MOUNTAIN_CAR_ACTION_SIZE])
DEFAULT_POLICY_SIZES[TTT_MODE] = _layer_safe([MODE_INPUT_SIZES[TTT_MODE], 16, 16, TTT_ACTION_SIZE])
_snake_bulge_layer_size = _layer_safe([MODE_INPUT_SIZES[SNAKE_MODE] * 1.5])[0]
DEFAULT_POLICY_SIZES[SNAKE_MODE] = _layer_safe([MODE_INPUT_SIZES[SNAKE_MODE],
                                                _snake_bulge_layer_size,
                                                (_snake_bulge_layer_size + SNAKE_ACTION_SIZE) / 2,
                                                SNAKE_ACTION_SIZE])
