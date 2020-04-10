BLANK_INIT = True    # False로 하면 기존의 모델을 불러온다.

STATE_DIM = FEATURE_DIM * 2
ACTION_DIM = FEATURE_DIM + CLASSES

own_label = 1

COLUMN_LABEL = '_label'
COLUMN_DROP  = ['_count']

META_COSTS = 'cost'
META_AVG   = 'avg'
META_STD   = 'std'

#================== RL
FEATURE_FACTOR   =   0.005
REWARD_CORRECT   =   0
REWARD_INCORRECT =  -1

#================== TRAINING
AGENTS = 1000

TRAINING_EPOCHS = 5000

EPOCH_STEPS = 2

EPSILON_START  = 0.80
EPSILON_END    = 0.10
EPSILON_EPOCHS = 2500	 	# epsilon will fall to EPSILON_END after EPSILON_EPOCHS
EPSILON_UPDATE_EPOCHS = 10  # update epsilon every x epochs

#================== LOG
from config.initial_log import TRACKED_STATES
LOG_TRACKED_STATES = TRACKED_STATES

LOG_EPOCHS = 500  			# states prediction will be logged every LOG_EPOCHS

LOG_PERF_EPOCHS = 100
LOG_PERF_VAL_SIZE = 1000

#================== NN
BATCH_SIZE =    10000     # 만
POOL_SIZE  =  1000000     # 백만

NN_FC_DENSITY = 64
NN_HIDDEN_LAYERS = 2

OPT_LR = 1.0e-4
OPT_ALPHA = 0.95
OPT_MAX_NORM = 1.0

# LR scheduling => lower LR by LR_SC_FACTOR every LR_SC_EPOCHS epochs
LR_SC_FACTOR =   0.9
LR_SC_EPOCHS = 500
LR_SC_MIN = 1.0e-7

TARGET_RHO = 0.01

#================== AUX
SAVE_EPOCHS = 10
MAX_MASK_CONST = 1.e6

SEED = 112233
