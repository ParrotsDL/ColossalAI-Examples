from colossalai.amp import AMP_TYPE
from titans.loss.lm_loss import GPTLMLoss
# from titans.model.gpt import gpt2_10B
# from titans.model.gpt import gpt2_65B
from titans.model.gpt import gpt2_13B
# from titans.model.gpt import gpt2_13B as gpt2_small
# from titans.model.gpt import gpt2_small
# from titans.model.gpt import gpt2_small as gpt2_65B
# from titans.model.gpt import gpt2_13B as gpt2_10B # gpt2_xl as
# from titans.model.gpt import gpt2_xl as gpt2_10B # gpt2_xl as
#from model_zoo.gpt.gpt import gpt2_small_pipeline
from torch.optim import Adam
from colossalai.zero.shard_utils import BucketTensorShardStrategy

BATCH_SIZE = 4
SEQ_LEN = 512
NUM_EPOCHS = 60
HIDDEN_SIZE = 768
NUM_MICRO_BATCHES = 2
PIPELINE = 8 #2 #4

MODE = '2d' #'2d' #'1d'
TENSOR_PARALLEL = 4 #4 #2

# zero = dict(
#     model_config=dict(
#         tensor_placement_policy='cpu',
#         shard_strategy=BucketTensorShardStrategy()
#     ),
#     optimizer_config=dict()
# )

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

# fp16 = dict(
#     mode=AMP_TYPE.NAIVE
# )

loss = dict(
    type=GPTLMLoss,
)

model = dict(
    type=gpt2_13B,
    checkpoint=False,
)

# parallel = dict(
#     pipeline=PIPELINE,
#     tensor=dict(size=1, mode=None),
# )

parallel = dict(pipeline=PIPELINE, tensor=dict(mode=MODE, size=TENSOR_PARALLEL))