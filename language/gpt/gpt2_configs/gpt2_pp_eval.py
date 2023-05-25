from colossalai.amp import AMP_TYPE
from titans.loss.lm_loss import GPTLMLoss
from titans.model.gpt import gpt2_13B
#from model_zoo.gpt.gpt import gpt2_small_pipeline
from torch.optim import Adam


BATCH_SIZE = 6
SEQ_LEN = 1024
NUM_EPOCHS = 60
HIDDEN_SIZE = 768
NUM_MICRO_BATCHES = 1
PIPELINE = 8

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

parallel = dict(
    pipeline=PIPELINE,
    tensor=dict(size=1, mode=None),
)
