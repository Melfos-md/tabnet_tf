#%%
import sys
sys.path.append("/home/melfos/dev/tabnet_tf/src")
import tensorflow as tf
from models.attentive_transformer import AttentiveTransformer
# %%
input = tf.constant([[1.0,2.0,3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)
# %%
model = AttentiveTransformer()
output = model(input)

# %%
print(output)
# %%
