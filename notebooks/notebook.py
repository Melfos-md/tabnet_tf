#%%
import tensorflow as tf

# Utilisez cette couche pour le débogage
layer = tf.keras.layers.Softmax()

# Vos données d'entrée
logits = tf.constant([[1.0, 2.0, 3.0]])

# Utilisez la couche pour obtenir la sortie (à des fins de débogage)
output = layer(logits).numpy()

# Ensuite, examinez la sortie et vérifiez si elle est correcte
print(output)
# %%
