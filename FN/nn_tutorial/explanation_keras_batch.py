import numpy as np
from tensorflow.python import keras as K

# 2-layer neural network.
model = K.Sequential([
    # 1層目から2層目に値を送る際は活性化関数のactivation="sigmoid"を適用
    K.layers.Dense(units=4, input_shape=((2, )),
                   activation="sigmoid"),
    K.layers.Dense(units=4),
])

# Make batch size = 3 data (dimension of x is 2).
# 伝搬処理は複数件まとめて送ることが多いらしい。3件まとめたのがこんな感じ
# [
#  [x^1_1, x^1_2],
#  [x^2_1, x^2_2],
#  [x^3_1, x^3_2],
# ]
batch = np.random.rand(3, 2)

y = model.predict(batch)
print(y.shape)  # Will be (3, 4)
