import numpy as np
from tensorflow.python import keras as K

model = K.Sequential([
    # Dense=重みとバイアスをもつ層のこと。重みをかけ、バイアスを足す処理を行う層。
    # 重みとバイアスについては参考書のNo1168の行列の式をみればいいよ
    # units=出力サイズ。行動価値と同じ。上下左右の行動価値を行列で出力するようにしている？
    # input_shape=はinputのサイズ。ここだとstateの(x,y)の二次元の座標の値が入る
    K.layers.Dense(units=4, input_shape=((2, ))),
])

weight, bias = model.layers[0].get_weights()
print("Weight shape is {}.".format(weight.shape))
print("Bias shape is {}.".format(bias.shape))

x = np.random.rand(1, 2)
y = model.predict(x)
print("x is ({}) and y is ({}).".format(x.shape, y.shape))
