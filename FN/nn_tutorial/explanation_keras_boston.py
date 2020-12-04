import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python import keras as K

# 住居価格を予測するmodel
# 住宅価格って強化学習と関係あるのか・・・？

dataset = load_boston()

y = dataset.target
X = dataset.data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33)

model = K.Sequential([
    # 13の特徴量
    # BatchNormalizationはデータの正規化を行うための処理らしい。同じ値が同じ意味を持つように調整する。
    # 平均0, 分散1に揃えている。
    # TODO: 価格って今以上の金額が出たりすると思うけど、分散1に抑えちゃっていいの？
    K.layers.BatchNormalization(input_shape=(13,)),
    # 活性化関数としてsoftplusを使用している。なぜsoftplusなのかは不明
    # softplus ... https://www.atmarkit.co.jp/ait/articles/2004/22/news014.html
    K.layers.Dense(units=13, activation="softplus", kernel_regularizer="l1"),
    # ここで出力を1つにしている
    K.layers.Dense(units=1)
])
# lossにmean_squared_errorと書くのは、二乗誤差を再消化する式を作りたいため
# 最適化には確率的勾配降下法(sgd)を使っている。なぜこれを使っているのかは不明
model.compile(loss="mean_squared_error", optimizer="sgd")
model.fit(X_train, y_train, epochs=8)

predicts = model.predict(X_test)

result = pd.DataFrame({
    "predict": np.reshape(predicts, (-1,)),
    "actual": y_test
})
limit = np.max(y_test)

result.plot.scatter(x="actual", y="predict", xlim=(0, limit), ylim=(0, limit))
plt.show()

# 個人的にmseいくら位になったのかを出力して欲しかった