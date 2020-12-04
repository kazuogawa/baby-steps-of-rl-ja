import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from tensorflow import keras as K

# CNNでの実装。手書き数字のデータ。8x8グレースケール画像

dataset = load_digits()
image_shape = (8, 8, 1)
num_class = 10

y = dataset.target
# num_classのカテゴリ数に変更しているみたい。one_hotベクトル　https://keras.io/ja/utils/np_utils/
y = K.utils.to_categorical(y, num_class)
X = dataset.data
# reshapeで8x8x1の次元に変えている
X = np.array([data.reshape(image_shape) for data in X])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33)

# 0-9の10個の確率値を返す
model = K.Sequential([
    # フィルターの枚数が5
    # filter(CNNで畳み込みを行うために一定領域に切り取ってるデータ)のサイズは3x3x1(実際の一定領域に位は深さがある立方体)
    # stride(フィルタによる畳み込みを行う時に1枚目と2枚目のずらす値)は1
    # padding(端の領域が畳み込まれる回数が少なくなるので、周りを拡張すること)はfilterサイズ分補うpaddingをする(縦と横に1枠増えるだけみたい)
    # http://ni4muraano.hatenablog.com/entry/2017/02/02/195505
    # 活性化関数はReLU
    K.layers.Conv2D(
        5, kernel_size=3, strides=1, padding="same",
        input_shape=image_shape, activation="relu"),
    K.layers.Conv2D(
        3, kernel_size=2, strides=1, padding="same",
        activation="relu"),
    # 畳み込みそうの出力をFlattenで1次元ベクトルに変換
    K.layers.Flatten(),
    K.layers.Dense(units=num_class, activation="softmax")
])
model.compile(loss="categorical_crossentropy", optimizer="sgd")
model.fit(X_train, y_train, epochs=8)

predicts = model.predict(X_test)
predicts = np.argmax(predicts, axis=1)
actual = np.argmax(y_test, axis=1)
print(classification_report(actual, predicts))
