import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold

# データの前処理------------------------------------------------------
# csvファイルからPandas DataFrameへ読み込み
train_data = pd.read_csv('train_SMOTE.csv', delimiter=',', low_memory=False)

#train_dataのtargetをカテゴリーに変換
train_data.target = train_data.target.astype('category')

# ラベルエンコーディング（LabelEncoder）
le = LabelEncoder()
encoded = le.fit_transform(train_data.target.values)
decoded = le.inverse_transform(encoded)
train_data.target = encoded
'''
# 遺伝子座に基づいて特徴量抽出する
drop_count = 0
dna = np.array([1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0])
input_num = np.sum(dna)
for i, d in enumerate(dna):
    if d == 0:
        train_data = train_data.drop(train_data.columns[[i+1-drop_count]], axis=1)
        drop_count += 1
'''
# 訓練データを分割する
X, y = train_data.drop(['target'], axis=1).drop(['id'], axis=1).values, train_data.target.values
input_num = X.shape[1]

#メイン-------------------------------------------------------------
def main():
    # ベイズ最適化実行
    optimizer = bayesOpt()
    print(optimizer.res)


#ベイズ最適化---------------------------------------------------------
def bayesOpt():
    # 最適化するパラメータの下限・上限
    pbounds = {
        'l1': (10, 400),
        'l2': (10, 400),
        'l1_drop': (0.0, 0.5),
        'l2_drop': (0.0, 0.5),
        'epochs': (5, 500),
        'batch_size': (128, 2048)
    }
    # 関数と最適化するパラメータを渡す
    optimizer = BayesianOptimization(f=validate, pbounds=pbounds)
    # 最適化
    optimizer.maximize(init_points=5, n_iter=10, acq='ucb')
    return optimizer


#評価------------------------------------------------------------------
def validate(l1, l2, l1_drop, l2_drop, epochs, batch_size):

    #モデルを構築&コンパイル----------------------
    def set_model():
        #モデルを構築
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(input_num,)),
            keras.layers.Dense(int(l1), activation='relu'),
            keras.layers.Dropout(l1_drop),
            keras.layers.Dense(int(l2), activation='relu'),
            keras.layers.Dropout(l2_drop),
            keras.layers.Dense(9, activation='softmax')
        ])
        #モデルをコンパイル
        model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model


    #交叉検証------------------------------------
    def Closs_validate():
        # 交差検証を実行
        valid_scores = []  # 評価を格納する配列
        kf = KFold(n_splits=5, shuffle=True, random_state=42) #データの分割の仕方を決定
        for fold, (train_indices, valid_indices) in enumerate(kf.split(X)):
            X_train, X_valid = X[train_indices], X[valid_indices]
            y_train, y_valid = y[train_indices], y[valid_indices]

            # モデルをセット
            model = set_model()
            
            # 学習させる
            model.fit(X_train, y_train,
                    validation_data=(X_valid, y_valid),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    verbose=0)

            # テストデータを適用する
            y_valid_pred = model.predict(X_valid)
            y_valid_pred = [np.argmax(i) for i in y_valid_pred]
            
            # 識別率を求める
            score = accuracy_score(y_valid, y_valid_pred)

            # 評価を格納する
            valid_scores.append(score)

        cv_score = np.mean(valid_scores)
        return cv_score
        
    return Closs_validate()#l1, l2, l1_drop, l2_drop, epochs, batch_size


if __name__ == '__main__':
    main()