import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization


# csvファイルからPandas DataFrameへ読み込み
train_data = pd.read_csv('train.csv', delimiter=',', low_memory=False)

#train_dataのtargetをカテゴリーに変換
train_data.target = train_data.target.astype('category')

# ラベルエンコーディング（LabelEncoder）
le = LabelEncoder()
encoded = le.fit_transform(train_data.target.values)
decoded = le.inverse_transform(encoded)
train_data.target = encoded


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
        'l3': (10, 400),
        'l1_drop': (0.0, 0.5),
        'l2_drop': (0.0, 0.5),
        'l3_drop': (0.0, 0.5),
        'epochs': (5, 500),
        'batch_size': (128, 2048)
    }
    # 関数と最適化するパラメータを渡す
    optimizer = BayesianOptimization(f=validate, pbounds=pbounds)
    # 最適化
    optimizer.maximize(init_points=5, n_iter=15, acq='ucb')
    return optimizer


#評価------------------------------------------------------------------
def validate(l1, l2, l3, l1_drop, l2_drop, l3_drop, epochs, batch_size):

    #モデルを構築&コンパイル----------------------
    def set_model(l1, l2, l3, l1_drop, l2_drop, l3_drop, epochs, batch_size):
        #モデルを構築
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(75,)),
            keras.layers.Dense(int(l1), activation='relu'),
            keras.layers.Dropout(l1_drop),
            keras.layers.Dense(int(l2), activation='relu'),
            keras.layers.Dropout(l2_drop),
            keras.layers.Dense(int(l3), activation='relu'),
            keras.layers.Dropout(l3_drop),
            keras.layers.Dense(9, activation='softmax')
        ])

        #モデルをコンパイル
        model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model


    #訓練データとテストデータに分割------------------
    def split_data(i):
        train_scope = [(50001, 200000), (100001, 50000), (150001, 100000), (1, 150000)]  #データを分割するための範囲
        test_scope = [(1, 50000), (50001, 100000), (100001, 150000), (150001, 200000)]  #データを分割するための範囲
        if i != 3:
            train = train_data[(train_data.id+1 >= train_scope[i][0]) | (train_data.id+1 <= train_scope[i][1])]
            test = train_data[(train_data.id+1 >= test_scope[i][0]) & (train_data.id+1 <= test_scope[i][1])]
        else:
            train = train_data[(train_data.id+1 >= train_scope[i][0]) & (train_data.id+1 <= train_scope[i][1])]
            test = train_data[(train_data.id+1 >= test_scope[i][0]) & (train_data.id+1 <= test_scope[i][1])]
        return train, test


    #交叉検証------------------------------------
    def Closs_validate(l1, l2, l3, l1_drop, l2_drop, l3_drop, epochs, batch_size):
        eval_sum = 0.0  #評価を格納
        for i in range(4):
            #訓練データとテストデータに分割
            train, test = split_data(i)

            #データとラベルを分割する
            x_train, y_train = train.drop(['target'], axis=1).drop(['id'], axis=1), train.target
            x_test, y_test = test.drop(['target'], axis=1).drop(['id'], axis=1), test.target

            #モデルをセット
            model = set_model(l1, l2, l3, l1_drop, l2_drop, l3_drop, epochs, batch_size)

            #モデルを学習
            model.fit(x_train, y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=0)

            #テストデータを適用
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

            #評価を格納
            eval_sum += test_acc
        return eval_sum/4
        
    return Closs_validate(l1, l2, l3, l1_drop, l2_drop, l3_drop, epochs, batch_size)


if __name__ == '__main__':
    main()