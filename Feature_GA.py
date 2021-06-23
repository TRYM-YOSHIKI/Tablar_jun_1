import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
#from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import random


# csvファイルからPandas DataFrameへ読み込み
train_data = pd.read_csv('train.csv', delimiter=',', low_memory=False)

#train_dataのtargetをカテゴリーに変換
train_data.target = train_data.target.astype('category')

# ラベルエンコーディング（LabelEncoder）
le = LabelEncoder()
encoded = le.fit_transform(train_data.target.values)
decoded = le.inverse_transform(encoded)
train_data.target = encoded

# 訓練データを分割する
X, y = train_data.drop(['target'], axis=1).drop(['id'], axis=1), train_data.target


#メイン---------------------------------------------------------------------------
def main():
    #初期解生成
    dna_1 = np.array(random.choices([0,1], k=75))
    dna_2 = np.array(random.choices([0,1], k=75))
    dna_3 = np.array(random.choices([0,1], k=75))
    for sedai in range(1000):
        #交叉
        dna_4, dna_5, dna_6, dna_7, dna_8, dna_9 = closs(dna_1, dna_2, dna_3)
        #突然変異
        dna_4, dna_5, dna_6, dna_7, dna_8, dna_9 = mutation(dna_4, dna_5, dna_6, dna_7, dna_8, dna_9)
        #評価
        eval_lst = evalate(dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9)
        #出力
        output(dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9, eval_lst, sedai)
        #終了条件
        if sedai >= 999:
            break
        #淘汰
        dna_1, dna_2, dna_3 = selection(dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9, eval_lst)


#出力-----------------------------------------------------------------------------
def output(dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9, eval_lst, sedai):
    dna_lst = [dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9]
    max_idx = np.argmax(eval_lst)
    print('第{}世代'.format(sedai))
    print(eval_lst)
    print(dna_lst[max_idx])
    print('Test accuracy: {}'.format(eval_lst[max_idx]))
    print()


#交叉-----------------------------------------------------------------------------
def closs(dna_1, dna_2, dna_3):
    dna_4 = np.copy(dna_1)
    dna_5 = np.copy(dna_1)
    dna_6 = np.copy(dna_2)
    dna_7 = np.copy(dna_2)
    dna_8 = np.copy(dna_3)
    dna_9 = np.copy(dna_3)
    #一様交叉
    Mask = np.array(random.choices([0,1], k=75))
    for i, mask in enumerate(Mask):
        if mask == 0:
            dna_5[i] = dna_2[i]
        else:
            dna_4[i] = dna_2[i]
    Mask = np.array(random.choices([0,1], k=75))
    for i, mask in enumerate(Mask):
        if mask == 0:
            dna_7[i] = dna_3[i]
        else:
            dna_6[i] = dna_3[i]
    Mask = np.array(random.choices([0,1], k=75))
    for i, mask in enumerate(Mask):
        if mask == 0:
            dna_8[i] = dna_1[i]
        else:
            dna_9[i] = dna_1[i]
    return dna_4, dna_5, dna_6, dna_7, dna_8, dna_9


#突然変異---------------------------------------------------------------------------
def mutation(dna_4, dna_5, dna_6, dna_7, dna_8, dna_9):
    dna_lst = [dna_4, dna_5, dna_6, dna_7, dna_8, dna_9]
    for i, dna in enumerate(dna_lst):
        #30%の確率で突然変異
        x = True if 30 >= random.randint(0,100) else False
        if x == True:
            rand_num = random.choice([i for i in range(int(len(dna)*0.3)) if i >= 5])
            rand_posi = random.sample(range(len(dna)), k=rand_num)
            for posi in rand_posi:
                if dna[posi] == 0:
                    dna[posi] = 1
                else:
                    dna[posi] = 0
    return dna_4, dna_5, dna_6, dna_7, dna_8, dna_9


#評価--------------------------------------------------------------------------------
def evalate(dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9):
    dna_lst = [dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9]

    #モデルを構築&コンパイル----------------------
    def set_model(input_num):
        #モデルを構築
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(input_num,)),
            keras.layers.Dense(80, activation='relu'),
            keras.layers.Dense(9, activation='softmax')
        ])
        #モデルをコンパイル
        model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model


    # 遺伝子に基づいて特徴量抽出する------------------
    def extract_data():
        drop_count = 0
        X_extract = X
        for i, d in enumerate(dna):
            if d == 0:
                X_extract = X_extract.drop(X_extract.columns[[i-drop_count]], axis=1)
                drop_count += 1
        return X_extract


    #交差検証------------------------------------
    def Closs_validate():
        # 特徴量抽出を行う
        X_extract = extract_data()

        # numpyに変換する
        X_extract = X_extract.values
        y_extract = y.values

        # 交差検証を実行
        valid_scores = []  # 評価を格納する配列
        kf = KFold(n_splits=5, shuffle=True, random_state=42) #データの分割の仕方を決定
        for fold, (train_indices, valid_indices) in enumerate(kf.split(X_extract)):
            X_train, X_valid = X_extract[train_indices], X_extract[valid_indices]
            y_train, y_valid = y_extract[train_indices], y_extract[valid_indices]

            # モデルをセット
            model = set_model(X_train.shape[1])
            
            # 学習させる
            model.fit(X_train, y_train,
                    validation_data=(X_valid, y_valid),
                    epochs=10,
                    batch_size=512,
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

    # 全遺伝子を評価していく
    eval_lst = []  # 評価を格納する配列
    for dna in dna_lst:
        input_num = np.sum(dna)
        eval_lst.append(Closs_validate())
    return eval_lst


#淘汰--------------------------------------------------------------------------------
def selection(dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9, eval_lst):
    dna_lst = [dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9]
    #トーナメント形式で次世代を決定する
    next_gene = []
    tnmt_num_list = random.sample(range(len(dna_lst)), k=len(dna_lst))
    #一回戦
    eval_tnm_lst = [eval_lst[tnmt_num_list[0]], eval_lst[tnmt_num_list[1]], eval_lst[tnmt_num_list[2]]]
    next_gene.append(tnmt_num_list[np.argmax(eval_tnm_lst)])
    #二回戦
    eval_tnm_lst = [eval_lst[tnmt_num_list[3]], eval_lst[tnmt_num_list[4]], eval_lst[tnmt_num_list[5]]]
    next_gene.append(tnmt_num_list[np.argmax(eval_tnm_lst)+3])
    #三回戦
    eval_tnm_lst = [eval_lst[tnmt_num_list[6]], eval_lst[tnmt_num_list[6]], eval_lst[tnmt_num_list[8]]]
    next_gene.append(tnmt_num_list[np.argmax(eval_tnm_lst)+6])
    return dna_lst[next_gene[0]], dna_lst[next_gene[1]], dna_lst[next_gene[2]]


if __name__ == '__main__':
    main()