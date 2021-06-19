import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
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
    print(dna_lst)
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
    split_num = 5

    #モデルを構築&コンパイル----------------------
    def set_model():
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


    #訓練データとテストデータに分割------------------
    def split_data(i):
        train_scope = [(40001, 200000), (80001, 40000), (120001, 80000), (160001, 120000), (1, 160000)]  #データを分割するための範囲
        test_scope = [(1, 40000), (40001, 80000), (80001, 120000), (120001, 160000), (160001, 200000)]  #データを分割するための範囲
        if i != split_num-1 or i != 0:
            train = train_data[(train_data.id+1 >= train_scope[i][0]) | (train_data.id+1 <= train_scope[i][1])]
            test = train_data[(train_data.id+1 >= test_scope[i][0]) & (train_data.id+1 <= test_scope[i][1])]
        else:
            train = train_data[(train_data.id+1 >= train_scope[i][0]) & (train_data.id+1 <= train_scope[i][1])]
            test = train_data[(train_data.id+1 >= test_scope[i][0]) & (train_data.id+1 <= test_scope[i][1])]

        # 遺伝子に基づいて特徴量抽出する
        drop_count = 0
        for i, d in enumerate(dna):
            if d == 0:
                train = train.drop(train.columns[[i+1-drop_count]], axis=1)
                test = test.drop(test.columns[[i+1-drop_count]], axis=1)
                drop_count += 1
        return train, test


    #交叉検証------------------------------------
    def Closs_validate():
        eval_sum = 0.0  #評価を格納
        for i in range(split_num):
            #訓練データとテストデータに分割
            train, test = split_data(i)

            #データとラベルを分割する
            x_train, y_train = train.drop(['target'], axis=1).drop(['id'], axis=1), train.target
            x_test, y_test = test.drop(['target'], axis=1).drop(['id'], axis=1), test.target

            #モデルをセット
            model = set_model()

            #モデルを学習
            model.fit(x_train, y_train, epochs=10, batch_size=512, verbose=0)

            #テストデータを適用
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

            #評価を格納
            eval_sum += test_acc
        return eval_sum/split_num

    # 評価していく
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