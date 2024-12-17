# ****************************************************************
# k-NN法で文書をカテゴリ識別する
# (c) Kansai University, 2024
# ****************************************************************

import math
import sys

# 2つの特徴点間の直線距離を計算する関数
def calcDistance(v1, v2):
    sum2=0 # ベクトルの成分の差の二乗和
    dim=len(v1) # 次元数
    for i in range(dim):
        sum2+=((v1[i]-v2[i])*(v1[i]-v2[i])) 
    return math.sqrt(sum2)

# 学習データの表示
def printTrainingData(vecTrainingData, categoryTrainingData, categoryName):
    num=len(vecTrainingData) # num=len(categoryTrainingData)でも可
    for i in range(num):
        print('v(d{:d})=['.format(i+1),end='') # end=''は改行の抑制
        dim=len(vecTrainingData[i]) # ベクトルの次元を取得
        for j in range(dim):
            print('{:6.2f}'.format(vecTrainingData[i][j]), end='')
        print(']^t ', end='')
        categoryNo=categoryTrainingData[i]
        print(categoryName[categoryNo])
    print()

# 分類対象データ（文書ベクトル)の表示
def printTestData(vecTestData):
    num=len(vecTestData) 
    for i in range(num):
        print('v(d{:d})=['.format(i+101),end='') # end=''は改行の抑制
        dim=len(vecTestData[i])
        for j in range(dim):
            print('{:6.2f}'.format(vecTestData[i][j]), end='')
        print(']^t ')
    print()


# 以下、kNN関連の関数の定義

# 分類データの文書ベクトルとすべての学習データとの距離を求める
# (入力) vec: 分類データの文書ベクトル, wdMat: 学習データの単語文書行列
# (出力) distanceList: 文書間距離のリスト
def calcAllDistances(vec, wdMat):
    numTrainingData=len(wdMat)
    distanceList=[] 
    for i in range(numTrainingData):
        distance=calcDistance(vec, wdMat[i])
        distanceList.append(distance)
    return distanceList

# 距離リストを元にトップk個の学習データ（文書番号）リストを求める
# (入力) distanceList: 距離リスト, k: kの値
# (出力) topk: 類似度トップk個の文書の文書番号リスト
def getTopM(distanceList,k):
    numTrainingData=len(distanceList)
    topk=[]
    for i in range(k):
        # topkに登録されているもの以外で最大の類似度を求める
        minDocNo=-1
        minDistance=sys.maxsize
        for j in range(numTrainingData):
            if j in topk: # topkリストに含まれている文書番号はスキップ
                continue
            if distanceList[j]<=minDistance: # 距離が小さい方が類似度が高い
                minDistance=distanceList[j]
                minDocNo=j
        topk.append(minDocNo)
    return topk

# 類似度上位k個の学習データのカテゴリで多数決をとり、分類データのカテゴリ（番号）を推定する
# (入力) topk: 類似度top k個の文書の文書番号リスト，catetoryTrainingData: 各学習データのカテゴリ, categoryName: カテゴリ名のリスト
# (出力) estimatedCategoryNo: 推定結果（カテゴリー番号）
def estimateCategory(topk, categoryTrainingData, categoryName):
    numCategory=len(categoryName) # カテゴリ数を得る
    count=[] # 各カテゴリの得票数
    for i in range(numCategory): # 初期化（すべてのカテゴリの得票数を0にセット)
        count.append(0)

    for docNo in topk: # topkリストに含まれる文書の番号を順番に取得
        category=categoryTrainingData[docNo] # 文書のカテゴリを取得
        count[category]+=1 # category番号のカテゴリに1票投票
 
    # 最大得票数のカテゴリを調べる
    estimatedCateogryNo=0 # 最大得票数のカテゴリ番号
    maxCount=0
    for i in range(numCategory):
        if(count[i]>maxCount): # update 
            maxCount=count[i]
            estimatedCategoryNo=i

    return estimatedCategoryNo


# プログラムの実行開始ポイント

# カテゴリ名
categoryName=['カテゴリ1','カテゴリ2','カテゴリ3']

# 学習データ
# 学習データの文書カテゴリ(カテゴリ番号(categoryNameの要素番号に対応))
categoryTrainingData=[
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2]
# 学習データの文書ベクトル(単語文書行列)
vecTrainingData=[
    [ 7,14],[ 3,18],[ 7,19],[ 4,22],[10,20],[12,17],
    [19,16],[19,20],[22,19],[22,22],[19,23],[25,22],
    [27, 2],[23, 7],[29, 6],[25,10],[29,11],[30, 2]]

# 分類データ（カテゴリ識別対象文書のベクトル)
vecDoc=[
    [9,16],[16,18],[22,15],[26,7]
]

numDoc=len(vecDoc) # 分類データの数
k=3 # 上位k個の文書で多数決を取る

# 学習データ
print('学習データ')
printTrainingData(vecTrainingData, categoryTrainingData, categoryName)

# 分類データ
print('分類データ')
printTestData(vecDoc)

for i in range(numDoc):
    print('分類対象文書'+str(i+101)+'のカテゴリ識別')
    distanceList=calcAllDistances(vecDoc[i], vecTrainingData) # 分類データと学習データの類似度(距離)を求める
    # print(distanceList)
    topk=getTopM(distanceList, k) # 類似度top k個の学習データ（の文書番号）を取得
    print('類似度トップ'+str(k)+': ', end='')
    for i in range(k):
        docNo=topk[i]
        categoryNo=categoryTrainingData[docNo]
        print('文書'+str(docNo+1)+'('+categoryName[categoryNo]+')', end=' ')
    estimatedCategoryNo=estimateCategory(topk, categoryTrainingData, categoryName) # 多数決をとって分類データのカテゴリを決定
    print()
    print('⇒ 識別結果:'+categoryName[estimatedCategoryNo])
    print()
