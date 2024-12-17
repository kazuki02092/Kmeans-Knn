# ****************************************************************
# k-NN法で都道府県データをカテゴリ識別するプログラム
# 情22-0419 藤里 和輝
# ****************************************************************
import math
import sys
import random

# ファイルから都道府県データを読み込む関数
def LoadData():
    f=open('data2.txt')
    lines=f.readlines() # 文字列のリストとして読み込む(1行を1つの文字列として読み込む)
    prefName=[] # 都道府県名のリスト
    prefAreaNo=[] # 都道府県の地域番号のリスト
    prefLocation=[] # 都道府県の緯度経度データのリスト
    for line in lines:
        valList=line.split()
        prefName.append(valList[0]) # 都道府県名を追加
        prefAreaNo.append(int(valList[1])) # 地域番号(int型に変換)を追加
        loc=[float(valList[2]), float(valList[3])] # 緯度経度(float型2個のリスト)
        prefLocation.append(loc) # 緯度経度データをを追加
    return prefName, prefAreaNo, prefLocation

# 2つの特徴点間の直線距離を計算する関数
def calcDistance(v1, v2):
    sum2=0 # ベクトルの成分の差の二乗和
    dim=len(v1) # 次元数
    for i in range(dim):
        sum2+=((v1[i]-v2[i])*(v1[i]-v2[i])) 
    return math.sqrt(sum2)

# 学習データの表示
def printTrainingData(vecTrainingData, categoryTrainingData, prefName, trainIndex, categoryName):
    num=len(vecTrainingData) # num=len(categoryTrainingData)でも可
    for i in range(num):
        print('v(d{:d})=['.format(i+1),end='') # end=''は改行の抑制
        dim=len(vecTrainingData[i]) # ベクトルの次元を取得
        for j in range(dim):
            print('{:6.2f}'.format(vecTrainingData[i][j]), end='')
        print(']^t ', end='')
        print(prefName[trainIndex[i]], end=' ')
        categoryNo=categoryTrainingData[i]
        print(categoryName[categoryNo]+'地方')
    print()

# 分類対象データ（文書ベクトル)の表示
def printTestData(vecTestData, prefName, testIndex):
    num=len(vecTestData) 
    for i in range(num):
        print('v(d{:d})=['.format(i+101),end='') # end=''は改行の抑制
        dim=len(vecTestData[i])
        for j in range(dim):
            print('{:6.2f}'.format(vecTestData[i][j]), end='')
        print(']^t ', end=' ')
        print(prefName[testIndex[i]])
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
def main():
    # 都道府県データの読み込み
    prefName, prefAreaNo, prefLocation=LoadData()
    # カテゴリ名
    categoryName=['東北・北海道', '関東', '中部', '近畿', '中国', '四国', '九州・沖縄']
    '''
    print(prefName) # 確認
    print(prefAreaNo) # 確認
    print(prefLocation) # 確認
    '''

    # 学習データとテストデータの分類
    totalIndex=list(range(len(prefName))) # 47都道府県のindexリスト
    testIndex=random.sample(totalIndex, 5) # テストデータとするindexのリスト
    trainIndex=[]
    for i in totalIndex:
        if i not in testIndex:
            trainIndex.append(i) # 学習データとするindexのリスト

    # 学習データ
    # 学習データの文書カテゴリ(カテゴリ番号(prefAreaNoに対応))
    categoryTrainingData=[]
    for i in trainIndex:
        categoryTrainingData.append(prefAreaNo[i])
    # 学習データの文書ベクトル(単語文書行列)
    vecTrainingData=[]
    for i in trainIndex:
        vecTrainingData.append(prefLocation[i])

    # 分類データ（カテゴリ識別対象文書のベクトル)
    vecDoc=[]
    for i in testIndex:
        vecDoc.append(prefLocation[i])

    numDoc=len(vecDoc) # 分類データの数
    k=3 # 上位k個の文書で多数決を取る

    # 学習データ
    print('学習データ')
    printTrainingData(vecTrainingData, categoryTrainingData, prefName, trainIndex, categoryName)

    # 分類データ
    print('分類データ')
    printTestData(vecDoc, prefName, testIndex)

    correct_count = 0 # 成功回数

    print('都道府県の地方カテゴリ識別')
    for i in range(numDoc):
        distanceList=calcAllDistances(vecDoc[i], vecTrainingData) # 分類データと学習データの類似度(距離)を求める
        # print(distanceList)
        topk=getTopM(distanceList, k) # 類似度top k個の学習データ（の文書番号）を取得
        print(prefName[testIndex[i]]+'（トップ'+str(k)+'）', end='')
        for j in range(k):
            docNo=topk[j]
            categoryNo=categoryTrainingData[docNo]
            print(prefName[trainIndex[docNo]]+'('+str(categoryNo)+')', end=' ')
        estimatedCategoryNo=estimateCategory(topk, categoryTrainingData, categoryName) # 多数決をとって分類データのカテゴリを決定
        print('⇒ （識別結果）'+categoryName[estimatedCategoryNo]+'地方', end=' ')
        if estimatedCategoryNo == prefAreaNo[testIndex[i]]:
            print('識別成功')
            correct_count=correct_count+1
        else:
            print('識別失敗')
    
    print('識別成功率：'+str(correct_count)+'/'+str(len(vecDoc)))

if __name__ == "__main__":
    main()