# ****************************************************
# k-means法で都道府県データをクラスタリングするプログラム
# 情22-0419 藤里 和輝
# ****************************************************
import math
import random

# ファイルからデータ（特徴ベクトルのリスト）を読み込む関数
def LoadData():
    prefName=[] # 都道府県名のリスト
    prefLocation=[] # 都道府県の緯度経度データのリスト
    f=open('data1.txt') # ファイルをオープンする
    lines=f.readlines() # 1行分を1つの文字列として読み込み、文字列リストを作成する。
    for line in lines:
        valList=line.split() # 空白を区切りとして文字列を区切る
        prefName.append(valList[0]) # 都道府県名を追加
        loc=[float(valList[1]),float(valList[2])] # 緯度経度データ（要素数2個のリスト）
        prefLocation.append(loc) # 緯度経度データを追加
    f.close()
    return prefName, prefLocation

# 2つの特徴点間の直線距離を計算する関数
def calcDistance(v1, v2):
    sum2=0 # 各成分の差の二乗和
    dim=len(v1) # 次元数
    for i in range(dim):
        sum2+=((v1[i]-v2[i])*(v1[i]-v2[i])) 
    return math.sqrt(sum2)

# 2つの特徴点間の距離の2乗を計算する関数
def calcDistance2(v1, v2):
    sum2=0 # 各成分の差の二乗和
    dim=len(v1) # 次元数
    for i in range(dim):
        sum2+=((v1[i]-v2[i])*(v1[i]-v2[i])) 
    return sum2

# 単語文書行列の正規化
def regulateMat(wdMat):
    numDoc=len(wdMat) # 文書数
    dim=len(wdMat[0]) # 次元数
    wdMatNew=wdMat
    for i in range(numDoc):
        vec=wdMat[i]
        sum2=0
        for j in range(dim):
            sum2+=(vec[j]*vec[j])
        for j in range(dim):
            wdMatNew[i][j]=wdMat[i][j]/math.sqrt(sum2)
    return wdMatNew

# 単語文書行列を表示
# 小数点以下は2桁で表示するものとする
def printWordDocumentMatrix(mat):
    for i in range(len(mat)):
        print('v(d{:d})=['.format(i+1),end='') # end=''は改行の抑制
        for j in range(len(mat[i])):
            print('{:6.2f}'.format(mat[i][j]), end='')
        print(']^t')
    print()

# 以下kmeans法関連の関数

# step 1. 代表点の初期化
# (入力)wdMat:単語文書行列、k:クラスタ数
# (出力)centers:代表点のリスト
def initCenters(prefName, wdMat, k):
    centers=[] # 代表点のリスト
    numDoc=len(wdMat)
    selectedDocs=[] # 初期の代表点を決定するするために選択された文書
    while(len(selectedDocs)<k):
        docNo=random.randint(0,numDoc-1)
        if docNo not in selectedDocs:
            selectedDocs.append(docNo)
    print('選択された都道府県:', end='')
    for docNo in selectedDocs:
        print(prefName[docNo], end=' ')
        centers.append(wdMat[docNo])
    print()
    return centers

# step 2. クラスタ割り当て
# (入力) wdMat: 単語文書行列, centers: 代表点のリスト
# (出力) clusters: 各クラスタに割り当てられた文書
def assignDocs(wdMat, centers):
    k=len(centers) # クラスタ数
    numDoc=len(wdMat) # 文書数
    clusters=[] # 各クラスタに対する文書の割り当て結果
    for i in range(k):
        clusters.append([])
    # 文書ごとに最も類似度の高い代表点を求める。
    for docNo in range(numDoc):
        docVec=wdMat[docNo] # 文書ベクトル(=特徴点)
        assignedClusterNo=0 # 割り当てられたクラスタ番号(逐次更新される)
        minDist=calcDistance(docVec, centers[0]) # 最小距離（逐次更新される）
        # k個の代表点との距離を順番に計算
        for clusterNo in range(k): 
            dist=calcDistance(docVec, centers[clusterNo])
            if dist<=minDist:
                minDist=dist
                assignedClusterNo=clusterNo
        clusters[assignedClusterNo].append(docNo)
    return clusters

# step 3. 代表点の更新
# (入力) wdMat: 単語文書行列, clusters: 各クラスタに割り当てられた文書
# (出力) 更新された代表点
def updateCenters(wdMat, clusters):
    k=len(clusters)
    dim=len(wdMat[0]) # ベクトルの次元数
    centers=[]
    for clusterNo in range(k):
        center=[0]*dim # 更新後の代表点
        for i in range(dim):
            for docNo in clusters[clusterNo]:
                center[i]+=wdMat[docNo][i]
            center[i]/=len(clusters[clusterNo])
        centers.append(center)
    return centers

# クラスタ割り当て結果を表示
# (入力) clusters: 各クラスタに割り当てられた文書
def printClusters(prefName, clusters):
    k=len(clusters)
    for clusterNo in range(k):
        print('クラスタ'+str(clusterNo+1)+':', end='')
        for docNo in clusters[clusterNo]:
            print(prefName[docNo], end=' ')
        print()

# クラスタの代表点を表示
# (入力) centers: 代表点のリスト
def printCenters(centers):
    k=len(centers)
    for clusterNo in range(k):
        print('クラスタ'+str(clusterNo+1)+'の代表点: [', end='')
        vec=centers[clusterNo]
        dim=len(vec)
        for i in range(dim):
            print('{:5.2f}'.format(vec[i]), end=' ')
        print(']^t')



# クラスタ内分散（クラスタの代表点と所属文書の点までの距離の平均）を計算
# (入力) wdMat: 単語文書行列, centers: 代表点のリスト, clusters: 各クラスタに割り当てられた文書
def calcIntraDist(wdMat, centers, clusters):
    # totalAveDist=0 # (←この変数は未使用なので不要)
    k=len(centers) # クラスタ数
    numDoc=len(wdMat) # 文書数
    sum=0
    for i in range(k):
        center=centers[i] # i番目のクラスタの代表点位置
        # numDoc=len(clusters[i]) # i番目のクラスタの所属文書数　(←この行は不要)
        for docNo in clusters[i]: # i 番目のクラスタの文書を順番に調べる
            dist=calcDistance2(center, wdMat[docNo])
            sum+=dist
    return sum/numDoc

# クラスタ間分散（代表点間の距離の平均）を計算
# (入力) centers: 代表点のリスト
def calcInterDist(centers):
    k=len(centers) # クラスタ数
    sum=0
    for i in range(k):
        for j in range(k):
            if i>=j:
                continue
            sum+=calcDistance2(centers[i], centers[j])
    # クラスタの組み合わせの数kC2=k(k-1)/2
    return sum/(k*(k-1)/2)


# プログラムの実行開始ポイント
def main():
    prefName, prefLocation=LoadData() # 都道府県データの読み込み
    '''
    print(prefName) # 確認
    print(prefLocation) # 確認
    '''

    # 単語文書行列（文書ベクトルのリスト）の定義
    wdMat=prefLocation

    k=8 # クラスタ数の設定

    print('step 1. 代表点の初期化')
    centers=initCenters(prefName, wdMat, k)
    print('初期代表点')
    printCenters(centers)
    prevCenters=centers

    while(True):
        print('step 2. クラスタ割り当て')
        clusters=assignDocs(wdMat, centers)
        printClusters(prefName, clusters)

        print('step 3. 代表点の更新')
        centers=updateCenters(wdMat, clusters)
        printCenters(centers)

        if centers==prevCenters: # 前回の代表点位置と比較
            print('代表点が変化しなかったので処理を終了')
            break

        prevCenters=centers # 代表点の位置を別変数に記録しておく

    print('クラスタリング結果評価')
    # クラスタ内分散
    Sintra=calcIntraDist(wdMat, centers, clusters)
    print('クラスタ内分散:'+str(Sintra))

    # クラスタ間分散
    Sinter=calcInterDist(centers)
    print('クラスタ間分散:'+str(Sinter))

    # クラスタリング結果の評価値
    print('クラスタリング結果の評価値:'+str(Sinter/Sintra))

if __name__ == "__main__":
    main()