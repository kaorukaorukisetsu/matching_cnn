import cv2
import copy
import cv2
import numpy as np
import pandas as pd
import random
import time
from sklearn.ensemble import IsolationForest
import seaborn as sns
AKAZE_th = -0.000005

limitpx = 3
ransacnum = 5000









def AKAZE(img):
    kp,f=cv2.AKAZE_create(threshold=AKAZE_th).detectAndCompute(img,None)

    return kp,f

#def CNN(img):

#kp  tokuchouten
#f tokuchouryou


def matchRatio(keyPoint1, feature1, keyPoint2, feature2, knn, ratio):
    matcher = cv2.BFMatcher()
    a = None
    if type(feature2) == type(a):
        return 0, 0

    else:
        #matches = matcher.knnMatch(feature1, feature2, k=knn)

        matches = matcher.knnMatch(feature1.transpose(1, 0), feature2.transpose(1, 0), k=2)

        good = []
        img1_pt = []
        img2_pt = []
        img1_f = []
        img2_f = []

        for n in range(len(matches)):
            first = matches[n][0]

            if matches[n][0].distance <= ratio * matches[n][1].distance:
                good.append([matches[n][0]])
                img2_pt.append(keyPoint2[first.trainIdx].pt)
                img1_pt.append(keyPoint1[first.queryIdx].pt)
                #img2_f.append(feature2[first.trainIdx])
                #img1_f.append(feature1[first.queryIdx])

        return img1_pt, img2_pt,


# 3ピクセル以内にあるかどうかの判定
def fabs(pt1, pt2, limitPx, check=False):
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]

    d = (dx ** 2) + (dy ** 2)

    if check == True:
        print(d)

    if d < (limitPx ** 2):
        if check == True:
            return True, d
        else:
            return True

    else:
        if check == True:
            return False, d
        else:
            return False


# 変換行列計算　相似変換
def Similarity_tform(pt1, pt2, num):
    u = np.zeros([2 * num, 5])
    for j in range(num):
        u[2 * j][0] = -1 * pt1[j][1]
        u[2 * j][1] = pt1[j][0]
        u[2 * j][2] = 0
        u[2 * j][3] = -1
        u[2 * j][4] = pt2[j][1]

        u[2 * j + 1][0] = pt1[j][0]
        u[2 * j + 1][1] = pt1[j][1]
        u[2 * j + 1][2] = 1
        u[2 * j + 1][3] = 0
        u[2 * j + 1][4] = -1 * pt2[j][0]

    U, s, V = np.linalg.svd(u)
    ans = V[-1:]

    tform = np.zeros([3, 3])

    tform[0][0] = ans[0][0] / ans[0][4]
    tform[0][1] = ans[0][1] / ans[0][4]
    tform[0][2] = ans[0][2] / ans[0][4]
    tform[1][0] = ans[0][1] / ans[0][4] * -1
    tform[1][1] = ans[0][0] / ans[0][4]
    tform[1][2] = ans[0][3] / ans[0][4]
    tform[2][0] = 0
    tform[2][1] = 0
    tform[2][2] = 1

    return tform


# 変換行列計算　Affine変換
def Affine_tform(pt1, pt2, num):
    u = np.zeros([2 * num, 7])
    for j in range(num):
        u[2 * j][0] = 0
        u[2 * j][1] = 0
        u[2 * j][2] = 0
        u[2 * j][3] = -1 * pt1[j][0]
        u[2 * j][4] = -1 * pt1[j][1]
        u[2 * j][5] = -1
        u[2 * j][6] = pt2[j][1]

        u[2 * j + 1][0] = pt1[j][0]
        u[2 * j + 1][1] = pt1[j][1]
        u[2 * j + 1][2] = 1
        u[2 * j + 1][3] = 0
        u[2 * j + 1][4] = 0
        u[2 * j + 1][5] = 0
        u[2 * j + 1][6] = -1 * pt2[j][0]

    U, s, V = np.linalg.svd(u)
    ans = V[-1:]
    tform = np.zeros([3, 3])

    tform[0][0] = ans[0][0] / ans[0][6]
    tform[0][1] = ans[0][1] / ans[0][6]
    tform[0][2] = ans[0][2] / ans[0][6]
    tform[1][0] = ans[0][3] / ans[0][6]
    tform[1][1] = ans[0][4] / ans[0][6]
    tform[1][2] = ans[0][5] / ans[0][6]
    tform[2][0] = 0
    tform[2][1] = 0
    tform[2][2] = 1

    return tform


# 変換行列計算　射影変換
def Projective_tform(pt1, pt2, num):
    u = np.zeros([2 * num, 9])
    for j in range(num):
        u[2 * j][0] = pt1[j][0]
        u[2 * j][1] = pt1[j][1]
        u[2 * j][2] = 1
        u[2 * j][3] = 0
        u[2 * j][4] = 0
        u[2 * j][5] = 0
        u[2 * j][6] = -1 * pt1[j][0] * pt2[j][0]
        u[2 * j][7] = -1 * pt1[j][1] * pt2[j][0]
        u[2 * j][8] = -1 * pt2[j][0]

        u[2 * j + 1][0] = 0
        u[2 * j + 1][1] = 0
        u[2 * j + 1][2] = 0
        u[2 * j + 1][3] = pt1[j][0]
        u[2 * j + 1][4] = pt1[j][1]
        u[2 * j + 1][5] = 1
        u[2 * j + 1][6] = -1 * pt1[j][0] * pt2[j][1]
        u[2 * j + 1][7] = -1 * pt1[j][1] * pt2[j][1]
        u[2 * j + 1][8] = -1 * pt2[j][1]

    U, s, V = np.linalg.svd(u)
    ans = V[-1:]

    tform = np.zeros([3, 3])

    tform[0][0] = ans[0][0]
    tform[0][1] = ans[0][1]
    tform[0][2] = ans[0][2]
    tform[1][0] = ans[0][3]
    tform[1][1] = ans[0][4]
    tform[1][2] = ans[0][5]
    tform[2][0] = ans[0][6]
    tform[2][1] = ans[0][7]
    tform[2][2] = ans[0][8]

    return tform


# 変換行列計算まとめ
def tformCompute(pt1, pt2, mode, num=0):
    random.seed(0)
    random.shuffle(pt1)

    random.seed(0)
    random.shuffle(pt2)

    if num == 0:
        num = mode

    if mode == 2:
        tform = Similarity_tform(pt1, pt2, num)
    elif mode == 3:
        tform = Affine_tform(pt1, pt2, num)
    elif mode == 4:
        tform = Projective_tform(pt1, pt2, num)

    else:
        print("modeを確認してください")

    return np.matrix(tform)


# ransacの終了関数
def computeLoopNumbers(numpts, inlierNum, sampleSize):
    eps = 1.0e-15
    inlierProbability = 1.0
    factor = inlierNum / numpts
    confidence = 99.99

    for i in range(sampleSize):
        inlierProbability *= factor

    if (inlierProbability < eps):
        nn = 2 ** 15
    else:
        conf = confidence / 100
        numerator = np.log10(1 - conf)
        denominator = np.log10(1 - inlierProbability)

        nn = int(numerator / denominator)

    return nn


# ransac本体
def ransac(pt1, pt2, mode):
    n = len(pt1)
    num = mode
    if n < num:
        return 0
    else:

        i = 0
        max_accuracy = 0
        max_mpt = []
        max_cpt = []

        while i < ransacnum:
            tform = tformCompute(pt1, pt2, mode)

            s = 0
            mappt = []
            cappt = []
            for k in range(mode, n):
                c_pt = np.matrix([[pt1[k][0]], [pt1[k][1]], [1]])
                m_pt = tform * c_pt

                pred = [m_pt[0][0], m_pt[1][0]]

                if fabs(pred, pt2[k], limitpx) == True:
                    s += 1
                    cappt.append(pt1[k])
                    mappt.append(pt2[k])
                else:
                    pass

            accuracy = s / n
            if max_accuracy < accuracy:
                max_accuracy = accuracy
                max_mpt = mappt
                max_cpt = cappt
                iternum = computeLoopNumbers(n, s, mode)

                if iternum <= i:
                    break
            i += 1

        return max_cpt, max_mpt


# 平均値標準偏差での絞り込み
def select_inlinears(pt1, pt2, true_point, mode, sgms=1):
    ave = [0.0, 0.0]
    sgm = [0.0, 0.0]

    predpts = []
    n = len(pt1)

    gpt1 = []
    gpt2 = []
    distance = []

    for i in range(int(n / mode)):
        cpt = []
        mpt = []

        for j in range(mode):
            cpt.append(pt1[i * mode + j])
            mpt.append(pt1[i * mode + j])

        tform = tformCompute(pt1, pt2, mode)
        pred = tform * np.matrix([[255.5], [255.5], [1.0]])
        outpt = [float(pred[0][0] / pred[2][0]), float(pred[1][0] / pred[2][0])]
        predpts.append(outpt)

    n = len(predpts)

    for i in range(n):
        ave[0] += (predpts[i][0] / n)
        ave[1] += (predpts[i][1] / n)

    for j in range(n):
        sgm[0] += (((predpts[j][0] - ave[0]) ** 2) / n)
        sgm[1] += (((predpts[j][1] - ave[1]) ** 2) / n)

    for k in range(n):
        dist = ((predpts[k][0] - ave[0]) ** 2) + ((predpts[k][1] - ave[1]) ** 2)
        distance.append(int(dist))
        check = ((predpts[k][0] - ave[0]) ** 2) / (sgm[0] * (sgms ** 2)) + ((predpts[k][1] - ave[1]) ** 2) / (
                    sgm[1] * (sgms ** 2))
        if check < 1.0:
            for l in range(mode):
                gpt1.append(pt1[k * mode + l])
                gpt2.append(pt2[k * mode + l])
    """
    distance.sort()
    for i in range(5):
        distance.pop(-1)

    plt.hist(distance, bins = 10)
    plt.show()
    """

    return gpt1, gpt2


# マッチングの判定
def judge(true_point, tform, limitpx):
    pt = np.matrix([[255.5], [255.5], [1.0]])
    out = tform * pt

    pred = [out[0][0] / out[2][0], out[1][0] / out[2][0]]
    print(pred, true_point)

    return fabs(pred, true_point, limitpx, True)


# 真値の読み込み
def getTruePoint(data2):
    true_point = [0.0, 0.0]
    x = ""
    j = 0
    for k in data2[0]:
        if k != ",":
            x += k
        else:
            true_point[j] = float(x)
            x = ""
            j += 1

            if j == 2:
                break
    data2.pop(0)

    return true_point


def split_xy(pt):
    x = []
    y = []
    for i in pt:
        x.append(i[0])
        y.append(i[1])

    return x, y


def fab(pt1, pt2):
    dx = (pt1[0] - pt2[0]) ** 2
    dy = (pt1[1] - pt2[1]) ** 2
    r2 = dx + dy
    d = r2 ** 0.5
    return d


def get_index(r, k):
    r2 = copy.copy(r)
    index_list = []
    for i in range(k):
        n = r.index(min(r2))
        index_list.append(n)
        r2.pop(r2.index(min(r2)))
    return index_list


def get_distance_list(pred):
    r = []
    for i in range(len(pred)):
        r1 = []
        for j in range(len(pred)):
            r1.append(fab(pred[i], pred[j]))
        r.append(r1)
    return r


def LOF(pt1, pt2, k, LOF_th, mode, true_point):
    predpts = []
    n = len(pt1)

    L = 0

    gpt1 = []
    gpt2 = []
    preds = []

    for i in range(int(n / mode)):
        cpt = []
        mpt = []

        for j in range(mode):
            cpt.append(pt1[i * mode + j])
            mpt.append(pt2[i * mode + j])

        tform = tformCompute(cpt, mpt, mode)
        pred = tform * np.matrix([[255.5], [255.5], [1.0]])
        outpt = [float(pred[0][0] / pred[2][0]), float(pred[1][0] / pred[2][0])]
        predpts.append(outpt)

    r = get_distance_list(predpts)
    lrd = []
    index = []
    for i in range(len(predpts)):
        # 最近傍k個の点のインデックス取得
        index.append(get_index(r[i], k))

        rd = []
        for j in index[i]:
            get_index(r[j], k)
            rd.append(max(r[i][j], r[j][k - 1]))
        num = 0
        for l in range(k):
            num += rd[l] / k
        lrdk = 1 / num
        lrd.append(lrdk)
        for m in range(len(index)):
            LOF = 0
            for n in range(len(index[m])):
                LOF += lrd[n] / lrd[m] / k
                L += LOF
            if LOF < LOF_th:
                for p in range(mode):
                    preds.append(predpts[m])
                    gpt1.append(pt1[m * mode + p])
                    gpt2.append(pt2[m * mode + p])
            else:
                pass
            return gpt1, gpt2, L / len(index)


def predPoint(pt1, pt2, mode):
    predpts = []
    n = len(pt1)

    for i in range(int(n / mode)):
        cpt = []
        mpt = []

        for j in range(mode):
            cpt.append(pt1[i * mode + j])
            mpt.append(pt2[i * mode + j])

        tform = tformCompute(cpt, mpt, mode)
        pred = tform * np.matrix([[255.5], [255.5], [1.0]])
        outpt = [float(pred[0][0] / pred[2][0]), float(pred[1][0] / pred[2][0])]
        predpts.append(outpt)
    x, y = split_xy(predpts)
    return x, y


def isolationForest(pt1, pt2, mode):
    predpts = []
    n = len(pt1)

    gpt1 = []
    gpt2 = []

    for i in range(int(n / mode)):
        cpt = []
        mpt = []

        for j in range(mode):
            cpt.append(pt1[i * mode + j])
            mpt.append(pt2[i * mode + j])

        tform = tformCompute(cpt, mpt, mode)
        pred = tform * np.matrix([[255.5], [255.5], [1.0]])
        outpt = [float(pred[0][0] / pred[2][0]), float(pred[1][0] / pred[2][0])]
        predpts.append(outpt)

    predpts = np.array(predpts)

    df = pd.DataFrame(predpts, columns=["x", "y"])

    clf = IsolationForest(n_estimators=100, random_state=123)
    clf.fit(df)
    df["predict"] = clf.predict(df)

    i = 0
    while i < int(n / mode):
        if int(df.at[df.index[i], "predict"]) == 1:
            for p in range(mode):
                gpt1.append(pt1[i * mode + p])
                gpt2.append(pt2[i * mode + p])
        else:
            pass
        i += 1
    return gpt1, gpt2

