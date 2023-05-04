import glob
import cv2
from func import AKAZE
import time
from func import *
suc_mum=0
knn = 2
ratio = 0.95  # 射影なら0.84
limitpx = 3
ransacnum = 5000
AKAZE_th = -0.000005
sgms = 1.0

LOF_ave = 0

rootfolder = "/home/natori21_u/JAXA_database/"
dir_path = "512/jpg8k/"

noise = 400
start = time.time()
mode = 2
# img1_path=rootfolder+dir_path+str(noise)+"/*"
img1_path = rootfolder + dir_path + str(noise) + "/"
img2_path = rootfolder + "TCO_CST1_TM_SIM_a7351_i3438_h36000_lanczos3.bmp"
truepoint_path = rootfolder + dir_path + str(noise) + "/" + "true_point.csv"

# imgファイルの名前読み込み
# f = open(img1_path + "imgfile.txt")
f = open(rootfolder + dir_path + str(noise) + "/" + "imgfile.txt")
data = f.readlines()
f.close()

del f

# 真値の値を持っているファイルの読み込み
f2 = open(truepoint_path)
data2 = f2.readlines()
f2.close()

del f2

Scount = 0

f3 = open("s_Matching" + str(noise) + ".csv", "w")

img2 = cv2.imread(img2_path)
print(img2.shape)
mpt, mf = AKAZE(img2)
#print(mpt)
print(mf)

ng_lis = []
ng_lis1 = []
ng_lis2 = []
ng_lis3 = []
print(img1_path)

for l in range(len(data)):
    st = time.time()
    # 実際にマッチング
    path1 = img1_path + data[l].rstrip("\n")
    print(path1)

    img1 = cv2.imread(path1)
    # print(img1)
    # 真値の読み込み
    true_point = getTruePoint(data2)

    # 特徴量抽出
    phase = "特徴量抽出"
    c_pt, cf = AKAZE(img1)
    ##print(c_pt.shape)  #list
    #print(cf.shape)#list
    print(len(c_pt))
    print(len(cf))

    # ratiotest
    phase = "RatioTest"
    label1 = phase
    c_pt, c_f, m_pt, m_f = matchRatio(c_pt, cf, mpt, mf, knn, ratio)
    print(c_pt)
    print(m_pt)

    # LOF
    phase = "LOF"
    # c_pt, m_pt, lof = LOF(c_pt, m_pt, k=30, LOF_th=2.5, mode=mode, true_point=true_point)

    # LOF_ave += lof / 1000

    # ransac
    phase = "Ransac"
    label3 = phase
    c_pt, m_pt = ransac(c_pt, m_pt, mode)

    #print(c_pt)
    #print(m_pt)


    try:
        tform = tformCompute(c_pt, m_pt, mode, num=len(c_pt))
    except IndexError:
        ng_lis2.append(data[l])
        check = False
    else:
        check, d = judge(true_point, tform, limitpx=3)
        if check == False:
            ng_lis3.append(data[l].rstrip("\n"))

        else:
            pass
    # f3.write("\n" + str(d) + "," + str(lof))
    f3.write("\n" + str(d) + ",")
    gl = time.time()
    if d<3:
        suc_mum+=1

    print(suc_mum)

goal = time.time()
score = goal - start
print(int(score / 3600), "時間", int((score % 3600) / 60), "分", score % 60, "秒")

