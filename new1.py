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
import sys
sys.path.append('SuperPointPretrainedNetwork')
from demo_superpoint import SuperPointFrontend
LOF_ave = 0
from tqdm import tqdm
rootfolder = "/home/natori21_u/JAXA_database/"
dir_path = "512/jpg8k/"

cuda = True
weights_path = 'SuperPointPretrainedNetwork/superpoint_v1.pth'
nms_dist = 1
#conf_thresh = 0.00005
conf_thresh = 0.005
#conf_thresh=0.015
nn_thresh = 0.1


def getPairs(img1, fe):
    # find the keypoints and descriptors with SIFT
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    kp1, des1, _ = fe.run(img1.astype('float32')/255)


    kp1 = [cv2.KeyPoint(k[0], k[1], k[2]) for k in kp1.transpose(1,0)]
    #kp2 = [cv2.KeyPoint(k[0], k[1], k[2]) for k in kp2.transpose(1,0)]

    #print(kp1)
    #print(kp2)


    return kp1,des1



fe = SuperPointFrontend(weights_path=weights_path,
                          nms_dist=nms_dist,
                          conf_thresh=conf_thresh,
                          nn_thresh=nn_thresh,
                          cuda=cuda)



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


ng_lis = []
ng_lis1 = []
ng_lis2 = []
ng_lis3 = []

f3 = open("s_Matching" + str(noise) + ".csv", "w")
img2 = cv2.imread(img2_path)
mpt, mf = getPairs(img2, fe)
# print(mpt)  #keypoint
#print(mf)

for l in tqdm(range(len(data))):
    st=time.time()
    path1 = img1_path + data[l].rstrip("\n")
    print(path1)
    true_point = getTruePoint(data2)
    img1 = cv2.imread(path1)
    c_pt, cf = getPairs(img1, fe)

    phase = "RatioTest"
    label1 = phase
    #c_pt, c_f, m_pt, m_f = matchRatio(c_pt, cf, mpt, mf, knn, ratio)
    c_pt,  m_pt = matchRatio(c_pt, cf, mpt, mf, knn, ratio)
    phase = "Ransac"
    label3 = phase

    c_pt, m_pt = ransac(c_pt, m_pt, mode)



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
    if d < limitpx**2:
        suc_mum += 1

    print(suc_mum)

goal = time.time()
score = goal - start
print(int(score / 3600), "時間", int((score % 3600) / 60), "分", score % 60, "秒")







