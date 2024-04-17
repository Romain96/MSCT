import numpy as np
import cv2
import helper

def DTECMA(binary, distanceThreshold = 1, lam = 1,
           ratio = np.arange(1, 4, 0.2), angle = np.arange(0, 180, 5),
           minArea = None, percentile = [30,50,70], coverRate = 0.95):
    n, label, stats, _ = cv2.connectedComponentsWithStats(binary)
    #print("CCstats")
    ellipses = []
    for i in range(1, n):
        #print(f"{i} in range")
        x, y, dx, dy, _ = stats[i]
        clump = label[y:(y+dy),x:(x+dx)]
        clump = np.array(clump == i, dtype = np.uint8)
        cntrs, _ = cv2.findContours(clump, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)     # Get rid of small contours ???
        #print("contours")
        cocvs = [[] for _ in cntrs]
        for j, cntr in enumerate(cntrs):
            poly = cv2.approxPolyDP(cntr, distanceThreshold, True)
            cocv = helper.concaveDetect(poly)
            for k, p in enumerate(cntr):
                if (p[0,0], p[0,1]) in cocv:
                    cocvs[j].append(k)
        cntrImg = np.zeros(clump.shape, dtype=np.uint8)
        _ = cv2.drawContours(cntrImg, cntrs, -1, 1, 1, cv2.LINE_4)
        #print("find ellipse")
        elist = helper.findEllipse(clump, ratio, angle, minArea)
        #print("overlay")
        elist = helper.overlay(cntrImg, elist, percentile, coverRate)
        #print("dist")
        distMat = helper.computeDist(elist, cntrs, cocvs, clump.shape)
        clumpSqrtArea = np.sqrt(clump.sum())
        #print("integerprogramming")
        if len(elist) < 1:
            return fix_when_empty(binary)
        idxOpt = helper.intergerProgamming(distMat, lam * clumpSqrtArea)
        elist = elist[idxOpt,:].reshape(-1,5)
        elist[:,0] += y
        elist[:,1] += x
        ellipses.append(elist)
    ellipses = np.concatenate(ellipses)
    #print("PLOT")
    #output = helper.plotEllipse(binary, ellipses)
    #return ellipses, output
    return ellipses

def fix_when_empty(binary):
    coords = np.nonzero(binary)
    rows = [i for i in coords[0]]
    cols = [i for i in coords[1]]
    centroid = [(int(round(sum(rows) / len(rows))), int(round(sum(cols) / len(cols))))]
    return centroid














