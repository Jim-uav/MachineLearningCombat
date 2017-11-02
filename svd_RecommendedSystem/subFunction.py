# coding: utf-8
from numpy import *
from numpy import linalg as la

def ecludSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))

def pearSim(inA,inB):
    if len(inA) <3 : return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):
    num=float(inA.T*inB)
    denom=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def standEst(dataMat,user,simMeas,item):
    n=shape(dataMat)[1]
    simTotal=0.0
    ratSimTotal=0.0
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0:
            continue
        overLap=nonzero(logical_and(dataMat[:,item].A>0, dataMat[:,j].A>0))[0]
        if len(overLap)==0:
            similarity=0
        else:
            similarity=simMeas(dataMat[overLap,item],dataMat[overLap,j])
        simTotal+=similarity
        ratSimTotal+=similarity*userRating
    if simTotal==0:
        return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    unratedItems=nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems)==0:
        return 'you rated everything'
    itemScores=[]
    for item in unratedItems:
        estimatedScore=estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]

def svdEst(dataMat,user,simMeas,item):
    n=shape(dataMat)[1]
    simTotal=0.0
    ratSimTotal=0.0
    U, Sigma, VT=la.svd(dataMat)

    # 下面两行的计算结果是一样的，第二行参考链接：http://www.cnblogs.com/who-a/p/5649787.html
    Sig4 = mat(eye(4)*Sigma[:4])
    # Sig4 = np.diag(Sigma)

    # 下面这两行的计算出的结果是一样的，第二行参考链接：http://www.cnblogs.com/who-a/p/5649787.html
    xformedItems = dataMat.T*U[:,:4]*Sig4.I
    # xformedItems = np.dot(np.dot(U,Sig4), VT)

    # print xformedItems
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0 or j==item:
            continue
        similarity=simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' % (item,j,similarity))
        simTotal+=similarity
        ratSimTotal+=similarity*userRating
    if simTotal==0:
        return 0
    else:
        return ratSimTotal/simTotal


# author:qth
# def stanEst(dataMat,user,simMea,item):
#     n=shape(dataMat)[1]
#     simTotal=0.0
#     ratingTotal=0.0
#     for j in range(n):
#         ratingItem = dataMat[user, j]
#         if ratingItem == 0:
#             continue
#         overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
#         if len(overLap)==0:
#             simScore=0.0
#         else:
#             simScore=simMea(dataMat[overLap,item],dataMat[overLap,j])
#         simTotal+=simScore
#         ratingTotal+=simScore*ratingItem
#     if simTotal==0:
#         return 0
#     else:
#         return ratingTotal/simTotal
#
# def recommend(dataMat,user,N=3,simMea=cosSim,estMethod=stanEst):
#     unratedItem=nonzero(dataMat[user,:].A==0)[1]
#     if len(unratedItem)==0:
#         return 'you rated everything'
#     itemScores = []
#     for item in unratedItem:
#         estimateScore=estMethod(dataMat,user,simMea,item)
#         itemScores.append((item,estimateScore))
#     return sorted(itemScores,key= lambda jj:jj[1],reverse=True)[:N]

# def svdEst(dataMat,user,simMea,item):
#     n=shape(dataMat)[1]
#     U,Sigma,VT=la.svd(dataMat)
#     newSigma=eye(n)*Sigma[:3]
#     newDataMat=U*newSigma*VT