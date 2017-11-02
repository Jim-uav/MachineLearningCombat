from numpy import *
from svd_RecommendedSystem import subFunction
from svd_RecommendedSystem import dataSet

myMat = mat(dataSet.loadExData())

# for testing recommendation system
myMat[0, 1] = myMat[0, 0]=myMat[1,0]=myMat[2,0]=4
myMat[3, 3] = 2
print(myMat)
print(subFunction.recommend(myMat, 2))

# print subFunction.recommend(myMat,2,simMeas=subFunction.ecludSim)
# print subFunction.recommend(myMat,2,simMeas=subFunction.pearsSim)

# for testing Recommendation System using SVD method
# print myMat
# print subFunction.recommend(myMat,1,estMethod=subFunction.svdEst)
# print subFunction.recommend(myMat,1,estMethod=subFunction.svdEst,simMeas=subFunction.pearSim)
# print subFunction.recommend(myMat,1)