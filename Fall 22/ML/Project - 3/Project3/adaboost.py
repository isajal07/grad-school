import numpy as np 

def calulateAlpha(e):
	if e == 0:
		return 10000
	elif e == 0.5:
		return 0.001
	else:
		return 0.5 * np.log((1-e)/e)

def calulateWeights(W,alpha,y,pred):
	newWeights=[]
	for i in range(len(y)):
		newWeights.append(W[i] * np.exp(-1 * alpha * y[i] * pred[i]))
	return np.array(newWeights/sum(newWeights)).reshape([len(y),1])

def calulateErrorRate(y,pred,W):
	ret=0
	for i in range(len(y)):
		if y[i]!=pred[i]:
			ret+=W[i]
	return ret

def calulateFinalPrediction(i,alpha,pred,y):
    ret=np.array([0.0]*len(y))
    for j in range(i+1):
        ret+=alpha[j]*pred[j]
    return np.sign(ret)

def calculateFinalErrorRate(y,cal_final_predict):
    ret=0
    for i in range(len(y)):
        if y[i]==cal_final_predict[i]:
            ret+=1
    return ret/len(y)

def calulateDic(X):
		retGt={}
		for i in range(X.shape[1]):
			retGt[i]=[]
			for j in range(X.shape[0]):
				tempThreshold=X[j,i]
				tempLine=[]
				for k in range(X.shape[0]):
					if X[k,i]>=tempThreshold:
						tempLine.append(1)
					else:
						tempLine.append(-1)
				retGt[i].append(tempLine)

		retLt={}
		for i in range(X.shape[1]):
			retLt[i]=[]
			for j in range(X.shape[0]):
				tempThreshold=X[j,i]
				tempLine=[]
				for k in range(X.shape[0]):
					if X[k,i]<=tempThreshold:
						tempLine.append(1)
					else:
						tempLine.append(-1)
				retLt[i].append(tempLine)
		ret={}
		ret['gt']=retGt
		ret['lt']=retLt
		return ret

#calculate error for one dimension array
def calculate1DArrayError(We,y,line):
		ret=0
		for i in range(len(y)):
			if y[i]!=line[i]:
				ret = ret + We[i]
		return ret

#calculate error for two dimension array
def calculate2DArrayError(We, y,lines):
		ret=[]
		for i in lines:
			ret.append(calculate1DArrayError(We, y,i))
		return ret

#calculate error for all possible data and get e_dic
def calulateAllError(We,y,dic):
		retGt={}
		for i in dic['gt']:
			retGt[i]=(calculate2DArrayError(We, y,dic['gt'][i]))
		retLt={}
		for i in dic['lt']:
			retLt[i]=(calculate2DArrayError(We, y,dic['lt'][i]))
		ret={}
		ret['gt']=retGt
		ret['lt']=retLt
		return ret

#select min error for e_dic
def calulateMinError(eDic):
		ret=100000
		for key in eDic:
			for i in eDic[key]:
				temp=min(eDic[key][i])
				if ret>temp:
					ret=temp
		for key in eDic:
			for i in eDic[key]:
				if ret == min(eDic[key][i]):
					return key,i,eDic[key][i].index(ret)

def adaboost_train(X,Y,M):
    X = np.array(X)
    Y = np.array(Y)
    W={}
    alpha={}
    pred={}
    
    for i in range(M):
        W.setdefault(i)
        alpha.setdefault(i)
        pred.setdefault(i)
    
    for i in range(M):
        #for the first iteration,initial W
        if i == 0:
            W[i]=np.array([1]*len(Y))/len(Y)
            W[i]=W[i].reshape([len(Y),1])
		#if not the first iteration,calculate new Weight
        else:
            W[i]=calulateWeights(W[i-1],alpha[i-1],Y,pred[i-1])
     
        We = W[i]
        dic = calulateDic(X)
        errorDic = calulateAllError(We,Y,dic)
        decision_key, decision_feature,e_min_i = calulateMinError(errorDic)
        pred[i]=dic[decision_key][decision_feature][e_min_i]
        
        err=calulateErrorRate(Y,pred[i],W[i])
        
        alpha[i]=calulateAlpha(err)

        cal_final_predict=calulateFinalPrediction(i,alpha,pred,Y)
        
        # print('iteration:%d'%(i+1))
        # print('decision_key=%s'%(decision_key))
        # print('decision_feature=%d'%(decision_feature))
        # print('W=%s'%(W[i]))
        # print('pred=%s'%(pred[i]))
        # print('e:%f alpha:%f'%(e,alpha[i]))
        # print('cal_final_predict:%s'%(cal_final_predict))
        # print('calculateFinalErrorRate:%s%%'%(calculateFinalErrorRate(Y,cal_final_predict)*100))
   
        if calculateFinalErrorRate(Y,cal_final_predict)==0 or err==0:
            break
    
    return cal_final_predict, alpha[i]

def adaboost_test(X,Y,f,alpha):
    return  calculateFinalErrorRate(Y,f) 

    