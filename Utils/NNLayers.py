import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np

paramId = 0               # 用来标识生成的名字
biasDefault = False
params = {}               # 全局的tf.variable变量字典
regParams = {}            # 全局的正则化参数字典
ita = 0.2
leaky = 0.1

def getParamId():
	global paramId
	paramId += 1
	return paramId

def setIta(ITA):
	ita = ITA

def setBiasDefault(val):
	global biasDefault
	biasDefault = val

def getParam(name):
	return params[name]

def addReg(name, param):
	global regParams
	if name not in regParams:
		regParams[name] = param
	else:
		print('ERROR: Parameter already exists')

def addParam(name, param):
	global params
	if name not in params:
		params[name] = param

def defineRandomNameParam(shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	name = 'defaultParamName%d'%getParamId()
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def defineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):    # 返回一个ret,就是一个tf.get_variable的变量
	global params
	global regParams
	assert name not in params, 'name %s already exists' % name   # 这行代码是一个断言语句，用于检查变量 name 是否已经存在于字典 params 中。如果 name 已存在，程序会抛出一个错误，并显示消息 "name [变量名] already exists"。如果 name 不存在，代码继续执行。
	if initializer == 'xavier':
		ret = tf.get_variable(name=name, dtype=dtype, shape=shape,
			initializer=xavier_initializer(dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'trunc_normal':
		ret = tf.get_variable(name=name, initializer=tf.random.truncated_normal(shape=[int(shape[0]), shape[1]], mean=0.0, stddev=0.03, dtype=dtype))
	elif initializer == 'zeros':
		ret = tf.get_variable(name=name, dtype=dtype,
			initializer=tf.zeros(shape=shape, dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'ones':
		ret = tf.get_variable(name=name, dtype=dtype, initializer=tf.ones(shape=shape, dtype=tf.float32), trainable=trainable)
	elif not isinstance(initializer, str):
		ret = tf.get_variable(name=name, dtype=dtype,
			initializer=initializer, trainable=trainable)
	else:
		print('ERROR: Unrecognized initializer')
		exit()
	params[name] = ret
	if reg:
		regParams[name] = ret
	return ret

def getOrDefineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True, reuse=False): # 这段代码的目的是检查全局参数字典 params 中是否已经存在名为 name 的参数。如果已经存在，则根据 reuse 参数决定是否复用该参数，同时如果需要正则化 (reg=True) 且该参数还未加入正则化参数字典 regParams，则将其加入。如果参数存在并且不需要重新使用，直接返回已存在的参数。  
	global params
	global regParams
	if name in params:
		assert reuse, 'Reusing Param %s Not Specified' % name
		if reg and name not in regParams:
			regParams[name] = params[name]
		return params[name]
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def BN(inp, name=None):
	global ita
	dim = inp.get_shape()[1]
	name = 'defaultParamName%d'%getParamId()
	scale = tf.Variable(tf.ones([dim]))
	shift = tf.Variable(tf.zeros([dim]))
	fcMean, fcVar = tf.nn.moments(inp, axes=[0])
	ema = tf.train.ExponentialMovingAverage(decay=0.5)
	emaApplyOp = ema.apply([fcMean, fcVar])
	with tf.control_dependencies([emaApplyOp]):
		mean = tf.identity(fcMean)
		var = tf.identity(fcVar)
	ret = tf.nn.batch_normalization(inp, mean, var, shift,
		scale, 1e-8)
	return ret

def FC(inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None, initializer='xavier', reuse=False, biasReg=False, biasInitializer='zeros'):     # 返回一个经过WX+b,然后再经过激活函数，得到所求的
	global params
	global regParams
	global leaky
	inDim = inp.get_shape()[1]     # 获取最后一个维度，这里就是获取ego_embedding(user.number+item.number,d)的d维
	temName = name if name!=None else 'defaultParamName%d'%getParamId() 
	# 如果 name 不为 None，则 temName 的值为 name。如果 name 为 None，则 temName 的值为 'defaultParamName%d' % getParamId()。
	W = getOrDefineParam(temName, [inDim, outDim], reg=reg, initializer=initializer, reuse=reuse)
	if dropout != None:
		ret = tf.nn.dropout(inp, rate=dropout) @ W
	else:
		ret = inp @ W   # 矩阵乘法     ----> 结果输出是(user+item)*outDim,outDim就是行为的类型。，也就是得到每个节点的注意力权重
	if useBias:
		ret = Bias(ret, name=name, reuse=reuse, reg=biasReg, initializer=biasInitializer)   # 这里加个b
	if useBN:
		ret = BN(ret)
	if activation != None:
		ret = Activate(ret, activation)              # 这里过个激活函数
	return ret  

def Bias(data, name=None, reg=False, reuse=False, initializer='zeros'):
	inDim = data.get_shape()[-1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	temBiasName = temName + 'Bias'
	bias = getOrDefineParam(temBiasName, inDim, reg=False, initializer=initializer, reuse=reuse)
	if reg:
		regParams[temBiasName] = bias
	return data + bias     # 广播机制

def ActivateHelp(data, method):           # 返回经过激活函数的值
	if method == 'relu':
		ret = tf.nn.relu(data)
	elif method == 'sigmoid':
		ret = tf.nn.sigmoid(data)
	elif method == 'tanh':
		ret = tf.nn.tanh(data)
	elif method == 'softmax':
		ret = tf.nn.softmax(data, axis=-1)
	elif method == 'leakyRelu':
		ret = tf.maximum(leaky*data, data)  # Returns the max of x and y (i.e. x > y ? x : y) element-wise. 大于0就是data    小于0就是0.1*data
	elif method == 'twoWayLeakyRelu6':
		temMask = tf.to_float(tf.greater(data, 6.0))
		ret = temMask * (6 + leaky * (data - 6)) + (1 - temMask) * tf.maximum(leaky * data, data)
	elif method == '-1relu':
		ret = tf.maximum(-1.0, data)
	elif method == 'relu6':
		ret = tf.maximum(0.0, tf.minimum(6.0, data))
	elif method == 'relu3':
		ret = tf.maximum(0.0, tf.minimum(3.0, data))
	else:
		raise Exception('Error Activation Function')
	return ret

def Activate(data, method, useBN=False):     # 激活函数
	global leaky
	if useBN:
		ret = BN(data)
	else:
		ret = data
	ret = ActivateHelp(ret, method)
	return ret

def Regularize(names=None, method='L2'):
	ret = 0
	if method == 'L1':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.abs(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.abs(regParams[name]))
	elif method == 'L2':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.square(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.square(regParams[name]))
	return ret

def Dropout(data, rate):
	if rate == None:
		return data
	else:
		return tf.nn.dropout(data, rate=rate)

def selfAttention(localReps, number, inpDim, numHeads):
	Q = defineRandomNameParam([inpDim, inpDim], reg=True)
	K = defineRandomNameParam([inpDim, inpDim], reg=True)
	V = defineRandomNameParam([inpDim, inpDim], reg=True)
	rspReps = tf.reshape(tf.stack(localReps, axis=1), [-1, inpDim])
	q = tf.reshape(rspReps @ Q, [-1, number, 1, numHeads, inpDim//numHeads])
	k = tf.reshape(rspReps @ K, [-1, 1, number, numHeads, inpDim//numHeads])
	v = tf.reshape(rspReps @ V, [-1, 1, number, numHeads, inpDim//numHeads])
	att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(inpDim/numHeads), axis=2)
	attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
	rets = [None] * number
	paramId = 'dfltP%d' % getParamId()
	for i in range(number):
		tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
		# tem2 = FC(tem1, inpDim, useBias=True, name=paramId+'_1', reg=True, activation='relu', reuse=True) + localReps[i]
		rets[i] = tem1 + localReps[i]
	return rets

def lightSelfAttention(localReps, number, inpDim, numHeads):
	Q = defineRandomNameParam([inpDim, inpDim], reg=True)
	rspReps = tf.reshape(tf.stack(localReps, axis=1), [-1, inpDim])
	tem = rspReps @ Q
	q = tf.reshape(tem, [-1, number, 1, numHeads, inpDim//numHeads])
	k = tf.reshape(tem, [-1, 1, number, numHeads, inpDim//numHeads])
	v = tf.reshape(rspReps, [-1, 1, number, numHeads, inpDim//numHeads])
	# att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) * tf.sqrt(inpDim/numHeads), axis=2)
	att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(inpDim/numHeads), axis=2)
	attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
	rets = [None] * number
	paramId = 'dfltP%d' % getParamId()
	for i in range(number):
		tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
		# tem2 = FC(tem1, inpDim, useBias=True, name=paramId+'_1', reg=True, activation='relu', reuse=True) + localReps[i]
		rets[i] = tem1 + localReps[i]
	return rets#, tf.squeeze(att)