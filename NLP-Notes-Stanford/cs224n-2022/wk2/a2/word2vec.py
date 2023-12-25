#!/usr/bin/env python

import argparse
import numpy as np
import random

from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax


def sigmoid(x):
	"""
	Compute the sigmoid function for the input here.
	Arguments:
	x -- A scalar or numpy array.
	Return:
	s -- sigmoid(x)
	"""

	### YOUR CODE HERE (~1 Line)
	s = 1. / (1. + np.exp(-x))
	### END YOUR CODE

	return s


def naiveSoftmaxLossAndGradient(
	centerWordVec,
	outsideWordIdx,
	outsideVectors,
	dataset
):
	""" Naive Softmax loss & gradient function for word2vec models

	Implement the naive softmax loss and gradients between a center word's 
	embedding and an outside word's embedding. This will be the building block
	for our word2vec models. For those unfamiliar with numpy notation, note 
	that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
	you can effectively treat as a vector with length x.

	Arguments:
	centerWordVec -- numpy ndarray, center word's embedding
					in shape (word vector length, )
					(v_c in the pdf handout)
	outsideWordIdx -- integer, the index of the outside word
					(o of u_o in the pdf handout)
	outsideVectors -- outside vectors is
					in shape (num words in vocab, word vector length) 
					for all words in vocab (tranpose of U in the pdf handout)
	dataset -- needed for negative sampling, unused here.

	Return:
	loss -- naive softmax loss
	gradCenterVec -- the gradient with respect to the center word vector
					 in shape (word vector length, )
					 (dJ / dv_c in the pdf handout)
	gradOutsideVecs -- the gradient with respect to all the outside word vectors
					in shape (num words in vocab, word vector length) 
					(dJ / dU)
	"""

	### YOUR CODE HERE (~6-8 Lines)

	### Please use the provided softmax function (imported earlier in this file)
	### This numerically stable implementation helps you avoid issues pertaining
	### to integer overflow. 
	
	### 提示: 参考assignment1中written部分naive-softmax的损失表达式, 以及(b)(c)(d)三问的结论
	
	scores = np.matmul(outsideVectors, centerWordVec)					# 计算得分向量 U' * v_c
	probs = softmax(scores)                          					# 计算概率分布向量 P(O=w|C=c) w=1,2,...,|V|
	loss = -np.log(probs[outsideWordIdx])								# 计算naive-softmax目标函数表达式
	y_hat = probs.copy()												# y_hat = P(O|C=c)
	y_hat[outsideWordIdx] = y_hat[outsideWordIdx] - 1					# 计算预测值与真实值的偏差: y_hat - y
	gradCenterVec = np.matmul(outsideVectors.T, y_hat)  				# 计算关于v_c的偏导(written部分的(b)问): U' * (y_hat - y)
	gradOutsideVecs = np.outer(y_hat, centerWordVec)					# 计算关于U的偏导(written部分的(c)(d)问): (y_hat - y)' * v_c 
	### END YOUR CODE

	return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
	""" Samples K indexes which are not the outsideWordIdx """

	negSampleWordIndices = [None] * K
	for k in range(K):
		newidx = dataset.sampleTokenIdx()
		while newidx == outsideWordIdx:
			newidx = dataset.sampleTokenIdx()
		negSampleWordIndices[k] = newidx
	return negSampleWordIndices


def negSamplingLossAndGradient(
	centerWordVec,
	outsideWordIdx,
	outsideVectors,
	dataset,
	K=10
):
	""" Negative sampling loss function for word2vec models

	Implement the negative sampling loss and gradients for a centerWordVec
	and a outsideWordIdx word vector as a building block for word2vec
	models. K is the number of negative samples to take.

	Note: The same word may be negatively sampled multiple times. For
	example if an outside word is sampled twice, you shall have to
	double count the gradient with respect to this word. Thrice if
	it was sampled three times, and so forth.

	Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
	"""

	# Negative sampling of words is done for you. Do not modify this if you
	# wish to match the autograder and receive points!
	negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
	indices = [outsideWordIdx] + negSampleWordIndices

	### YOUR CODE HERE (~10 Lines)

	### Please use your implementation of sigmoid in here.
	
	### 提示: 可以参考assignment1中written部分neg-sample的损失表达式, 以及(g)问的结论
	gradCenterVec = np.zeros(centerWordVec.shape)						# 初始化v_c偏导值
	gradOutsideVecs = np.zeros(outsideVectors.shape)					# 初始化u_o偏导值
	u_o = outsideVectors[outsideWordIdx]								# 取出U中所有语境词的词向量
	score = sigmoid(np.dot(u_o, centerWordVec))							# 计算\sigma(u_o^\top v_c)
	loss = -np.log(score)												# 损失函数解析式的第一部分: -\log(\sigma(u_o^\top v_c))
	gradCenterVec = (score - 1)	* u_o									# v_c偏导解析式的第一部分: (\sigma(u_o^\top v_c) - 1)u_o
	gradOutsideVecs[outsideWordIdx] = (score - 1) * centerWordVec		# u_o偏导解析式: (\sigma(u_o^\top v_c) - 1)v_c
	for i in range(K):
		u_w_s = outsideVectors[indices[i + 1]]							# 取得负样本的词向量
		score = sigmoid(np.dot(-u_w_s, centerWordVec))					# 计算\sigma(-u_{w_s}^\top v_c)
		loss -= np.log(score)											# 损失函数解析式的第二部分(求和式): -\log(-\sigma(u_{w_s}^\top v_c))
		gradCenterVec += (1 - score) * u_w_s							# v_c偏导解析式的第二部分(求和式): (1 - \sigma(-u_{w_s}^\top v_c))u_{w_s}
	### END YOUR CODE

	return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
			 centerWordVectors, outsideVectors, dataset,
			 word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
	""" Skip-gram model in word2vec

	Implement the skip-gram model in this function.

	Arguments:
	currentCenterWord -- a string of the current center word
	windowSize -- integer, context window size
	outsideWords -- list of no more than 2*windowSize strings, the outside words
	word2Ind -- a dictionary that maps words to their indices in
			  the word vector list
	centerWordVectors -- center word vectors (as rows) is in shape 
						(num words in vocab, word vector length) 
						for all words in vocab (V in pdf handout)
	outsideVectors -- outside vectors is in shape 
						(num words in vocab, word vector length) 
						for all words in vocab (transpose of U in the pdf handout)
	word2vecLossAndGradient -- the loss and gradient function for
							   a prediction vector given the outsideWordIdx
							   word vectors, could be one of the two
							   loss functions you implemented above.

	Return:
	loss -- the loss function value for the skip-gram model
			(J in the pdf handout)
	gradCenterVec -- the gradient with respect to the center word vector
					 in shape (num words in vocab, word vector length)
					 (dJ / dv_c in the pdf handout)
	gradOutsideVecs -- the gradient with respect to all the outside word vectors
					in shape (num words in vocab, word vector length) 
					(dJ / dU)
	"""

	loss = 0.0
	gradCenterVecs = np.zeros(centerWordVectors.shape)
	gradOutsideVectors = np.zeros(outsideVectors.shape)

	### YOUR CODE HERE (~8 Lines)
	
	### 提示: 可以参考assignment1中written部分(i)问的结论
	### 一直无法通过neg-sample的测试, 其他测试都可以通过, 感觉非常的奇怪, 是否测试有问题?
	center_word_id = word2Ind[currentCenterWord]
	center_word_vector = centerWordVectors[center_word_id]
	for outside_word in outsideWords:
		outside_word_id = word2Ind[outside_word]
		_loss, gradCenterVec, gradOutsideVecs = word2vecLossAndGradient(centerWordVec=center_word_vector, outsideWordIdx=outside_word_id, outsideVectors=outsideVectors, dataset=dataset)
		loss += _loss
		gradCenterVecs[center_word_id] += gradCenterVec
		gradOutsideVectors += gradOutsideVecs
	### END YOUR CODE
	
	return loss, gradCenterVecs, gradOutsideVectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
						 windowSize,
						 word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
	batchsize = 50
	loss = 0.0
	grad = np.zeros(wordVectors.shape)
	N = wordVectors.shape[0]
	centerWordVectors = wordVectors[:int(N/2),:]
	outsideVectors = wordVectors[int(N/2):,:]
	for i in range(batchsize):
		windowSize1 = random.randint(1, windowSize)
		centerWord, context = dataset.getRandomContext(windowSize1)

		c, gin, gout = word2vecModel(
			centerWord, windowSize1, context, word2Ind, centerWordVectors,
			outsideVectors, dataset, word2vecLossAndGradient
		)
		loss += c / batchsize
		grad[:int(N/2), :] += gin / batchsize
		grad[int(N/2):, :] += gout / batchsize

	return loss, grad

def test_sigmoid():
	""" Test sigmoid function """
	print("=== Sanity check for sigmoid ===")
	assert sigmoid(0) == 0.5
	assert np.allclose(sigmoid(np.array([0])), np.array([0.5]))
	assert np.allclose(sigmoid(np.array([1,2,3])), np.array([0.73105858, 0.88079708, 0.95257413]))
	print("Tests for sigmoid passed!")

def getDummyObjects():
	""" Helper method for naiveSoftmaxLossAndGradient and negSamplingLossAndGradient tests """

	def dummySampleTokenIdx():
		return random.randint(0, 4)

	def getRandomContext(C):
		tokens = ["a", "b", "c", "d", "e"]
		return tokens[random.randint(0,4)], \
			[tokens[random.randint(0,4)] for i in range(2*C)]

	dataset = type('dummy', (), {})()
	dataset.sampleTokenIdx = dummySampleTokenIdx
	dataset.getRandomContext = getRandomContext

	random.seed(31415)
	np.random.seed(9265)
	dummy_vectors = normalizeRows(np.random.randn(10,3))
	dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

	return dataset, dummy_vectors, dummy_tokens

def test_naiveSoftmaxLossAndGradient():
	""" Test naiveSoftmaxLossAndGradient """
	dataset, dummy_vectors, dummy_tokens = getDummyObjects()

	print("==== Gradient check for naiveSoftmaxLossAndGradient ====")
	def temp(vec):
		loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(vec, 1, dummy_vectors, dataset)
		return loss, gradCenterVec
	gradcheck_naive(temp, np.random.randn(3), "naiveSoftmaxLossAndGradient gradCenterVec")

	centerVec = np.random.randn(3)
	def temp(vec):
		loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(centerVec, 1, vec, dataset)
		return loss, gradOutsideVecs
	gradcheck_naive(temp, dummy_vectors, "naiveSoftmaxLossAndGradient gradOutsideVecs")

def test_negSamplingLossAndGradient():
	""" Test negSamplingLossAndGradient """
	dataset, dummy_vectors, dummy_tokens = getDummyObjects()

	print("==== Gradient check for negSamplingLossAndGradient ====")
	def temp(vec):
		loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(vec, 1, dummy_vectors, dataset)
		return loss, gradCenterVec
	gradcheck_naive(temp, np.random.randn(3), "negSamplingLossAndGradient gradCenterVec")

	centerVec = np.random.randn(3)
	def temp(vec):
		loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(centerVec, 1, vec, dataset)
		return loss, gradOutsideVecs
	gradcheck_naive(temp, dummy_vectors, "negSamplingLossAndGradient gradOutsideVecs")

def test_skipgram():
	""" Test skip-gram with naiveSoftmaxLossAndGradient """
	dataset, dummy_vectors, dummy_tokens = getDummyObjects()

	print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
	gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
		skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
		dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
	grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

	print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
	gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
		skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
		dummy_vectors, "negSamplingLossAndGradient Gradient")
	grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient)

def test_word2vec():
	""" Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
	test_sigmoid()
	test_naiveSoftmaxLossAndGradient()
	test_negSamplingLossAndGradient()
	test_skipgram()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Test your implementations.')
	parser.add_argument('function', nargs='?', type=str, default='all',
						help='Name of the function you would like to test.')

	args = parser.parse_args()
	if args.function == 'sigmoid':
		test_sigmoid()
	elif args.function == 'naiveSoftmaxLossAndGradient':
		test_naiveSoftmaxLossAndGradient()
	elif args.function == 'negSamplingLossAndGradient':
		test_negSamplingLossAndGradient()
	elif args.function == 'skipgram':
		test_skipgram()
	elif args.function == 'all':
		test_word2vec()
