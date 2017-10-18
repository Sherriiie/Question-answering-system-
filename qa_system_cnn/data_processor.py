# -*- coding: UTF-8 -*-

import numpy as np
import random
import os


def buildVocab(file_path):
	vocab = {}
	code = 0
	vocab['UNKNOWN'] = int(code)
	for line in open(file_path + 'data/train'):
		items = line.strip().split(' ')
		for i in range(2, 4):
			words = items[i].split('_')
			for word in words:
				if not word in vocab:
					code += 1
					vocab[word] = code

	for line in open(file_path + 'data/val'):
		items = line.strip().split(' ')
		for i in range(2, 4):
			words = items[i].split('_')
			for word in words:
				if not word in vocab:
					code += 1
					vocab[word] = code
	print('The number of words in the dictionary is {}\n'.format(code))
	return vocab


def encode(slist, vocab):
	slist_encoded = []
	for i in range(0, len(slist)):
		if slist[i] in vocab:
			slist_encoded.append(vocab[slist[i]])
		else:
			slist_encoded.append(vocab['UNKNOWN'])
	return slist_encoded


def loadDataSets(vocab, file_path, sequence_length, flag):
	myFile = ''
	if flag == 'train':
		myFile = file_path + 'data/train'
	elif flag == 'val':
		myFile = file_path + 'data/val'
	else:
		print ('--- Flag of datasets is wrong! ---')
	dataSets = []  # contains dataSets[[question,answer],[],[],...]
	if not os.path.exists(myFile):
		print('--- file doesnot exist ---\n')
	for line in open(myFile):
		items = line.strip().split(' ')
		items[2] = items[2].split('_')[0:(sequence_length)]
		items[3] = items[3].split('_')[0:(sequence_length)]
		items[2] = encode(items[2], vocab)
		items[3] = encode(items[3], vocab)
		dataSets.append([items[2], items[3]])  # [question,answer]
	return dataSets


def loadTrainData(vocab, file_path, sequence_length, size, step):
    input_x1 = []       # questions
    input_x2 = []	    # positive answers
    input_x3 = []	    # negative answers
    trainDataSets = loadDataSets(vocab, file_path, sequence_length, 'train')
    trainData = list()
    num_data_sets = len(trainDataSets)
    for i in range(size):
	    trainData.append(trainDataSets[(step*size+i)%num_data_sets])

    for i in range(size):
        input_x1.append(trainData[i][0])
        input_x2.append(trainData[i][1])
        random_sample = random.sample(trainDataSets,1)	# include question and answer
        while (trainData[i][0] == random_sample[0][0]):
	        random_sample = random.sample(trainDataSets, 1)  # include question and answer
	        print ('Re-random~~~')
        input_x3.append(random_sample[0][1])
    return np.array(input_x1), np.array(input_x2), np.array(input_x3)


def loadValData(vocab, file_path, sequence_length, ratio):
	input_x1 = []  # questions
	input_x2 = []  # positive answers
	input_x3 = []  # negative answers
	flag_list = []  # to indicate the answer in 'negative answer list' is true(1) or false(0)
	valDataSets = loadDataSets(vocab, file_path, sequence_length, 'val')
	random_sample = random.sample(valDataSets, 1)
	for i in range(ratio):
		input_x1.append(random_sample[0][0])
		input_x2.append(random_sample[0][1])
		random_ans = random.sample(valDataSets, 1)  # include question and answer
		if (random_sample[0][0] == random_ans[0][0]):
			flag_list.append(0)     # positive answer
		else:
			flag_list.append(1)
		input_x3.append(random_ans[0][1])
	return np.array(input_x1), np.array(input_x2), np.array(input_x3), np.array(flag_list)


def getSentence(data, vocab):
	de_vocab = deVocab(vocab)
	sentences = []
	for i in range(len(data)):
		temStrList = []
		for j in data[i]:
			tem = de_vocab[j]
			if not tem == '<a>':
				temStrList.append(tem)
		sentence = " ".join(temStrList)
		sentences.append(sentence)
	return sentences


def deVocab(vocab):
	de_vocab = {}
	for i in vocab:
		if not vocab[i] in de_vocab:
			de_vocab[vocab[i]] = i
	return de_vocab


def saveData(sentence):
	f = open("./data/saved_test_data.txt", 'a')
	f.write(sentence)
	f.flush()
	f.close()


if __name__ == '__main__':
	filePath = './'
	vocab = buildVocab(filePath)
	q, ap, an = loadValData(vocab, filePath, 200, 1000)

	question = getSentence(q, vocab)
	ans_pos = getSentence(ap, vocab)
	ans_neg = getSentence(an, vocab)
	print (np.shape(question))
	print ('Question: ' + question[0])
	print ('Ans_pos: ' + ans_pos[0])
	print ('Ans_neg: ' + ans_neg[0])


