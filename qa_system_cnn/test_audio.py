# -*- coding: UTF-8 -*-

import tensorflow as tf
import data_processor
import os
import random
import numpy as np

# import pyaudio to record sound
import pyaudio
import wave
import sys
import speech_recognition as sr
from os import path
from gtts import gTTS
import os


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS =10
WAVE_QUES_FILENAME = "ques.wav"
WAVE_ANSW_FILENAME = "answ.wav"


if sys.platform == 'darwin':
    CHANNELS = 1


# define a cnn module
embedding_size = 100     # 100
batch_size = 200
sequence_length = 200
filter_sizes=[1,2,3,5]
num_filters = 500
loss_margin = 0.009     # 0.05
learning_rate = 0.01        #0.1
num_epoch = 100 #140
eval_every = 200
ratio = batch_size     # for test  == batch_size
test_size = 100

# ==============================================================================
# reference: LSTM-based deep learning models for non-factoid answer selection
#
# ==============================================================================
# Build vocabulary first
print("------ validation.py~~ Loading data ------")
filePath = '/home/sherrie/PycharmProjects/cnnDemo/'
vocab = data_processor.buildVocab(filePath)
vocab_size = len(vocab)


graph = tf.Graph()
with graph.as_default():
    # define the parameters of cnn
    w_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")  # W is embedding matrixd
    w_conv = list()
    b_conv = list()
    for filter_size in filter_sizes:
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        w_conv.append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1)))
        b_conv.append(tf.Variable(tf.constant(0.1, shape=[num_filters])))

    # Placeholders for varialbes. input_x1,input_x2,input_x3分别为输入的问题，正向答案，负向答案。
    input_x1 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x1")
    input_x2 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x2")
    input_x3 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x3")

    # embedding layer, embed the words into vectors and expand them to 4 dim tensors for cnn architecture.
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        embedded_chars1 = tf.nn.embedding_lookup(w_embedding, input_x1)
        embedded_chars2 = tf.nn.embedding_lookup(w_embedding, input_x2)
        embedded_chars3 = tf.nn.embedding_lookup(w_embedding, input_x3)
        embedded_chars1_expanded = tf.expand_dims(embedded_chars1,
                                                       -1)  # add a dim at the end of the variable. input of image conv has 4 dims.
        embedded_chars2_expanded = tf.expand_dims(embedded_chars2, -1)
        embedded_chars3_expanded = tf.expand_dims(embedded_chars3, -1)
    pooled_outputs1 = []
    pooled_outputs2 = []
    pooled_outputs3 = []
    for i in range((len(filter_sizes))):
        conv = tf.nn.conv2d(embedded_chars1_expanded, w_conv[i], strides=[1, 1, 1, 1],
                            padding="VALID", name="conv")
        h = tf.nn.relu(tf.nn.bias_add(conv, b_conv[i]), name="relu")
        # print ("h", h.get_shape())
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_sizes[i] + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")  # shape of pooled is [batch_size,1,1,num_filters]
        # print ("pooled_outputs"+ str(i), pooled.get_shape())
        pooled_outputs1.append(pooled)
        conv = tf.nn.conv2d(embedded_chars2_expanded,  w_conv[i], strides=[1, 1, 1, 1],
                            padding="VALID", name="conv")
        h = tf.nn.relu(tf.nn.bias_add(conv, b_conv[i]), name="relu")
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_sizes[i] + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")  # shape of pooled is [batch_size,1,1,num_filters]
        pooled_outputs2.append(pooled)

        conv = tf.nn.conv2d(embedded_chars3_expanded,  w_conv[i], strides=[1, 1, 1, 1],
                            padding="VALID", name="conv")
        # print('\n--- shape of cov is {}'.format(conv.get_shape()))
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b_conv[i]), name="relu")
        # Max-pooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_sizes[i] + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")  # shape of pooled is [batch_size,1,1,num_filters]
        pooled_outputs3.append(pooled)

    # reshape the outputs to combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    # print ("pooled_outputs1",len(pooled_outputs1), pooled_outputs1[0].get_shape(),pooled_outputs1[1].get_shape(),pooled_outputs1[2].get_shape(),pooled_outputs1[3].get_shape())
    h_pooled1 = tf.concat(pooled_outputs1, 3)  # the 4th dim corresponds to num_filters
    h_pooled1_flat = tf.reshape(h_pooled1, [-1, num_filters_total])
    # self.h_pooled1_flat = tf.nn.dropout(self.h_pooled1_reshape,1.0)
    # print('\n--- shape of h_pooled1_flat {}'.format(self.h_pooled1_flat.get_shape()))
    h_pooled2 = tf.concat(pooled_outputs2, 3)
    h_pooled2_flat = tf.reshape(h_pooled2, [-1, num_filters_total])
    # self.h_pooled2_flat = tf.nn.dropout(self.h_pooled2_reshape,1.0)
    h_pooled3 = tf.concat(pooled_outputs3, 3)
    h_pooled3_flat = tf.reshape(h_pooled3, [-1, num_filters_total])
    # self.h_pooled3_flat = tf.nn.dropout(self.h_pooled3_reshape,1.0)

    len_pooled1 = tf.sqrt(
        tf.reduce_sum(tf.multiply(h_pooled1_flat, h_pooled1_flat), 1))  # length of quesiton vectors
    # print('\n--- shape of len_pooled1 {}'.format(len_pooled1.get_shape()))
    len_pooled2 = tf.sqrt(
        tf.reduce_sum(tf.multiply(h_pooled2_flat, h_pooled2_flat), 1))  # length of positive answer vectors
    len_pooled3 = tf.sqrt(
        tf.reduce_sum(tf.multiply(h_pooled3_flat, h_pooled3_flat), 1))  # length of negative answer vectors
    mul_12 = tf.reduce_sum(tf.multiply(h_pooled1_flat, h_pooled2_flat),
                           1)  # wisely multiple vectors,　向量的点积
    # print('\n--- shape of mul_12 {}'.format(mul_12.get_shape()))
    mul_13 = tf.reduce_sum(tf.multiply(h_pooled1_flat, h_pooled3_flat), 1)

    # output
    with tf.name_scope("output"):
        cos_12 = tf.div(mul_12, tf.multiply(len_pooled1, len_pooled2),
                             name="scores")  # computes the angle between the two vectors
        cos_13 = tf.div(mul_13, tf.multiply(len_pooled1, len_pooled3))

    zero = tf.constant(0, shape=[batch_size], dtype=tf.float32)
    margin = tf.constant(loss_margin, shape=[batch_size], dtype=tf.float32)

    with tf.name_scope("loss"):
        losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(cos_12, cos_13)))
        # print ("tf.sub(margin, tf.sub(cos_12, cos_13))",(tf.sub(margin, tf.sub(cos_12, cos_13))).get_shape(), losses.get_shape())
        loss = tf.reduce_sum(losses)
        loss = tf.div(loss, batch_size)

    # accuracy
    with tf.name_scope("accuracy"):
        correct = tf.equal(zero, losses)
        accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")


    # 显示/保存测试数据
    def save_test_data(y1, y2, y3, i):
        sen_y1 = data_processor.getSentence(y1, vocab)[0]
        sen_y2 = data_processor.getSentence(y2, vocab)[0]
        sen_y3 = data_processor.getSentence(y3, vocab)
        data_processor.saveData('\nQuestion ' + str(i + 1) + ':\n' + sen_y1)
        data_processor.saveData('\nPositive Answer:\n' + sen_y2)
        data_processor.saveData('\nNegative Answers:')
        for j in range(4):
            data_processor.saveData('\n' + str(j + 1) + ' ' + sen_y3[j])
        return


    def save_train_data(x1, x2, x3):
        sen_x1 = data_processor.getSentence(x1, vocab)
        sen_x2 = data_processor.getSentence(x2, vocab)
        sen_x3 = data_processor.getSentence(x3, vocab)

        for j in range(4):
            data_processor.saveData('\nQuestion ' + str(j + 1) + ':\n' + sen_x1[j])
            data_processor.saveData('\nPositive Answer' + ':\n' + sen_x2[j])
            data_processor.saveData('\nNegative Answer' + ':\n' + sen_x3[j])
        return


    def save_data_losses(x1, x2, x3, losses):
        sen_x1 = data_processor.getSentence(x1, vocab)
        sen_x2 = data_processor.getSentence(x2, vocab)
        sen_x3 = data_processor.getSentence(x3, vocab)
        # print (np.shape(losses),losses)
        num = 0
        for k in range(len(losses)):
            # print ("losses", np.shape(losses), type(losses))
            if (losses[k]!=0.0):
                data_processor.saveData('\nQuestion_wrong ' + str(num + 1) + ':\n' + sen_x1[k])
                data_processor.saveData('\nPositive Answer' + ':\n' + sen_x2[k])
                data_processor.saveData('\nNegative Answer' + ':\n' + sen_x3[k])
                num +=1
                if(num==4):
                    return
        return

    # define a validation/test step
    def test_step(input_y1, input_y2, input_y3, flag_list, sess):
        feed_dict = dict()
        feed_dict = {
            input_x1: input_y1,
            input_x2: input_y2,
            input_x3: input_y3}

        correct_flag = 0
        test_losses_ = sess.run(losses, feed_dict)
        test_losses_ = test_losses_ * flag_list
        test_loss_ = sum(test_losses_)  # add all the losses

        # print('The loss of validation is {}'.format(loss))
        if test_loss_ == 0.0:
            correct_flag = 1
        cos_pos_, cos_neg_, accuracy_ = sess.run([cos_12, cos_13, accuracy], feed_dict)
        data_processor.saveFeatures(cos_pos_, cos_neg_, test_loss_, accuracy_)
        return correct_flag, test_losses_


    def test():
        correct_num = int(0)
        for i in range(test_size):
            batch_y1, batch_y2, batch_y3, flag_list = data_processor.loadValData(vocab, filePath, sequence_length, ratio)      # batch_size*seq_len
            # 显示/保存测试数据
            save_test_data(batch_y1, batch_y2, batch_y3,i)
            correct_flag, test_losses_ = test_step(batch_y1, batch_y2, batch_y3, flag_list, sess)
            save_data_losses(batch_y1, batch_y2, batch_y3, test_losses_)
            # print ('corr_flag', correct_flag)
            correct_num += correct_flag
        print ('correct_num',correct_num)
        acc = correct_num / float(test_size)
        return acc


    f = open(filePath + "data/saved_features.txt", 'w')
    f.close()
    f = open(filePath + "data/saved_test_data.txt", 'w')
    f.close()


    def show(sess, vocab, file_path, sequence_length, ratio):
        batch_y1, batch_y2, batch_y3, flag_list = data_processor.loadValData(vocab, file_path, sequence_length,
                                                                             ratio)  # batch_size*seq_len
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedded_chars1 = tf.nn.embedding_lookup(w_embedding, batch_y1)
            embedded_chars2 = tf.nn.embedding_lookup(w_embedding, batch_y2)
            embedded_chars3 = tf.nn.embedding_lookup(w_embedding, batch_y3)
            embedded_chars1_expanded = tf.expand_dims(embedded_chars1,
                                                      -1)  # add a dim at the end of the variable. input of image conv has 4 dims.
            embedded_chars2_expanded = tf.expand_dims(embedded_chars2, -1)
            embedded_chars3_expanded = tf.expand_dims(embedded_chars3, -1)
        pooled_outputs1 = []
        pooled_outputs2 = []
        pooled_outputs3 = []
        for i in range((len(filter_sizes))):
            conv = tf.nn.conv2d(embedded_chars1_expanded, w_conv[i], strides=[1, 1, 1, 1],
                                padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b_conv[i]), name="relu")
            # print ("h", h.get_shape())
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_sizes[i] + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")  # shape of pooled is [batch_size,1,1,num_filters]
            # print ("pooled_outputs"+ str(i), pooled.get_shape())
            pooled_outputs1.append(pooled)
            conv = tf.nn.conv2d(embedded_chars2_expanded, w_conv[i], strides=[1, 1, 1, 1],
                                padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b_conv[i]), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_sizes[i] + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")  # shape of pooled is [batch_size,1,1,num_filters]
            pooled_outputs2.append(pooled)

            conv = tf.nn.conv2d(embedded_chars3_expanded, w_conv[i], strides=[1, 1, 1, 1],
                                padding="VALID", name="conv")
            # print('\n--- shape of cov is {}'.format(conv.get_shape()))
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b_conv[i]), name="relu")
            # Max-pooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_sizes[i] + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")  # shape of pooled is [batch_size,1,1,num_filters]
            pooled_outputs3.append(pooled)

        # reshape the outputs to combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        # print ("pooled_outputs1",len(pooled_outputs1), pooled_outputs1[0].get_shape(),pooled_outputs1[1].get_shape(),pooled_outputs1[2].get_shape(),pooled_outputs1[3].get_shape())
        h_pooled1 = tf.concat(pooled_outputs1, 3)  # the 4th dim corresponds to num_filters
        h_pooled1_flat = tf.reshape(h_pooled1, [-1, num_filters_total])
        # self.h_pooled1_flat = tf.nn.dropout(self.h_pooled1_reshape,1.0)
        # print('\n--- shape of h_pooled1_flat {}'.format(self.h_pooled1_flat.get_shape()))
        h_pooled2 = tf.concat(pooled_outputs2, 3)
        h_pooled2_flat = tf.reshape(h_pooled2, [-1, num_filters_total])
        # self.h_pooled2_flat = tf.nn.dropout(self.h_pooled2_reshape,1.0)
        h_pooled3 = tf.concat(pooled_outputs3, 3)
        h_pooled3_flat = tf.reshape(h_pooled3, [-1, num_filters_total])
        # self.h_pooled3_flat = tf.nn.dropout(self.h_pooled3_reshape,1.0)

        len_pooled1 = tf.sqrt(
            tf.reduce_sum(tf.multiply(h_pooled1_flat, h_pooled1_flat), 1))  # length of quesiton vectors
        # print('\n--- shape of len_pooled1 {}'.format(len_pooled1.get_shape()))
        len_pooled2 = tf.sqrt(
            tf.reduce_sum(tf.multiply(h_pooled2_flat, h_pooled2_flat), 1))  # length of positive answer vectors
        len_pooled3 = tf.sqrt(
            tf.reduce_sum(tf.multiply(h_pooled3_flat, h_pooled3_flat), 1))  # length of negative answer vectors
        mul_12 = tf.reduce_sum(tf.multiply(h_pooled1_flat, h_pooled2_flat),
                               1)  # wisely multiple vectors,　向量的点积
        # print('\n--- shape of mul_12 {}'.format(mul_12.get_shape()))
        mul_13 = tf.reduce_sum(tf.multiply(h_pooled1_flat, h_pooled3_flat), 1)

        # output

        cos_12 = tf.div(mul_12, tf.multiply(len_pooled1, len_pooled2),
                        name="scores")  # computes the angle between the two vectors
        cos_13 = tf.div(mul_13, tf.multiply(len_pooled1, len_pooled3) )

        max_idx = tf.argmax(cos_13, 0)
        pos = sess.run(cos_12)
        neg = sess.run(cos_13)
        max_idx = sess.run(max_idx)
        # tmp1 = neg[max_idx]
        if (pos[0] > neg[max_idx]):
            print('\033[1;31;34m')
            print "Correct prediction"
            print "Question:", '\033[0m'
            print data_processor.getSentence(batch_y1, vocab)[0],'\033[1;31;34m'
            print "Correct Answer:", '\033[0m'
            print data_processor.getSentence(batch_y2, vocab)[0], '\033[1;31;34m'
            print "Predicted Answer:", '\033[0m'
            print data_processor.getSentence(batch_y2, vocab)[0], '\033[0m'


        else:
            print('\033[1;31;34m')
            print "False prediction"
            print "Question:", '\033[0m'
            print data_processor.getSentence(batch_y1, vocab)[0], '\033[1;31;34m'
            print "Correct Answer:", '\033[0m'
            print data_processor.getSentence(batch_y2, vocab)[0], '\033[1;31;34m'
            print "Predicted Answer:", '\033[0m'
            print data_processor.getSentence(batch_y3, vocab)[max_idx], '\033[0m'


        #
        # if tf.greater(cos_12[0], cos_13[max_idx])
        # 	print "correct"
        # if cos_12 > cos_13[max_idx]:
        # 	print ("The predicted answer is ：/n ", data_processor.getSentence(batch_y1, vocab)[0])
        # else:
        #
        # 	print ("The predicted answer is wrong ：/n ", data_processor.getSentence(batch_y1, vocab)[max_idx])

    def trans(ques):
        ques = ques.split(" ")

        for i in range(sequence_length - len(ques)):
            ques.append("<a>")
        return ques


    def cosine(encoded_ques, encoded_answ):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedded_ques = tf.nn.embedding_lookup(w_embedding, encoded_ques)
            embedded_answ = tf.nn.embedding_lookup(w_embedding, encoded_answ)
            embedded_ques_expanded = tf.expand_dims(embedded_ques, -1)
            embedded_answ_expanded = tf.expand_dims(embedded_answ, -1)

        pooled_ques = []
        pooled_answ = []
        for i in range((len(filter_sizes))):
            conv = tf.nn.conv2d(embedded_ques_expanded, w_conv[i], strides=[1, 1, 1, 1],
                                padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b_conv[i]), name="relu")
            # print ("h", h.get_shape())
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_sizes[i] + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")  # shape of pooled is [batch_size,1,1,num_filters]
            # print ("pooled_outputs"+ str(i), pooled.get_shape())
            pooled_ques.append(pooled)

            conv = tf.nn.conv2d(embedded_answ_expanded, w_conv[i], strides=[1, 1, 1, 1],
                                padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b_conv[i]), name="relu")
            # print ("h", h.get_shape())
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_sizes[i] + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")  # shape of pooled is [batch_size,1,1,num_filters]
            # print ("pooled_outputs"+ str(i), pooled.get_shape())
            pooled_answ.append(pooled)

        # reshape the outputs to combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        # print ("pooled_outputs1",len(pooled_outputs1), pooled_outputs1[0].get_shape(),pooled_outputs1[1].get_shape(),pooled_outputs1[2].get_shape(),pooled_outputs1[3].get_shape())

        h_pooled_ques = tf.concat(pooled_ques, 3)
        h_pooled_ques_flat = tf.reshape(h_pooled_ques, [-1, num_filters_total])
        h_pooled_answ = tf.concat(pooled_answ, 3)
        h_pooled_answ_flat = tf.reshape(h_pooled_answ, [-1, num_filters_total])


        # output
        mul_qa = tf.reduce_sum(tf.multiply(h_pooled_ques_flat, h_pooled_answ_flat), 1)
        len_pooled_answ = tf.sqrt(
            tf.reduce_sum(tf.multiply(h_pooled_answ_flat, h_pooled_answ_flat), 1))  # length of answer vectors
        len_pooled_ques = tf.sqrt(
            tf.reduce_sum(tf.multiply(h_pooled_ques_flat, h_pooled_ques_flat), 1))  # length of quesiton vectors

        cos = tf.div(mul_qa, tf.multiply(len_pooled_answ, len_pooled_ques))
        return cos


    def gTTSfun(text):
        tts = gTTS(text=text, lang='en')
        tts.save(WAVE_ANSW_FILENAME)
        os.system("mpg321 --frames 500 " + WAVE_ANSW_FILENAME)

    def audio_SR():
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        print("* recording")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_QUES_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # obtain audio
        AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), WAVE_QUES_FILENAME)
        r = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILE) as source:
            audio = r.record(source)  # read the entire audio file
        text = r.recognize_google(audio)
        print("Google Speech Recognition thinks you said: " + text)
        return text

    def audio_show(sess, vocab, file_path, sequence_length, ratio):
        ques = audio_SR()
        print('\033[1;31;34m')
        print("Question: " + '\033[0m' + ques)
        ques_list = []
        ques_list.append(trans(ques))
        encoded_ques = []
        encoded_ques.append(data_processor.encode(ques_list[0], vocab))
        encoded_answ = []
        answ = []
        cnt = 0
        for line in open(file_path + 'data/val'):
            items = line.strip().split(' ')
            items[3] = items[3].split('_')[0:(sequence_length)]
            answ.append(items[3])
            items[3] = data_processor.encode(items[3], vocab)
            encoded_answ.append(items[3])
            cnt = cnt + 1
            if (cnt==80):
                break
        # encoded_ques = np.array(encoded_ques)
        # encoded_answ = np.array(encoded_answ)
        # cosine(encoded_ques = encoded_ques, encoded_answ = encoded_answ)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedded_ques = tf.nn.embedding_lookup(w_embedding, encoded_ques)
            embedded_answ = tf.nn.embedding_lookup(w_embedding, encoded_answ)
            embedded_ques_expanded = tf.expand_dims(embedded_ques, -1)
            embedded_answ_expanded = tf.expand_dims(embedded_answ, -1)

        pooled_ques = []
        pooled_answ = []
        for i in range((len(filter_sizes))):
            conv = tf.nn.conv2d(embedded_ques_expanded, w_conv[i], strides=[1, 1, 1, 1],
                                padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b_conv[i]), name="relu")
            # print ("h", h.get_shape())
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_sizes[i] + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")  # shape of pooled is [batch_size,1,1,num_filters]
            # print ("pooled_outputs"+ str(i), pooled.get_shape())
            pooled_ques.append(pooled)

            conv = tf.nn.conv2d(embedded_answ_expanded, w_conv[i], strides=[1, 1, 1, 1],
                                padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b_conv[i]), name="relu")
            # print ("h", h.get_shape())
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_sizes[i] + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")  # shape of pooled is [batch_size,1,1,num_filters]
            # print ("pooled_outputs"+ str(i), pooled.get_shape())
            pooled_answ.append(pooled)

        # reshape the outputs to combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        # print ("pooled_outputs1",len(pooled_outputs1), pooled_outputs1[0].get_shape(),pooled_outputs1[1].get_shape(),pooled_outputs1[2].get_shape(),pooled_outputs1[3].get_shape())

        h_pooled_ques = tf.concat(pooled_ques, 3)
        h_pooled_ques_flat = tf.reshape(h_pooled_ques, [-1, num_filters_total])
        h_pooled_answ = tf.concat(pooled_answ, 3)
        h_pooled_answ_flat = tf.reshape(h_pooled_answ, [-1, num_filters_total])


        # output
        mul_qa = tf.reduce_sum(tf.multiply(h_pooled_ques_flat, h_pooled_answ_flat), 1)
        len_pooled_answ = tf.sqrt(
            tf.reduce_sum(tf.multiply(h_pooled_answ_flat, h_pooled_answ_flat), 1))  # length of answer vectors
        len_pooled_ques = tf.sqrt(
            tf.reduce_sum(tf.multiply(h_pooled_ques_flat, h_pooled_ques_flat), 1))  # length of quesiton vectors

        cos = tf.div(mul_qa, tf.multiply(len_pooled_answ, len_pooled_ques))

        max_idx = sess.run(tf.argmax(cos, 0))
        max_value = np.max(sess.run(cos))
        pre_answ = answ[max_idx]
        for i in range(len(pre_answ)):
            if pre_answ[i]== '<a>':
                break
        pre_answ = pre_answ[0:i]
        print 'Predicting... \n' + '\033[1;31;34m' + 'Confidence Score: '+ '\033[0m' + str(max_value)
        print '\033[1;31;34m' + 'Predicted answer: ' + '\033[0m' + ' '.join(pre_answ)
        gTTSfun(' '.join(pre_answ))


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        model = tf.train.get_checkpoint_state('/home/sherrie/PycharmProjects/cnnDemo/checkpoints_train/')
        if model and model.model_checkpoint_path:
            saver.restore(sess, model.model_checkpoint_path)
        print('\n====== Model restored,  begin to test ')


        cmd = raw_input("\n====== Press \"1\" for audio test, \"2\" for random text test\n> ")
        while (1):
            if cmd.lower() == '1':
                audio_show(sess, vocab, filePath, sequence_length, ratio)
                cmd = raw_input(
                    "\n====== Press \"1\" for audio test, \"2\" for random text test\n> ")
            elif cmd.lower() == '2':
                show(sess, vocab, filePath, sequence_length, ratio)
                cmd = raw_input("\n====== Press \"1\" for audio test, \"2\" for random text test\n> ")
            else:
                cmd = raw_input("\n====== Press \"1\" for audio test, \"2\" for random text test\n> ")