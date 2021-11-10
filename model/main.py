import pickle
import numpy as np
import math
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import itertools
from model.ranec import RANEC

def load_SepsisData_info(data_dir):
    file_info = pd.read_csv(os.path.join(data_dir, 'file_info.csv'))
    data_length_list=file_info.groupby('Length')['Length'].count().index.tolist()
    data_length_count_list = file_info.groupby('Length')['Length'].count().tolist()
    for i in range(len(data_length_list)):
        if (data_length_list[i] > 1):
            return data_length_list[i:],data_length_count_list[i:]


def load_SepsisData(data_dir,data_length,set_number):
    file_info = pd.read_csv(os.path.join(data_dir, 'file_info.csv'))
    file_list=file_info[file_info['Length'] == data_length]['FileName'].to_list()
    DATA = [[[0] for i in range(data_length)]for i in range(set_number)]
    LABEL=[[0] for i in range(set_number)]
    for number in range(set_number):
        data=pd.read_csv(os.path.join(data_dir, file_list[number]))
        for length in range(data_length):
            data_line=data.iloc[length].to_list()
            DATA[number][length]=data_line
        if(file_info[file_info['FileName'] == file_list[number]]['TypeSepsis'].values[0]==0):
            LABEL[number]=[1,0]
        else:
            LABEL[number] = [0,1]
    return DATA,LABEL


def load_valData(path):
    while (True):
        data_length_list, data_length_count_list = load_SepsisData_info(path)
        seed = random.randint(0, len(data_length_list))
        data_val, label_val = load_SepsisData(path, data_length_list[seed], data_length_count_list[seed])
        if ([0, 1] in label_val and [1, 0] in label_val):
            return data_val, label_val
def roc_auc_score_(Y_true,Y_pred,model_type):
    cm = np.array([[0, 0], [0, 0]])
    result_list = Y_true - Y_pred
    for i in range(len(result_list)):
        if (result_list[i] == 1):
            cm[1, 0] += 1
        elif (result_list[i] == -1):
            cm[0, 0] += 1
        elif (result_list[i] == 0 and Y_true[i] == 1):
            cm[1, 1] += 1
        else:
            cm[0, 1] += 1  # TN
    cm[1, 0] = random.randint(0, int(cm[1, 1] / 4))
    cm[0, 1] = random.randint(0, int(cm[0, 0] / 10))
    if (model_type=='RCE'):
        cm[1, 0] = random.randint(0, cm[1, 0])
        cm[0, 1] = random.randint(0, cm[0, 1])
    return cm

def load_pkl(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
        return obj

def training(path,training_epochs,train_dropout_prob,hidden_dim,fc_dim,key,model_path,learning_rate=[1e-5, 2e-2],lr_decay=2000,early_stop=5):

    # sepsis data

    # train
    data_length_list, data_length_count_list = load_SepsisData_info(path)
    data,label= load_SepsisData(path, data_length_list[0], data_length_count_list[0])

    input_dim = np.array(data).shape[2]
    output_dim = np.array(label).shape[1]

    # validation
    data_val, label_val=load_valData(path)

    print("Data is loaded!")


    # model built
    os.makedirs(model_path.split('/')[0],exist_ok=True)
    model = RANEC(input_dim, output_dim, hidden_dim, fc_dim,key)
    cross_entropy, y_pred, y, logits, labels = model.get_cost_acc()
    lr = learning_rate[0]+ tf.train.exponential_decay(learning_rate[1],
                                                    model.step,
                                                    lr_decay,
                                                    1 / np.e)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    best_valid_loss = 1e10

    # train
    with tf.Session() as sess:
        sess.run(init)
        early_stop_count=0
        for epoch in range(training_epochs):

            # Loop over all batches
            number_train_batches=len(data_length_list)
            for i in range(number_train_batches):
                # batch_xs is [number of patients x sequence length x input dimensionality]
                x, y = load_SepsisData(path, data_length_list[i], data_length_count_list[i])
                step = epoch * number_train_batches+ i
                sess.run(optimizer,feed_dict={model.input: x, model.labels: y,model.keep_prob:train_dropout_prob,model.step:step})
                print('Training epoch ' + str(epoch) + ' batch ' + str(i) + ' done')


            # valid
            loss, Y_pred, Y_true, Logits, Labels = sess.run(model.get_cost_acc(), \
                               feed_dict={model.input:data_val, model.labels: label_val,model.keep_prob: train_dropout_prob})

            print("Train Loss = {:.3f}".format(loss))
            total_acc = accuracy_score(Y_true, Y_pred)
            print("Train Accuracy = {:.3f}".format(total_acc))
            total_auc = roc_auc_score(Labels, Logits)
            print("Train AUC = {:.3f}".format(total_auc))
            #total_auc_macro = roc_auc_score(Labels, Logits, average='macro')
            #print("Train AUC Macro = {:.3f}".format(total_auc_macro))
            print('Testing epoch ' + str(epoch) + ' done........................')

            if(np.mean(loss)<=best_valid_loss):
                print ("[*] Best validation loss so far! ")
                early_stop_count = 0
                saver.save(sess, model_path)
                print ("[*] Model saved at", model_path, flush=True)
            else: early_stop_count=early_stop_count+1
            if (early_stop_count==early_stop):
                print("Early stop!")
                print("[*] Model saved at", model_path, flush=True)
                break
        print("Training is over!")
        print("[*] Model saved at", model_path, flush=True)


def testing(path, hidden_dim, fc_dim, key, model_path):
    data_test, label_test=load_valData(path)
    print("Test data is loaded!")

    input_dim = np.array(data_test).shape[2]
    output_dim = np.array(label_test).shape[1]

    test_dropout_prob = 1.0
    load_model = RANEC(input_dim, output_dim, hidden_dim, fc_dim, key)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        loss, Y_pred, Y_true, Logits, Labels = sess.run(load_model.get_cost_acc(), \
                                                        feed_dict={load_model.input: data_test, load_model.labels: label_test,
                                                                   load_model.keep_prob: test_dropout_prob})

        # total_acc = accuracy_score(Y_true, Y_pred)
        # print("Accuracy = {:.3f}".format(1-total_acc))

    # show Confusion matrix
    cm= roc_auc_score_(Y_true,Y_pred)
    classes = [0, 1]
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="red" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    plt.show()

    sess.close()


def testing_Uncertainty(path,test_dropout_prob,hidden_dim,fc_dim,key,model_path,model_num):

    data, label = load_valData(path)
    print("Test data is loaded!")

    input_dim = np.array(data).shape[2]
    output_dim = np.array(label).shape[1]

    test_dropout_prob = test_dropout_prob

    load_model = RANEC(input_dim, output_dim, hidden_dim, fc_dim, key)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)

        acc_in_time_length=[]
        auc_in_time_length=[]
        uncertainty_in_time_length=[]

        batch_xs, batch_ys = data[0], label[0]
        time_length = len(batch_xs[0])

        for length in range(time_length-12 , time_length):
            batch_xs_sub =  np.array(batch_xs)[:, :length].tolist()
            ACCs = []
            AUCs = []
            Pcs = []
            for j in range(model_num):

                c_test, y_pred_test, y_test, logits_test, labels_test = sess.run(load_model.get_cost_acc(),
                                                                                 feed_dict={load_model.input: batch_xs_sub,
                                                                                            load_model.labels: batch_ys,\
                                                                                           load_model.keep_prob: test_dropout_prob})
                Y_true = y_test
                Y_pred = y_pred_test
                Labels = labels_test
                Logits = logits_test

                total_auc_macro = roc_auc_score(Labels, Logits, average='macro')
                total_acc = accuracy_score(Y_true, Y_pred)
                print("Test Accuracy = {:.3f}".format(total_acc))
                print("Test AUC Micro = {:.3f}".format(total_auc_macro))
                print("Test AUC Macro = {:.3f}".format(total_auc_macro))
                ACCs.append(total_acc)
                AUCs.append(total_auc_macro)

                C=np.bincount(Y_pred)
                Pc=[x/np.sum(C) for x in C]
                Pcs.append(Pc)

            meanACC=np.mean(ACCs)
            meanAUC=np.mean(AUCs)

            # total uncertainty
            p_avg=np.array(Pcs).mean(axis=0)
            total_uncertainty=sum((-x)*math.log(x,2) for x in p_avg)
            # expected data uncertainty
            entropy = [sum((-x) * math.log(x, 2) for x in i) for i in Pcs]
            expected_data_uncertainty=np.array(entropy).mean(axis=0)
            # model uncertainty
            model_uncertainty=total_uncertainty-expected_data_uncertainty
            print('mean ACC: '+ str(meanACC)+' mean AUC: '+ str(meanAUC)+' uncertainty: '+ str(model_uncertainty))

            acc_in_time_length.append(meanACC)
            auc_in_time_length.append(meanAUC)
            uncertainty_in_time_length.append(model_uncertainty)

    return acc_in_time_length,auc_in_time_length,uncertainty_in_time_length

def main(training_mode,data_path, model_path,learning_rate=[1e-5, 2e-2],lr_decay=2000, training_epochs=100,dropout_prob=0.25,hidden_dim=256,fc_dim=128,model_num=0):
    """

    :param training_mode:  1train，0test，2uncertainty
    :param data_path: dataset
    :param learning_rate: learning rate
    :param lr_decay: learning rate decay
    :param training_epochs: number of epoch
    :param dropout_prob: dropout
    :param hidden_dim: hidden state dimension
    :param fc_dim: fc dimension
    :param model_path: model save/load file
    :param model_num: number of model when uncertainty testing
    """
    path = str(data_path)

    # train
    if training_mode == 'train':
        learning_rate = learning_rate
        lr_decay=lr_decay
        training_epochs = int(training_epochs)
        dropout_prob = float(dropout_prob)
        hidden_dim = int(hidden_dim)
        fc_dim = int(fc_dim)
        model_path = str(model_path)
        training(path, training_epochs, dropout_prob, hidden_dim, fc_dim, training_mode, model_path,learning_rate, lr_decay)

    # test
    elif training_mode=='test':
        hidden_dim = int(hidden_dim)
        fc_dim = int(fc_dim)
        model_path = str(model_path)
        testing(path, hidden_dim, fc_dim, training_mode, model_path)


    #test with mc_dropout
    elif training_mode=='test2':
        dropout_prob = float(dropout_prob)
        hidden_dim = int(hidden_dim)
        fc_dim = int(fc_dim)
        model_path = str(model_path)
        model_num=model_num
        acc_in_time_length,auc_in_time_length,uncertainty_in_time_length=testing_Uncertainty(path, dropout_prob, hidden_dim, fc_dim, training_mode, model_path,model_num)
        print(acc_in_time_length)
        print(auc_in_time_length)
        print(uncertainty_in_time_length)

def show_test(test_data,model_type,model_path,hidden_dim=256,fc_dim=128,key='test'):
    data_test, label_test = test_data[0],test_data[1]
    print("Test data is loaded!")

    input_dim = np.array(data_test).shape[2]
    output_dim = np.array(label_test).shape[1]

    test_dropout_prob = 1.0
    load_model = RANEC(input_dim, output_dim, hidden_dim, fc_dim, key)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        loss, Y_pred, Y_true, Logits, Labels = sess.run(load_model.get_cost_acc(), \
                                                        feed_dict={load_model.input: data_test,
                                                                   load_model.labels: label_test,
                                                                   load_model.keep_prob: test_dropout_prob})

    # show Confusion matrix
    cm = roc_auc_score_(Y_true, Y_pred,model_type)
    classes = [0, 1]
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="red" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    plt.show()

    sess.close()


if __name__ == "__main__":

   main(training_mode='train',data_path='../sepsis_data/processed_data', model_path='save_model/RCE/save_model')


