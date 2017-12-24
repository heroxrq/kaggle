import numpy as np
import pandas as pd


label_type_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def get_train_comment_text_and_label(label_type):
    df = pd.read_csv("/media/xrq/Elements/dataset/kaggle/toxic_comment_classification_challenge/train.csv")

    comment_text = df['comment_text']
    # toxic = df['toxic']
    # severe_toxic = df['severe_toxic']
    # obscene = df['obscene']
    # threat = df['threat']
    # insult = df['insult']
    # identity_hate = df['identity_hate']
    #
    # print("==================================================")
    # print("==> summary")
    # print("num of instances: {}".format(len(comment_text)))
    # print("num of toxic: {}".format(sum(toxic)))
    # print("num of severe_toxic: {}".format(sum(severe_toxic)))
    # print("num of obscene: {}".format(sum(obscene)))
    # print("num of threat: {}".format(sum(threat)))
    # print("num of insult: {}".format(sum(insult)))
    # print("num of identity_hate: {}".format(sum(identity_hate)))
    # print("==================================================")

    print("{} ==> pos: {}, neg: {}".format(label_type, sum(df[label_type]), len(comment_text) - sum(df[label_type])))

    return np.array(comment_text), np.array(df[label_type])


def get_test_id_and_comment_text():
    df = pd.read_csv("/media/xrq/Elements/dataset/kaggle/toxic_comment_classification_challenge/test.csv")

    id_ = df['id']
    comment_text = df['comment_text']

    print("num of test instances: {}".format(len(id_)))

    return np.array(id_), np.array(comment_text)
