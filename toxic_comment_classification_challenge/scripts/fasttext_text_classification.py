import os
import time

from data_helpers import clean_str
from dataset_util import get_train_comment_text_and_label, get_test_id_and_comment_text, label_type_list


def gen_fasttext_train_file(label_type):
    comment_texts, labels = get_train_comment_text_and_label(label_type)
    comment_texts = [clean_str(comment_text) for comment_text in comment_texts]

    train_filename = "train.{}.txt".format(label_type)
    with open(train_filename, 'w') as train_file:
        for i in range(len(labels)):
            line = comment_texts[i] + " " + "__label__{}".format(labels[i]) + "\n"
            train_file.write(line)
    return train_filename


def gen_fasttext_test_file():
    ids, comment_texts = get_test_id_and_comment_text()
    comment_texts = [clean_str(str(comment_text)) for comment_text in comment_texts]

    test_filename = "test.txt"
    with open(test_filename, 'w') as test_file:
        for i in range(len(comment_texts)):
            line = comment_texts[i] + "\n"
            test_file.write(line)
    return test_filename, ids


def gen_fasttext_prob(cls_res):
    label_and_probs = []
    for line in cls_res:
        fields = line.split(" ")
        label = int(fields[0][-1])
        prob = float(fields[1].strip("\n"))

        if label == 0:
            pos_prob = 1 - prob
        else:
            pos_prob = prob
        pos_prob = max(min(pos_prob, 1), 0)
        label_and_probs.append((label, prob, pos_prob))
    return label_and_probs


def fasttext_train_and_predict():
    test_filename, ids = gen_fasttext_test_file()
    all_label_and_probs = []
    for label_type in label_type_list:
        train_filename = gen_fasttext_train_file(label_type)

        cmd = "fasttext supervised -input {} -output model.{} -minCount 10 -wordNgrams 1 -epoch 5 -ws 5".format(train_filename, label_type)
        os.system(cmd)

        cmd = "fasttext predict-prob model.{}.bin {}".format(label_type, test_filename)
        output = os.popen(cmd)

        label_and_probs = gen_fasttext_prob(output.readlines())
        all_label_and_probs.append(label_and_probs)

    # generate the submission file
    pred_filename = "submission-" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(pred_filename, 'w') as pred_file:
        pred_file.write("id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n")
        for i in range(len(ids)):
            pred = str(ids[i])
            for label_and_probs in all_label_and_probs:
                pred += "," + str(label_and_probs[i][2])
            pred += "\n"
            pred_file.write(pred)
    cmd = "zip {}.zip {}".format(pred_filename, pred_filename)
    os.system(cmd)


if __name__ == '__main__':
    fasttext_train_and_predict()
