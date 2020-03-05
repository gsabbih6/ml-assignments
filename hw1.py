# %%
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# this code is very trivial and trains a knn, svm and mlp model with sklearn to tell if a word is a certain language

# most important part of the code the preprosessing of the words as with every ML problem ie feature extraction

# functions are descriptive and so code should be easy to understand


def read_five_words_from_file(file_name):
    ls = []
    f = open(file_name, 'r+')
    lines = f.read().splitlines()
    f.close()

    for x in lines:
        if len(x) == 5:
            ls.append(x)

    return ls


def add_label_to_list(ls_data, language):
    new_list = []
    if language == 'english':
        for x in ls_data:
            new_list.append([x, 0])

    if language == 'german':
        for x in ls_data:
            new_list.append([x, 1])

    if language == 'italian':
        for x in ls_data:
            new_list.append([x, 2])

    return new_list


#  not used
def preprocessor_hash(list_of_five_letter_words):
    new_list = []

    for x in list_of_five_letter_words:
        new_list.append(hash(x))

    return new_list


def preprocessor_ord(list_of_five_letter_words):
    new_list = []

    for x in list_of_five_letter_words:
        word_list = []
        for char in x:
            word_list.append(ord(char))

        new_list.append(word_list)

    return new_list


def split_dataset(ls_data):  # spilt data into 80/20
    ls_size = len(ls_data)
    e_h = int(0.8 * ls_size)
    first_half = ls_data[0:e_h]
    second_half = ls_data[e_h:ls_size]

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for x in first_half:
        train_x.append(x[0])
        train_y.append(x[1])
    for t in second_half:
        test_x.append(t[0])
        test_y.append(t[1])

    return train_x, train_y, test_x, test_y


def randomise_data(ls_data, repeats=1):
    for i in range(repeats):
        shuffle(ls_data)

    return ls_data


def array_to_test(lst):
    str_ = ''
    for s in lst:
        str_ += chr(s)

    return str_


def to_language(y_code):
    stream = {
        0: 'english',
        1: 'german',
        2: 'italian'
    }

    return stream[y_code]


# %%

#  get words
eng_word_list = read_five_words_from_file('english.txt')
eng_word_list_processed = preprocessor_ord(eng_word_list)
eng_word_list_with_label = add_label_to_list(eng_word_list_processed, 'english')

# print(len(eng_word_list))
# display(eng_word_list_processed[1])

ger_word_list = read_five_words_from_file('german.txt')
ger_word_list_processed = preprocessor_ord(ger_word_list)
ger_word_list_with_label = add_label_to_list(ger_word_list_processed, 'german')

# print(len(ger_word_list))

ita_word_list = read_five_words_from_file('italian.txt')
ita_word_list_processed = preprocessor_ord(ita_word_list)
ita_word_list_with_label = add_label_to_list(ita_word_list_processed, 'italian')

# print(len(ita_word_list))

dataset_complete = ita_word_list_with_label + eng_word_list_with_label + ger_word_list_with_label
# print(len(dataset_complete))

random_dataset = randomise_data(dataset_complete, 6)

train_x, train_y, test_x, test_y = split_dataset(random_dataset)

# display(train_x[10])
# display(train_y)

knn_model = KNeighborsClassifier(n_neighbors=2)
svm_model = svm.SVC()
mlp_nn = MLPClassifier()

knn_model.fit(train_x, train_y)
svm_model.fit(train_x, train_y)
mlp_nn.fit(train_x, train_y)

predict_y_knn = knn_model.predict(test_x)
predict_y_svm = svm_model.predict(test_x)
predict_y_mlp = mlp_nn.predict(test_x)

# %%


# Label text for each graph
labels = ("KNN", "SVM", "MLP")
value = [
    accuracy_score(test_y, predict_y_knn, normalize=True) * 100,
    accuracy_score(test_y, predict_y_svm, normalize=True) * 100,
    accuracy_score(test_y, predict_y_mlp, normalize=True) * 100
]

print(value)

# Title of the plot
plt.title("Model Accuracy")

# Label for the x values of the bar graph
plt.xlabel("Accuracy")

# Drawing the bar graph
y_pos = np.arange(len(labels))
plt.barh(y_pos, value, align="center", alpha=0.5)
plt.yticks(y_pos, labels)

# Display the graph
plt.savefig('accuracy_graph')
plt.show()

