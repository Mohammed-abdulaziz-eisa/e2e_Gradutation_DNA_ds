# Demystify DNA Sequencing with Machine Learning and Python
# https://www.theaidream.com/post/demystify-dna-sequencing-with-machine-learning-and-python


import sys
import time
import os
os.system("cls")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from imblearn.over_sampling import SMOTE
# from pydna import PyDNA    

import warnings                                  
warnings.filterwarnings('ignore')

def get_program_running(start_time):
    end_time = time.time()
    diff_time = end_time - start_time
    result = time.strftime("%H:%M:%S", time.gmtime(diff_time))
    print("program runtime: {}".format(result))

def k_mer_words_original(dna_sequence_string, k_mer_length=6):
    k_mer_list = [dna_sequence_string[x:x + k_mer_length].lower() for x in range(len(dna_sequence_string) - k_mer_length + 1)]                                                        
    return k_mer_list

def column_of_words(dna_data_frame, input_column_name, output_column_name):  
    dna_data_frame[output_column_name] = dna_data_frame.apply(lambda x: k_mer_words_original(x[input_column_name]), axis=1)
    dna_data_frame = dna_data_frame.drop(input_column_name, axis=1)
    return dna_data_frame

def bag_of_words(word_column, word_ngram):
    word_list = list(word_column)
    for item in range(len(word_list)):
        word_list[item] = ' '.join(word_list[item])    
    count_vectorizer = CountVectorizer(ngram_range=(word_ngram, word_ngram))
    X = count_vectorizer.fit_transform(word_list)
    return X

def generate_k_mers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]

# prepared_pata_initial_100_rows.py
def main():
    """load oxnrous csv file
    """
    print("load oxnrous pkl file")    

    pkl_file_path = r"../data/processed/final_processed_data_target_60.pkl"
    df = pd.read_pickle(pkl_file_path)
    # df.info()
    # print(df)

    df = df[["Parent_full_DNA_Seq", "Child_full_DNA_Seq", "target"]]
    # df.info()
    print("df initial")
    print(df.shape)

    #print("df drop nan")
    #df = df.dropna(axis=0)
    #print(df.shape)

    #print("df drop duplicates by Parent_full_DNA_Seq and Child_full_DNA_Seq columns")
    #df = df.drop_duplicates(subset=["Parent_full_DNA_Seq", "Child_full_DNA_Seq", "target"], keep="first", ignore_index=True)
    #print("df")
    #df.info()
    #print(df.shape)
    

    X = df.drop("target", axis=1)
    print("X")
    print(X.shape)
    print(X)

    y = df["target"]
    print("y")
    print(y.shape)
    print(y)

    # --- my code
    # print("generate column of words...")
    # df = column_of_words(df, "Parent_full_DNA_Seq", "words")
    # df.info()
    # print(df.shape)

    # print("generate bag  of words...")
    # X = bag_of_words(df["words"], 4)  
    # X.info()
    # print(X.shape)
    # ---

    # --- oxnrous code
    print("generate k-mers...")
    k = 6
    df['parent_kmers'] = df['Parent_full_DNA_Seq'].apply(lambda x: ' '.join(generate_k_mers(x, k)))
    df['child_kmers'] = df['Child_full_DNA_Seq'].apply(lambda x: ' '.join(generate_k_mers(x, k)))

    print("vectorizer... 1")
    vectorizer1 = CountVectorizer()
    X_parent = vectorizer1.fit_transform (df['parent_kmers']).toarray()
    # X_parent_df = pd.DataFrame.sparse.from_spmatrix(X_parent)
    print("X_parent")
    print(X_parent.shape)
    print(X_parent)

    # X_child_df = pd.DataFrame.sparse.from_spmatrix(X_child)
    print("vectorizer ... 2")
    vectorizer2 = CountVectorizer()
    X_child = vectorizer2.fit_transform (df['child_kmers']).toarray()
    print("X_child")
    print(X_child.shape)
    print(X_child)

    X_parent = pd.DataFrame(X_parent)
    X_child = pd.DataFrame(X_child)
   
    print("concat X_parent and X_child...")
    X = pd.concat([X_parent, X_child], axis=1)
    print("X before SMOTE")
    print(X.shape)
    print(X)

    y = df['target']
    print("y")
    print(y.shape)

    print("df_final")
    df_final = pd.concat([X, y], axis=1)
    df_final.info()
    print(df_final.shape)
    # fingerprint_output_file_final = os.path.join(project_folder_path, "csv\hcv_ns5b_substructure_final.csv")
    #df_final.to_csv( r"G:\Visual WWW\Python\1000_python_workspace\genetic_diagnostics_deep_learning\0Xnrous_Eisa\csv\prepared_pata_initial_100_rows_final.csv", index=False)
   
    smote_over_sampling = SMOTE(random_state=50, n_jobs=-1) 
    X = np.array(X)   
    X, y = smote_over_sampling.fit_resample(X, y)
    print("X after SMOTE")
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    model_classifier = RandomForestClassifier(n_jobs=-1, random_state=50)

    model_classifier.fit(X_train, y_train)

    y_predicted = model_classifier.predict(X_test) 

    # calculate classification accuracy score
    accuracy_score_value = accuracy_score(y_test, y_predicted) * 100
    accuracy_score_value = float("{0:0.2f}".format(accuracy_score_value))    
    print("classification accuracy score:")
    print(accuracy_score_value)
    print()

    # calculate classification confusion matrix
    confusion_matrix_result = confusion_matrix(y_test, y_predicted)
    print("classification confusion matrix:")
    print(confusion_matrix_result)
    print()

    # calculate classification report
    classification_report_result = classification_report(y_test,y_predicted)
    print("classification report:")    
    print(classification_report_result)
    print()  

    print("end...")

if __name__ == '__main__':
    start_time = time.time()    
    main()
    get_program_running(start_time)
    
    
# ---------------------------------------------  
# ---------------------------------------------   
# ---------------------------------------------  
# --------------------------------------------- 
# test - 1 
# ---------------------------------------------   
#     df_final
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 29988 entries, 0 to 29987
# Columns: 129 entries, 0 to target
# dtypes: int64(129)
# memory usage: 29.5 MB
# --------------------------------------------- 
# (29988, 129)
# X after SMOTE
# (30118, 128)
# --------------------------------------------- 
# classification accuracy score:
# 57.2
# --------------------------------------------- 
# k-mer = 3 
# classification confusion matrix:
# [[1808 1220]
#  [1358 1638]]
# --------------------------------------------- 
# classification report:
#               precision    recall  f1-score   support

#            0       0.57      0.60      0.58      3028
#            1       0.57      0.55      0.56      2996

#     accuracy                           0.57      6024
#    macro avg       0.57      0.57      0.57      6024
# weighted avg       0.57      0.57      0.57      6024
# --------------------------------------------- 
# end...
# program runtime: 00:45:04
# ---------------------------------------------  
# --------------------------------------------- 
# ---------------------------------------------  
# --------------------------------------------- 