import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import datetime
import score


class data_instance:

    def read_data(self, file_name):
        df = pd.read_csv(file_name)
        return df

    def join_data(self, stances, bodies):
        return pd.merge(stances, bodies, how='left', left_on=['Body ID'], right_on=['Body ID'])

    def __init__(self, file_stances, file_bodies):
        self.stances_df = self.read_data(file_stances)
        self.bodies_df = self.read_data(file_bodies)
        self.all_data_df = self.join_data(self.stances_df, self.bodies_df)


def train_cosine(tfidf_vectorizer, headlines, bodies, stances):
    cosine_list = []
    for i in range(len(headlines)):
        head_vec = tfidf_vectorizer.transform([headlines[i]]).toarray()
        body_vec = tfidf_vectorizer.transform([bodies[i]]).toarray()
        # Reshape your data either using array.reshape(-1, 1) if data has a single feature
        # or array.reshape(1, -1) if it contains a single sample.
        cosine = cosine_similarity(head_vec.reshape(1, -1), body_vec.reshape(1, -1))
        cosine = cosine[0][0]
        cosine_list.append(cosine)
    related_cosine = []
    unrelated_cosine = []
    for i in range(len(stances)):
        if stances[i].strip().lower() == 'unrelated':
            unrelated_cosine.append(cosine_list[i])
        else:
            related_cosine.append(cosine_list[i])
    related_cosine = np.array(related_cosine).mean()
    unrelated_cosine = np.array(unrelated_cosine).mean()
    # print(related_cosine, unrelated_cosine)
    return (unrelated_cosine+related_cosine)/2


def test_cosine(tfidf_vectorizer, headlines, bodies, cosine_threshold):
    cosine_list = []
    for i in range(len(headlines)):
        head_vec = tfidf_vectorizer.transform([headlines[i]]).toarray()
        body_vec = tfidf_vectorizer.transform([bodies[i]]).toarray()
        # Reshape your data either using array.reshape(-1, 1) if data has a single feature
        # or array.reshape(1, -1) if it contains a single sample.
        cosine = cosine_similarity(head_vec.reshape(1, -1), body_vec.reshape(1, -1))
        cosine = cosine[0][0]
        cosine_list.append(cosine)
    predicted_stances = []
    for i in range(len(cosine_list)):
        if cosine_list[i] <= cosine_threshold:
            predicted_stances.append('unrelated')
        else:
            predicted_stances.append(None)
    return predicted_stances


def string_to_int_labels(stances):
    label_ref = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
    result = [label_ref[x] for x in stances]
    return result


def get_related_df(bodies):
    return bodies.loc[bodies['Stance'] != 'unrelated']


def get_model(train_df):
    bow_transformer = CountVectorizer(stop_words='english', max_features=5000).fit(train_df['articleBody'])
    train_bow = bow_transformer.transform(train_df['articleBody'])
    tfidf_transformer = TfidfTransformer().fit(train_bow)
    train_tfidf = tfidf_transformer.transform(train_bow)

    # scaler = StandardScaler().fit(train_tfidf.toarray())
    # train_scaled = scaler.transform(train_tfidf.toarray())

    C = [pow(10, a) for a in [x for x in range(-5, 3, 1)]]
    C = [1]
    gamma_rbf = [pow(10, a) for a in [x for x in range(-15, 2, 2)]]
    param_svm = [
        {'C': C, 'kernel': ['linear']},  # Linear parameters
        # {'classifier__C': C,  # RBF parameters
        #  'classifier__gamma': gamma_rbf,
        #  'classifier__kernel': ['rbf']},
        # {'kernel': ['rbf']},  # Linear parameters
    ]
    grid_svm = GridSearchCV(
        SVC(),  # clf from strategy instance
        param_grid=param_svm,  # parameters to tune via cross validation
        refit=True,  # fit using all data, on the best detected classifier
        n_jobs=-1,  # number of cores to use for parallelisation; -1 for "all cores"
        scoring='accuracy',  # what score are we optimizing?, for accuracy it has to exactly match
        cv=StratifiedKFold(3),  # what type of cross validation to use, prevents overfitting
    )
    svm_detector = grid_svm.fit(train_tfidf, train_df['Stance'])
    print(f'Best parameter: {svm_detector.best_params_}')
    print(f'Best score: {svm_detector.best_score_}')
    return svm_detector.best_estimator_, bow_transformer, tfidf_transformer, None


def run_model_on_test(test_data, bow_transformer, tfidf_transformer, scaler, clf):
    X_test = test_data['articleBody']
    # Y_target = test_data['Stance'].as_matrix()
    X_bow = bow_transformer.transform(X_test)
    X_tfidf = tfidf_transformer.transform(X_bow)
    # X_scaled = scaler.transform(X_tfidf.toarray())
    prediction = clf.predict(X_tfidf)
    # print(f'predictions: {string_to_int_labels(prediction)}')
    return prediction


def add_related_to_test(test_data, related_predictions):
    # print(test_data)
    stances = test_data['Stance']
    for i in range(len(stances)):
        if not stances[i]:
            test_data.at[i, 'Stance'] = related_predictions.pop(0)
    return test_data


def get_accuracy(test_data):
    predicted_stances = test_data['Stance'].as_matrix()
    actual_data = data_instance('./competition_test_stances.csv', './test_bodies.csv')
    actual_stances = actual_data.all_data_df['Stance'].as_matrix()
    print(f'Accuracy: {np.mean(predicted_stances==actual_stances)}')
    print(f'actual stances: {string_to_int_labels(actual_stances)}')
    print(f'predicted stances: {string_to_int_labels(predicted_stances)}')
    LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
    score.report_score([LABELS[e] for e in string_to_int_labels(actual_stances)],
                       [LABELS[e] for e in string_to_int_labels(predicted_stances)])


if __name__ == '__main__':
    start = datetime.datetime.now()
    # Get training data
    train_data = data_instance('./train_stances.csv', './train_bodies.csv')
    train_headlines = train_data.all_data_df['Headline']
    train_bodies = train_data.all_data_df['articleBody']
    train_stances = train_data.all_data_df['Stance']
    # Get testing data
    test_data = data_instance('./competition_test_stances_unlabeled.csv', './test_bodies.csv')
    test_headlines = test_data.all_data_df['Headline']
    test_bodies = test_data.all_data_df['articleBody']
    # test_stances = test_data.all_data_df['Stance']
    # create bag of words transformer & apply transformer to headlines and bodies
    # bow_transformer = CountVectorizer(stop_words='english', max_features=5000)
    # training_bow = bow_transformer.fit_transform(pd.concat([train_headlines, train_bodies]))
    # create term frequency transformer and tfidf bow for training and test data
    # training_termfreq = TfidfTransformer(use_idf=False).fit_transform(training_bow)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000).fit(pd.concat([train_headlines,
                                                                                               train_bodies,
                                                                                               test_headlines,
                                                                                               test_bodies]))
    cosine_threshold = train_cosine(tfidf_vectorizer, train_headlines, train_bodies, train_stances)
    unrelated_predicted_stances = test_cosine(tfidf_vectorizer, test_headlines, test_bodies, cosine_threshold)

    # test_labels = string_to_int_labels(test_stances)
    # predicted_labels = string_to_int_labels(predicted_stances)
    # print(test_labels)
    # print(predicted_labels)
    # test_labels = np.array(test_labels)
    # predicted_labels = np.array(predicted_labels)
    # print(np.mean(predicted_labels == test_labels))
    # get_related_df(train_bodies.to_frame(), predicted_stances)
    # get only related data of both train and test sets
    related_train_bodies = train_data.all_data_df[['articleBody', 'Stance']].copy()
    related_train_bodies = get_related_df(related_train_bodies)
    # print(related_train_bodies)
    # related_test_bodies = test_bodies.to_frame()
    # related_test_bodies['Stance'] = unrelated_predicted_stances
    test_data.all_data_df['Stance'] = unrelated_predicted_stances
    related_test_bodies = get_related_df(test_data.all_data_df)
    # print(test_data.all_data_df)
    svm_clf, bow_transformer, tfidf_transformer, scaler = get_model(related_train_bodies)
    related_predictions = run_model_on_test(related_test_bodies, bow_transformer, tfidf_transformer, scaler, svm_clf)
    test_data.all_data_df = add_related_to_test(test_data.all_data_df, related_predictions.tolist())
    # predicted_stances = test_data.all_data_df['Stance'].tomatrix()
    get_accuracy(test_data.all_data_df)
    print(f'Time spent: {datetime.datetime.now()-start}')


