# """
# No Crap, Only Models!

# You Learn: Cross Validation, Grid Search, Custom Metric in scikit-learn and ENSEMBLING!!!!

# NOTE: I'm still commenting and making this better ;)

# @author: Abhishek Thakur
# @email: abhishek4@gmail.com
# """
# # https://www.kaggle.com/abhishek/no-crap-only-models/code

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import os
import sys
import logging
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

CROSS_VALIDATION = False
GRID_SEARCH = False
ENSEMBLE = True
NUM_FOLDS = 5

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)


class Ensembler(object):
    def __init__(self, model_dict, num_folds=3, task_type='classification', optimize=roc_auc_score,
                 lower_is_better=False, save_path=None):
        """
        Ensembler init function
        :param model_dict: model dictionary, see README for its format
        :param num_folds: the number of folds for ensembling
        :param task_type: classification or regression
        :param optimize: the function to optimize for, e.g. AUC, logloss, etc. Must have two arguments y_test and y_pred
        :param lower_is_better: is lower value of optimization function better or higher
        :param save_path: path to which model pickles will be dumped to along with generated predictions, or None
        """

        self.model_dict = model_dict
        self.levels = len(self.model_dict)
        self.num_folds = num_folds
        self.task_type = task_type
        self.optimize = optimize
        self.lower_is_better = lower_is_better
        self.save_path = save_path

        self.training_data = None
        self.test_data = None
        self.y = None
        self.lbl_enc = None
        self.y_enc = None
        self.train_prediction_dict = None
        self.test_prediction_dict = None
        self.num_classes = None

    def fit(self, training_data, y, lentrain):
        """
        :param training_data: training data in tabular format
        :param y: binary, multi-class or regression
        :return: chain of models to be used in prediction
        """

        self.training_data = training_data
        self.y = y

        if self.task_type == 'classification':
            self.num_classes = len(np.unique(self.y))
            logger.info("Found %d classes", self.num_classes)
            self.lbl_enc = LabelEncoder()
            self.y_enc = self.lbl_enc.fit_transform(self.y)
            kf = StratifiedKFold(n_splits=self.num_folds)
            train_prediction_shape = (lentrain, self.num_classes)
        else:
            self.num_classes = -1
            self.y_enc = self.y
            kf = KFold(n_splits=self.num_folds)
            train_prediction_shape = (lentrain, 1)

        self.train_prediction_dict = {}
        for level in range(self.levels):
            self.train_prediction_dict[level] = np.zeros((train_prediction_shape[0],
                                                          train_prediction_shape[1] * len(self.model_dict[level])))

        for level in range(self.levels):

            if level == 0:
                temp_train = self.training_data
            else:
                temp_train = self.train_prediction_dict[level - 1]

            for model_num, model in enumerate(self.model_dict[level]):
                validation_scores = []
                foldnum = 1
                for train_index, valid_index in kf.split(self.train_prediction_dict[0], self.y_enc):
                    logger.info("Training Level %d Fold # %d. Model # %d", level, foldnum, model_num)

                    if level != 0:
                        l_training_data = temp_train[train_index]
                        l_validation_data = temp_train[valid_index]
                        model.fit(l_training_data, self.y_enc[train_index])
                    else:
                        l0_training_data = temp_train[0][model_num]
                        if type(l0_training_data) == list:
                            l_training_data = [x[train_index] for x in l0_training_data]
                            l_validation_data = [x[valid_index] for x in l0_training_data]
                        else:
                            l_training_data = l0_training_data[train_index]
                            l_validation_data = l0_training_data[valid_index]
                        model.fit(l_training_data, self.y_enc[train_index])

                    logger.info("Predicting Level %d. Fold # %d. Model # %d", level, foldnum, model_num)

                    if self.task_type == 'classification':
                        temp_train_predictions = model.predict_proba(l_validation_data)
                        self.train_prediction_dict[level][valid_index,
                        (model_num * self.num_classes):(model_num * self.num_classes) +
                                                       self.num_classes] = temp_train_predictions

                    else:
                        temp_train_predictions = model.predict(l_validation_data)
                        self.train_prediction_dict[level][valid_index, model_num] = temp_train_predictions
                    validation_score = self.optimize(self.y_enc[valid_index], temp_train_predictions)
                    validation_scores.append(validation_score)
                    logger.info("Level %d. Fold # %d. Model # %d. Validation Score = %f", level, foldnum, model_num,
                                validation_score)
                    foldnum += 1
                avg_score = np.mean(validation_scores)
                std_score = np.std(validation_scores)
                logger.info("Level %d. Model # %d. Mean Score = %f. Std Dev = %f", level, model_num,
                            avg_score, std_score)

            logger.info("Saving predictions for level # %d", level)
            train_predictions_df = pd.DataFrame(self.train_prediction_dict[level])
            train_predictions_df.to_csv(os.path.join(self.save_path, "train_predictions_level_" + str(level) + ".csv"),
                                        index=False, header=None)

        return self.train_prediction_dict

    def predict(self, test_data, lentest):
        self.test_data = test_data
        if self.task_type == 'classification':
            test_prediction_shape = (lentest, self.num_classes)
        else:
            test_prediction_shape = (lentest, 1)

        self.test_prediction_dict = {}
        for level in range(self.levels):
            self.test_prediction_dict[level] = np.zeros((test_prediction_shape[0],
                                                         test_prediction_shape[1] * len(self.model_dict[level])))
        self.test_data = test_data
        for level in range(self.levels):
            if level == 0:
                temp_train = self.training_data
                temp_test = self.test_data
            else:
                temp_train = self.train_prediction_dict[level - 1]
                temp_test = self.test_prediction_dict[level - 1]

            for model_num, model in enumerate(self.model_dict[level]):

                logger.info("Training Fulldata Level %d. Model # %d", level, model_num)
                if level == 0:
                    model.fit(temp_train[0][model_num], self.y_enc)
                else:
                    model.fit(temp_train, self.y_enc)

                logger.info("Predicting Test Level %d. Model # %d", level, model_num)

                if self.task_type == 'classification':
                    if level == 0:
                        temp_test_predictions = model.predict_proba(temp_test[0][model_num])
                    else:
                        temp_test_predictions = model.predict_proba(temp_test)
                    self.test_prediction_dict[level][:, (model_num * self.num_classes): (model_num * self.num_classes) +
                                                                                        self.num_classes] = temp_test_predictions

                else:
                    if level == 0:
                        temp_test_predictions = model.predict(temp_test[0][model_num])
                    else:
                        temp_test_predictions = model.predict(temp_test)
                    self.test_prediction_dict[level][:, model_num] = temp_test_predictions

            test_predictions_df = pd.DataFrame(self.test_prediction_dict[level])
            test_predictions_df.to_csv(os.path.join(self.save_path, "test_predictions_level_" + str(level) + ".csv"),
                                       index=False, header=None)

        return self.test_prediction_dict

def rmsle(h, y):
    """
    Compute the Root Mean Squared Log Error for hypothesis h and targets y

    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

def logged_rmsle(h, y):
    """
    Compute the Root Mean Squared Log Error for hypothesis h and targets y

    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    h = np.expm1(h)
    y = np.expm1(y)
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

def feature_generator(df):
    lbl_enc = LabelEncoder()
    features = pd.DataFrame()
    features['flag'] = lbl_enc.fit_transform(df.store_and_fwd_flag.values)
    features['week_of_year'] = df['pickup_datetime'].dt.weekofyear
    features['month_of_year'] = df['pickup_datetime'].dt.month
    features['hour'] = df['pickup_datetime'].dt.hour
    features['dayofyear'] = df['pickup_datetime'].dt.dayofyear
    features['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    features['vendor_id'] = df.vendor_id.values
    features['passenger_count'] = df.passenger_count.values
    features['pickup_longitude'] = df.pickup_longitude.values
    features['pickup_latitude'] = df.pickup_latitude.values
    features['dropoff_longitude'] = df.dropoff_longitude.values
    features['dropoff_latitude'] = df.dropoff_latitude.values
    return features

if __name__ == '__main__':
    sample_df = pd.read_csv('../input/sample_submission.csv')
    df_train = pd.read_csv('../input/train.csv', parse_dates=[2, 3])
    df_test = pd.read_csv('../input/test.csv', parse_dates=[2, ])

    y = df_train.trip_duration.values
    df_train = df_train.drop(['id', 'trip_duration', 'dropoff_datetime'], axis=1)
    df_test = df_test.drop('id', axis=1)

    lentrain = len(df_train)
    data = pd.concat([df_train, df_test])
    data = data.reset_index(drop=True)

    features = feature_generator(data)
    X = features.values[:lentrain, :]
    X_test = features.values[lentrain:, :]

    y_log = np.log1p(y)

    # cross validation
    if CROSS_VALIDATION:
        errors = []
        clf = xgb.XGBRegressor(max_depth=3, n_estimators=100, learning_rate=0.1, silent=True, nthread=10)
        for i in range(NUM_FOLDS):
            print ("########################################")
            print ("Fold = ", i + 1)
            xtrain, xtest, ytrainlog, ytestlog, ytrain, ytest = train_test_split(X, y_log, y, random_state=i)
            clf.fit(xtrain, ytrainlog)
            preds = clf.predict(xtest)
            error = rmsle(ytest, np.expm1(preds))
            print ("RMSLE Error = ", error)
            errors.append(error)
            print ("########################################")
        print ("Mean Error = ", np.mean(errors))
        print ("Standard Deviation = ", np.std(errors))

        print ("Fitting on full data...")
        clf.fit(X, y_log)
        preds = np.expm1(clf.predict(X_test))

        # Create submission
        sample_df.trip_duration = preds
        sample_df.to_csv('cv_submission.csv', index=False)

    if GRID_SEARCH:
        xgb_model = xgb.XGBRegressor(nthread=10)
        clf = pipeline.Pipeline([('xgb', xgb_model)])
        param_grid = {'xgb__max_depth': [2, 3],
                      'xgb__learning_rate': [0.1, 0.2],
                      'xgb__n_estimators': [10, 50]}

        # RMSLE Scorer
        rmsle_scorer = metrics.make_scorer(logged_rmsle, greater_is_better=False)

        # Initialize Grid Search Model
        model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=rmsle_scorer, verbose=10, n_jobs=1,
                             iid=True, refit=True, cv=3)

        # Fit Grid Search Model
        model.fit(X, y_log)
        print("Best score: %0.3f" % model.best_score_)
        print("Best parameters set:")
        best_parameters = model.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        # Get best model
        best_model = model.best_estimator_

        # Fit model with best parameters optimized for rmsle (see diff between rmsle and logged rmsle functions)
        print ("Fitting model on full data...")
        best_model.fit(X, y_log)
        preds = np.expm1(best_model.predict(X_test))

        # Create submission
        sample_df.trip_duration = preds
        sample_df.to_csv('gs_submission.csv', index=False)

    if ENSEMBLE:
        train_data_dict = {0: [X, X], 1: [X]}
        test_data_dict = {0: [X_test, X_test], 1: [X_test]}

        model_dict = {0: [RandomForestRegressor(n_jobs=-1, n_estimators=10),
                          ExtraTreesRegressor(n_jobs=-1, n_estimators=10)],

                      1: [xgb.XGBRegressor(silent=True, n_estimators=50)]}

        ens = Ensembler(model_dict=model_dict, num_folds=3, task_type='regression',
                        optimize=logged_rmsle, lower_is_better=True, save_path='')

        ens.fit(train_data_dict, y_log, lentrain=X.shape[0])
        preds = ens.predict(test_data_dict, lentest=X_test.shape[0])
        # Create submission
        sample_df.trip_duration = np.expm1(preds[1])
        sample_df.to_csv('ensemble_submission.csv', index=False)