from logging import getLogger
import pandas as pd
import pickle
import warnings
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout 
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

from src.data.hdd_preprocessing import load_preprocess_data, train_test_splitter
from src.features.feature_engineering import hdd_preprocessor, log_transformer

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier

from mlflow.sklearn import save_model

RSEED = 42

warnings.filterwarnings("ignore")
logger = getLogger(__name__)

def __create_ann_model__(input_dim=19):
        # initiate the instance
        model = Sequential()
        # layers
        model.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu', input_dim=input_dim))
        model.add(Dropout(0.2))
        model.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        # compiling the ANN
        model.compile(optimizer = 'adam', 
                    loss = 'binary_crossentropy', 
                    metrics = ['Recall', 'Precision'])
        return model
        
def __get_data():
    logger.info("Loading and preprocessing data")
    X, y = load_preprocess_data(   days=30, filename="ST4000DM000_history_total", 
                                    path=os.getcwd())
    logger.info("Train-test splitting")
    X_train, X_test, y_train, y_test = train_test_splitter(
        X, y, test_size=0.30, random_state=RSEED
        )
    logger.info("Feature engineering on train")
    preprocessor = hdd_preprocessor(days=30, trigger=0.05)
    X_train = preprocessor.fit_transform(X_train)
    logger.info("Feature engineering on test")
    X_test = preprocessor.transform(X_test)
    return X_train, X_test, y_train, y_test


def __compute_and_log_metrics(
    y_true: pd.Series, y_pred: pd.Series, prefix: str = "train"
):
    pass


def run_training():
    logger.info(f"Getting the data")
    X_train, X_test, y_train, y_test = __get_data()

    logger.info("Training")
    # Scaling pipeline
    scaling_pipe = Pipeline([
                ('scaler_log', log_transformer(offset=1)),
                ('scaler_minmax ', MinMaxScaler()),
                ])
    ann_classifier = KerasClassifier(build_fn=__create_ann_model__, epochs=5, batch_size= 40000, 
                            class_weight={0 : 1.0, 1 : len(y_train)/y_train.sum()}, verbose=2)
    ann_classifier._estimator_type = "classifier"
    estimators = [
        ('xgb', XGBClassifier(  disable_default_eval_metric=True,
                            validate_parameters=True,
                            objective="binary:logistic",
                            #eval_metric=xgb_scorer, # Evaluation metric, can use our xgb_scorer
                            scale_pos_weight=len(y_train)/y_train.sum(), # ratio of number of negative class to the positive class
                            colsample_bytree=0.4, # 1, Number of features used by tree, lower to regularize
                            subsample=0.3, # 1, ratio of the training instances used, lower to regularize
                            eta=0.01, # 0.3, learning rate, lower values to regularize
                            gamma=1, # 0, regularization parameter, higher to regularize
                            max_depth=6, # 6, maximum tree depth
                            n_estimators=50 , # 100, number of trees to grow
                            min_child_weight=2 , # 1, minimum sum of instance weight needed in a leaf, higher to regularize
                            reg_lambda=0.7, # 1, L2 regularization
                            reg_alpha=1, # 0, L1 regularization
                            #early_stopping_rounds = 5, #100
                            use_label_encoder=False,
                            )),
        ('nn', ann_classifier),
        ]
    clf = StackingClassifier(estimators = estimators, final_estimator=LogisticRegression(), n_jobs=-1)
    model = Pipeline([
                ('scaling', scaling_pipe),
                ('stacking', clf),
            ])
    logger.info("Fitting in progress")
    model.fit(X_train, y_train)
    logger.info("Pickle")
    filename = 'deployment.bin'
    #with open(filename, 'wb') as file_out:
    #    pickle.dump(model, file_out)
    # saving the model
    logger.info("Saving model in the model folder")
    path = "models/linear"
    save_model(sk_model=model, path=path)


if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    run_training()