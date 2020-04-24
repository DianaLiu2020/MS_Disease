import lightgbm as lgb
import pandas as pd

from bayes_opt import BayesianOptimization
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from utilities import age_categorize, recode_region, grid_search_wrapper, draw_ROC_curve


class MS:
    '''
    Descriptive Statistics for Highmark
    '''
    def __init__(self, df, test):
        self.df_ = df.copy() # Keep original one for reporting
        self.df = df.copy()
        self.test = test.copy()

    def data_recoding(self):
        '''
        Recode and categorize variables
        '''
        self.df['MS'] = self.df['分组'].apply(lambda x: 1 if x == '阳性组' else 0)
        self.df['是否住院'] = self.df['是否住院'].apply(lambda x: 1 if x == '是' else 0)
        self.df['年龄'] = self.df['年龄'].apply(lambda x: age_categorize(x))
        self.df['region'] = self.df['籍贯'].apply(lambda x: recode_region(x))
        self.df['region'] = self.df['region'].astype('category')
        return self.df

    def feature_engineering(self):
        '''
        Generate new variables - interaction terms
        '''
        self.df['脑梗死_其他脑血管病'] = self.df.apply(lambda x: 1 if (x['脑梗死'] > 0) & (x['其他脑血管病'] > 0) else 0, axis=1)
        self.df['脑梗死_大脑缺血性发作'] = self.df.apply(lambda x: 1 if (x['脑梗死'] > 0) & (x['大脑缺血性发作'] > 0) else 0, axis=1)
        self.df['脑梗死_动脉粥样硬化'] = self.df.apply(lambda x: 1 if (x['脑梗死'] > 0) & (x['动脉粥样硬化'] > 0) else 0, axis=1)

        return self.df

    def training_prepare(self):
        '''
        Standardize and split training/testing
        '''
        self.y = self.df['MS']
        self.X = self.df.drop(['MS', '行标签', '籍贯', '分组', '地域'], axis=1)

        # split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, stratify=self.y, test_size=0.2, random_state=0)

        self.X_train = pd.get_dummies(self.X_train, columns=['年龄', '性别', 'region', '甘油三酯', '血糖'], drop_first=True)
        self.X_test = pd.get_dummies(self.X_test, columns=['年龄', '性别', 'region', '甘油三酯', '血糖'], drop_first=True)

        normalize_var = ['visit次数']
        scaler = preprocessing.MinMaxScaler().fit(self.X_train[normalize_var])

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(self.X_train)
        principalDf = pd.DataFrame(data = principalComponents, 
                                   columns = ['principal_component1', 'principal_component2'])
        self.X_train = pd.concat([principalDf, self.X_train], axis = 1)

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(self.X_test)
        principalDf = pd.DataFrame(data = principalComponents, 
                                   columns = ['principal_component1', 'principal_component2'])
        self.X_test = pd.concat([principalDf, self.X_test], axis = 1)

        # transform the training data and use them for the model training
        self.X_train[normalize_var] = scaler.transform(self.X_train[normalize_var])
        self.X_test[normalize_var] = scaler.transform(self.X_test[normalize_var])
        return self.X_train, self.X_test, self.y_train, self.y_test

    def LassoReg(self):
        LLGrid = {"C": [3.99, 4, 4.01, 4.02, 4.03]}
        clf = LogisticRegression(penalty='l1', random_state=0)
        grid_search_LR_f1 = grid_search_wrapper(self.X_train, self.y_train, self.X_test, self.y_test, clf, LLGrid, refit_score='precision_score')
        best_lasso_model = grid_search_LR_f1.best_estimator_
        coef = pd.DataFrame(best_lasso_model.coef_.transpose(), index=self.X_train.columns,
                            columns=['Coefficients']).sort_values('Coefficients', ascending=False)
        coef_r = coef[coef.Coefficients != 0]

    def RandomForest(self):
        '''
        Use Top 20 Logistic Lasso selected variables in RF 
        Apply Bayesian Optimization on CV 
        '''
        LassoVars = data.head(20).index
        self.X_train_selected = self.X_train[LassoVars]
        self.X_test_selected = self.X_test[LassoVars]

        def rf_evaluate(n_estimators,
                        max_depth,
                        min_samples_split
                        ):
            params = dict()
            # params['learning_rate'] = 0.1
            params['n_estimators'] = int(max(n_estimators, 0)),
            params['max_depth'] = int(max(max_depth, 1)),
            params['min_samples_split'] = int(max(min_samples_split, 2))

            return cross_val_score(RandomForestClassifier(random_state=0,
                                                          n_estimators=int(max(n_estimators, 0)),
                                                          max_depth=int(max(max_depth, 1)),
                                                          min_samples_split=int(max(min_samples_split, 2)),
                                                          n_jobs=-1
                                                          # class_weight='balanced'
                                                          # verbose_eval=False
                                                          ),
                                   X=self.X_train_selected,
                                   y=self.y_train,
                                   cv=5, scoring='precision').max()

        rf_BO = BayesianOptimization(rf_evaluate,
                                     {'n_estimators': (10, 1000),
                                      'max_depth': (1, 150),
                                      'min_samples_split': (2, 10)
                                      }
                                     )
        rf_BO.maximize(init_points=5, n_iter=20)

        results = []
        values = []
        for i in range(len(rf_BO.res)):
            results.append(rf_BO.res[i]['params'])
            values.append(rf_BO.res[i]['target'])

        rf_BO_scores = pd.DataFrame(results)
        rf_BO_scores['score'] = pd.DataFrame(values)
        rf_BO_scores = rf_BO_scores.sort_values(by='score', ascending=False).head(1)

        self.rf_best = RandomForestClassifier(n_estimators=int(rf_BO_scores['n_estimators']),
                                              max_depth=int(rf_BO_scores['max_depth']),
                                              min_samples_split=int(rf_BO_scores['min_samples_split']),
                                              n_jobs=-1,
                                              random_state=0)
        self.rf_best.fit(self.X_train_selected, self.y_train)

        rf_importance = pd.DataFrame(self.rf_best.feature_importances_, index=self.X_train_selected.columns,
                                     columns=['importance']).sort_values('importance', ascending=False)

        # Testing
        y_pred = self.rf_best.predict(self.X_test_selected)
        
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
        # draw_ROC_curve(y_test, y_pred)

        print('Confusion matrix of Random Forest optimized for Precision Score on the test data:')
        cm = confusion_matrix(y_test, y_pred)
        cmDF = pd.DataFrame(cm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
        print(cmDF)

        print("\t%s: %r" % ("roc_auc_score is: ", roc_auc_score(y_test, y_pred)))
        print("\t%s: %r" % ("f1_score is: ", f1_score(y_test, y_pred)))
        print('recall = ', float(cm[1,1]) / (cm[1,0] + cm[1,1]))
        print('precision = ', float(cm[1,1]) / (cm[1, 1] + cm[0,1]))
        print("Accuracy on MS data: ",accuracy_score(y_test, y_pred))

        return self.rf_best


    def LightGBM(self):
        '''
        Use Top 20 Logistic Lasso selected variables in LightGBM
        Apply Bayesian Optimization on CV 
        '''
        LassoVars = data.head(20).index
        self.X_train_selected = self.X_train[LassoVars]
        self.X_test_selected = self.X_test[LassoVars]
        lgb_train = lgb.Dataset(self.X_train, y_train)

        def lightgbm_evaluate(max_bins,
                             num_leaves,
                             feature_fraction,
                             bagging_fraction,
                             bagging_freq
                        ):
            params = dict()
            params['objective'] = 'binary'
            params['learning_rate'] = 0.1
            params['max_bins'] = int(max_bins)   
            params['num_leaves'] = int(num_leaves)    
            params['feature_fraction'] = feature_fraction
            params['bagging_fraction'] = bagging_fraction
            params['bagging_freq'] = int(bagging_freq)

            cv_results = lgb.cv(params,
                    lgb_train,
                    num_boost_round=100000,
                    nfold=5,
                    early_stopping_rounds=100,
                    metrics='precision',
                    shuffle=False,
                    verbose_eval=False
                   )

            return pd.DataFrame(cv_results)['precision'].max()


        lgb_BO = BayesianOptimization(lgb_evaluate, 
                             {'max_bins': (127, 1023),
                              'num_leaves': (15, 512),
                              'feature_fraction': (0.2, 0.8),
                              'bagging_fraction': (0.7, 1),
                              'bagging_freq': (1, 5)
                             }
                            )

        lgb_BO.maximize(init_points=5, n_iter=20)

        results = []
        values = []
        for i in range(len(lgb_BO.res)):
            results.append(lgb_BO.res[i]['params'])
            values.append(ldb_BO.res[i]['target'])

        lgb_BO_scores = pd.DataFrame(results)
        lgb_BO_scores['score'] = pd.DataFrame(values)
        lgb_BO_scores = lgb_BO_scores.sort_values(by='score', ascending=False).head(1)

        self.lgb_best = lightgbm.LGBMClassifier(max_bins=int(lgb_BO_scores['max_bins']),
                                              num_leaves=int(lgb_BO_scores['num_leaves']),
                                              feature_fraction=int(lgb_BO_scores['feature_fraction']),
                                              bagging_fraction=int(lgb_BO_scores['bagging_fraction']),
                                              bagging_freq=int(rf_lgb_scores['bagging_freq']),
                                              n_jobs=-1,
                                              random_state=0)
        self.lgb_best.fit(self.X_train_selected, self.y_train)

        # Testing
        y_pred = self.lgb_best.predict(self.X_test_selected)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
        # draw_ROC_curve(y_test, y_pred)

        print('Confusion matrix of LightGBM optimized for Precision Score on the test data:')
        cm = confusion_matrix(y_test, y_pred)
        cmDF = pd.DataFrame(cm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
        print(cmDF)

        print("\t%s: %r" % ("roc_auc_score is: ", roc_auc_score(y_test, y_pred)))
        print("\t%s: %r" % ("f1_score is: ", f1_score(y_test, y_pred)))
        print('recall = ', float(cm[1,1]) / (cm[1,0] + cm[1,1]))
        print('precision = ', float(cm[1,1]) / (cm[1, 1] + cm[0,1]))
        print("Accuracy on MS data: ",accuracy_score(y_test, y_pred))

        return self.lgb_best

    def predict(self):
        '''
        Finalize prediction with lightgbm due to better precision score
        '''
        normalize_var = ['visit次数']
        scaler = preprocessing.MinMaxScaler().fit(self.test[normalize_var])
        self.test[normalize_var] = scaler.transform(self.test[normalize_var])
        y_prob = self.lgb_best.predict_proba(self.test)[:, 1]
        y_prob_concat = pd.Series(y_prob1, index=self.test.index)
        new_data_predict = pd.concat([self.test, y_prob_concat], axis=1)
        new_data_predict.rename(columns={0: 'Predicted_Probability'}, inplace=True)
        new_data_predict.to_csv('~disease_prediction.csv', encoding='utf-8-sig',
                                index=False)

    def run(self):
        # Categorizing variables
        self.data_recoding()

        # Generate interaction terms
        self.feature_engineering()

        # Prepare for training in Lasso
        self.training_prepare()

        # Selected variables using Lasso Regression
        self.LassoReg()

        # Use RF for prediction based on variables from Lasso
        self.RandomForest()

        # Use LightGBM for prediction based on variables from Lasso
        self.LightGBM()

        # Prediction with the best model
        self.predict()

