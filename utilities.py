from sklearn.metrics import f1_score, make_scorer, recall_score, \
    precision_score
from sklearn.model_selection import GridSearchCV


def age_categorize(x):
    if x <= 20:
        return '10-20'
    elif x <= 30:
        return '20-30'
    elif x <= 40:
        return '30-40'
    elif x <= 50:
        return '40-50'
    elif x <= 60:
        return '50-60'
    elif x <= 70:
        return '60-70'
    elif x <= 80:
        return '70-80'
    else:
        return '80-90'


region1 = ['山西', '北京', '内蒙', '河北', '天津']
region2 = ['上海', '山东', '聊城', '江西', '福建', '安徽', '江苏', '浙江']
region3 = ['辽宁', '吉林', '沈阳', '黑龙', '本溪']
region4 = ['湖南', '河南', '湖北']
region5 = ['广东', '广西']
region6 = ['四川', '重庆', '云南', '贵州']
region7 = ['陕西', '甘肃', '宁夏', '新疆', '青海']


def recode_region(x):
    if x in region1:
        return "华北"
    elif x in region2:
        return '华东'
    elif x in region3:
        return '东北'
    elif x in region4:
        return '华中'
    elif x in region5:
        return '华南'
    elif x in region6:
        return '西南'
    elif x in region7:
        return '西北'
    else:
        return '无籍贯'


scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'f1_score': make_scorer(f1_score, pos_label=1)
}


def grid_search_wrapper(X_train, y_train, X_test, y_test, model, parameters, refit_score='precision_score'):
    '''
    fits a GridSearchCV classifier using refit_score for optimization(refit on the best model according to refit_score)
    prints classifier performance metrics
    '''
    grid_search = GridSearchCV(model, parameters, scoring=scorers, refit=refit_score,
                           cv=5, return_train_score=True, n_jobs = -1)
    grid_search.fit(X_train, y_train)

    # make the predictions
    y_pred = grid_search.predict(X_test)
    y_prob = grid_search.predict_proba(X_test)[:, 1]
    
    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('Confusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    cm = confusion_matrix(y_test, y_pred)
    cmDF = pd.DataFrame(cm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
    print(cmDF)
    
    print("\t%s: %r" % ("roc_auc_score is: ", roc_auc_score(y_test, y_pred)))
    print("\t%s: %r" % ("f1_score is: ", f1_score(y_test, y_pred)))#string to int

    print('recall = ', float(cm[1,1]) / (cm[1,0] + cm[1,1]))
    print('precision = ', float(cm[1,1]) / (cm[1, 1] + cm[0,1]))
    print("Accuracy on MS data: ",accuracy_score(y_test, y_pred))
    
    draw_ROC_curve(y_test,y_pred)

    return grid_search


def draw_ROC_curve(y_test,y_pred):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test, y_pred)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,color='orange',label='AUC = %0.2f'% roc_auc_score(y_test, y_pred))
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1], color='darkblue', linestyle='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


