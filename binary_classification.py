import warnings
warnings.filterwarnings("ignore")
import pandas as pd
#from mlxtend.classifier import StackingCVClassifier
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import keras as ks
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression



class westphalia_classification:
    def __init__(self,d_name,target_name):
        self.d_name=d_name
        self.target_name=target_name
        
    def get_data(self):
        self.data=pd.read_csv(self.d_name)
    def clean_data(self):
        #check if the number of NAN value for any feature is more than 30 percent, delete it
        n_nan=self.data.isnull().sum()/float(len(self.data))
        cols=(self.data.drop(columns=[self.target_name],axis=1)).columns
        for col in cols:
            if n_nan.loc[col]>=30:
                self.data.drop(columns=[col],axis=1,inplace=True)
        self.data.dropna(inplace=True)
    def normalize_feature(self):
        self.data.duration=np.log1p(self.data.duration)#maybe not right place
        
    def object_to_numbers_label_encoding(self):
        cols=self.data.columns
        for col in cols:
            if self.data[col].dtypes=='object':
                self.data[col]=LabelEncoder().fit_transform(self.data[col])
    def object_to_numbers_get_dummies(self):
        self.data[self.target_name]=LabelEncoder().fit_transform(self.data[self.target_name])
        cols=(self.data.drop(columns=[self.target_name],axis=1)).columns
        for col in cols:
            if self.data[col].dtypes=='object':
                dummy=pd.get_dummies(self.data[col])
                self.data.drop(columns=[col],axis=1,inplace=True)
                self.data=pd.concat([self.data,dummy],axis=1)
    def scale_data(self):
        y=self.data[self.target_name]
        self.data.drop(columns=[self.target_name],axis=1,inplace=True)
        
        x = self.data.as_matrix().astype(np.float)
        self.y = y.values#as_matrix().astype(np.float)
        
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.scalarx = scaler.fit(x)
        self.rescaledX=self.scalarx.transform(x)
        
        self.x_test_balanced=self.scalarx.transform(self.x_test_balanced)
        
    def get_test_and_train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.rescaledX, self.y, test_size=0,random_state=42)
        
    def engineering(self):
        self.data.duration=np.log1p(self.data.duration)
        #self.data.balance=self.data.balance-np.min(self.data.balance)
        #self.data.balance=np.log1p(self.data.balance)
        cols=['marital','education' ,'loan','housing','pdays','previous','job','poutcome','contact']
        agg_func=['mean','min','max']
        for col in cols:
            db_balance=self.data.groupby([col])['balance'].agg(agg_func)
            self.data=self.data.merge(db_balance, on=col)
            db_duration=self.data.groupby([col])['duration'].agg(agg_func)
            self.data=self.data.merge(db_duration, on=col)
            self.data.drop(columns=[col],axis=1,inplace=True)
        

    def balance_data(self):
        n_lower=len(self.data[self.target_name][self.data[self.target_name]==1])
        index_more=self.data.y[self.data[self.target_name]==0].index
        index_less=self.data.y[self.data[self.target_name]==1].index
        index_more_sample=random.sample(index_more,n_lower)
        index_balance=list(index_less.values)+list(index_more_sample)
        self.data=self.data.loc[index_balance]

    def get_balanced_test(self):
        ind_tot=self.data.index
        index_0=self.data.y[self.data[self.target_name]==0].index
        index_1=self.data.y[self.data[self.target_name]==1].index
        ind_0_s=random.sample(index_0,400)
        ind_1_s=random.sample(index_1,400)
        ind_01_s=list(ind_0_s)+list(ind_1_s)
        ind_rest=[item for item in ind_tot if item not in ind_01_s]
        data_test_b=self.data.loc[ind_01_s]
        self.data=self.data.loc[ind_rest]
        ###self.data=self.data.reset_index()
        
        self.y_test_balancd=data_test_b[self.target_name]
        data_test_b.drop(columns=[self.target_name],axis=1,inplace=True)
        self.y_test_balanced = self.y_test_balancd.as_matrix().astype(np.float)
        self.x_test_balanced= data_test_b.as_matrix().astype(np.float)
    def oversampling(self):
        sm = SMOTE(random_state=42, ratio = 1.0)
        self.X_train, self.y_train = sm.fit_sample(self.X_train, self.y_train)

    def stack_model_rf(self):
        clf = RandomForestClassifier(n_estimators=2000)
        clf.fit(self.X_train, self.y_train)
        ypr=clf.predict_proba(self.x_test_balanced)[:,1]
        ypr_train=clf.predict_proba(self.X_train)[:,1]

        ypr_train=ypr_train.reshape(-1,1)
        ypr=ypr.reshape(-1,1)
        return ypr_train,ypr
    
    def test_xgboos(self):
        xgbmodel = xgb.XGBClassifier(max_depth=30, n_estimators=2000, learning_rate=0.02,random_state=42)
        xgbmodel.fit(self.X_train,self.y_train)
        self.ypr=xgbmodel.predict(self.x_test_balanced)
        self.ypr_train=xgbmodel.predict(self.X_train)
        
    def stack_model_nn(self):
        model=ks.models.Sequential()
        
        model.add(ks.layers.Dense(16, input_dim=self.X_train.shape[1], activation='relu'))
        model.add(ks.layers.Dense(16,activation='relu'))
        model.add(ks.layers.Dense(16,activation='relu'))
        model.add(ks.layers.Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(self.X_train, self.y_train, validation_data=(self.x_test_balanced,self.y_test_balanced),epochs=160,verbose=0)
        
        ypr=model.predict(self.x_test_balanced)
        ypr_train=model.predict(self.X_train)
        ypr_train=ypr_train.reshape(-1,1)
        ypr=ypr.reshape(-1,1)
        return ypr_train,ypr

    def stack_model_lgb(self):
        lightgbm_params = {
            'n_estimators':200,
            'learning_rate':0.01,
            'objective': 'binary',
            'num_leaves':123,
            'colsample_bytree':0.8,
            'subsample':0.9,
            'max_depth':20,
            'reg_alpha':0.1,
            'reg_lambda':0.1,
            'min_split_gain':0.01,
            'min_child_weight':2
        }
        
        lgb_train = lgb.Dataset(self.X_train, self.y_train.ravel())
        lgb_eval = lgb.Dataset(self.x_test_balanced,self.y_test_balanced.ravel(), reference=lgb_train)

        self.lgbm_model = lgb.train(lightgbm_params,lgb_train,valid_sets=lgb_eval,num_boost_round=30)
        
        ypr=self.lgbm_model.predict(self.x_test_balanced)
        ypr_train=self.lgbm_model.predict(self.X_train)
        ypr_train=ypr_train.reshape(-1,1)
        ypr=ypr.reshape(-1,1)
        return ypr_train,ypr
    def stack_model_rgb(self):
        xgb_params = {
            'colsample_bytree': 0.7,
            'silent': 1,
            'subsample': 0.7,
            'learning_rate': 0.01,
            'objective': 'binary:logistic',
            'max_depth': 4,
            'num_parallel_tree': 1,
            'min_child_weight': 1,
            'nrounds': 200
        }
        
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        gbdt = xgb.train(xgb_params, dtrain, 250)
        
        ypr=gbdt.predict(xgb.DMatrix(self.x_test_balanced))
        ypr_train=gbdt.predict(xgb.DMatrix(self.X_train))
        
        ypr_train=ypr_train.reshape(-1,1)
        ypr=ypr.reshape(-1,1)
        return ypr_train,ypr
    def evaluate_prediction(self,ypr_train,ypr,model_name):
        ypr[ypr<0.5]=0
        ypr[ypr>=0.5]=1
        ypr_train[ypr_train<0.5]=0
        ypr_train[ypr_train>=0.5]=1

        bools=ypr.ravel()==self.y_test_balanced.ravel()
        print model_name+' accuracy is: {}'.format(len(bools[bools==True])/float(len(bools)))
        print model_name+' Recall score is: {}'.format(recall_score(self.y_test_balanced, ypr))
    def stacking_manual(self):

        ypr_train_xgb,ypr_xgb=self.stack_model_rgb()
        self.evaluate_prediction(ypr_train_xgb,ypr_xgb,"Xgboost")

        ypr_train_lgb,ypr_lgb=self.stack_model_lgb()
        self.evaluate_prediction(ypr_train_lgb,ypr_lgb,"Lightgbm")
        

        ypr_train_rf,ypr_rf=self.stack_model_rf()
        self.evaluate_prediction(ypr_train_rf,ypr_rf,"Random forest")
        
        
        ypr_train_nn,ypr_nn=self.stack_model_nn()
        self.evaluate_prediction(ypr_train_nn,ypr_nn,"Neural network")

        x_train = np.concatenate((ypr_train_xgb, ypr_train_lgb,ypr_train_rf,ypr_train_nn), axis=1)
        x_test = np.concatenate((ypr_xgb, ypr_lgb,ypr_rf,ypr_nn), axis=1)

        logistic_regression = LogisticRegression()
        logistic_regression.fit(x_train,self.y_train)
        

        self.ypr=logistic_regression.predict(x_test)
        self.ypr_train=logistic_regression.predict(x_train)
        self.evaluate_prediction(self.ypr_train,self.ypr,"Stacked second level")
    def downsampling_method(self):
        self.balance_data()
        self.scale_data()
        self.get_test_and_train()
        self.stacking_manual()
        
if __name__ == '__main__':
    
    data='./bank-full.csv'
    target='y'
    w_classifier=westphalia_classification(data,target)
    w_classifier.get_data()
    w_classifier.clean_data()
    w_classifier.normalize_feature()
    w_classifier.object_to_numbers_get_dummies()
    w_classifier.get_balanced_test()
    w_classifier.downsampling_method()

    
    
