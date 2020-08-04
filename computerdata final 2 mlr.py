# import pacakages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.compat import lzip
import pylab 
import scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

#print dataset
compdata = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Multi linear regression\\ComputerData.csv")
compdata.head()
compdata.corr()
colnames = list(compdata.columns)

# convert str into unique int values in dataset
compdata.loc[compdata['cd']=='no','cd']=0
compdata.loc[compdata['cd']=='yes','cd']=1

compdata.loc[compdata['multi']=='no','multi']=0
compdata.loc[compdata['multi']=='yes','multi']=1

compdata.loc[compdata['multi']=='no','multi']=0
compdata.loc[compdata['multi']=='yes','multi']=1

compdata.loc[compdata['premium']=='no','premium']=0
compdata.loc[compdata['premium']=='yes','premium']=1

# check skewnee & kurtosis
# graph plot
compdata['price'].skew()
compdata['price'].kurt()
plt.hist(compdata['price'],edgecolor='k')
sns.distplot(compdata['price'],hist=False)
plt.boxplot(compdata['price'])
# bar graph plot and display count.
# bar plot for screen
sc = compdata['screen'].value_counts()
scr=[sc[14],sc[15],sc[17]]
scn=compdata['screen'].unique()
plt.bar(scn,sc,edgecolor='k')
for i, v in enumerate(scr):  
    plt.text(scn[i], 
              v, 
              scr[i], 
              fontsize=18, 
              color="red")
plt.show()
# bar plot for ram
ra = compdata['ram'].value_counts()
ram=[ra[8],ra[4],ra[16],ra[2],ra[24],ra[32]]
ram1=compdata['ram'].unique()
plt.bar(ram1,ra,edgecolor='k')
for i, v in enumerate(ram):  
    plt.text(ram1[i], 
              v, 
              ram[i], 
              fontsize=10, 
              color="red")
plt.grid()
plt.show()


# Getting coefficients of variables  
co1 = smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=compdata).fit() 
co1.params             # Getting coefficients of variables 
co1.summary()      # Summary
# preparing model based only on Volume
co2_ps=smf.ols('price~speed',data = compdata).fit()  
co2_ps.summary() 
co3_w=smf.ols('price~screen',data = compdata).fit()  
co3_w.summary() 
co4_hd=smf.ols('price~hd',data = compdata).fit()  
co4_hd.summary() 
co5_pr=smf.ols('price~ram',data = compdata).fit()  
co5_pr.summary() 

# Predicted values of profit 
Profit1_pred = co1.predict(compdata[['speed','hd','ram','screen','cd','multi','premium','ads','trend']])
Profit1_pred

# calculating VIF's values of independent variables
rsq_speed = smf.ols('speed~hd+ram+screen',data=compdata).fit().rsquared  
vif_speed = 1/(1-rsq_speed) 
rsq_hd = smf.ols('hd~speed+ram+screen',data=compdata).fit().rsquared  
vif_hd = 1/(1-rsq_hd) 
rsq_ram = smf.ols('ram~hd+speed+screen',data=compdata).fit().rsquared  
vif_ram = 1/(1-rsq_ram) 
rsq_screen = smf.ols('screen~hd+ram+speed',data=compdata).fit().rsquared  
vif_screen = 1/(1-rsq_screen) 

# Storing vif values in a data frame
comp1 = {'Variables':['speed','hd','ram','screen'],'VIF':[vif_speed,vif_hd,vif_ram,vif_screen]}
Vif_frame = pd.DataFrame(comp1)  
Vif_frame

# split into train and test dataset.      
compdata_train,compdata_test  = train_test_split(compdata,test_size = 0.2)
# preparing the model on train data 
model_train = smf.ols("price~speed+hd+ram+screen+cd+multi+premium+ads+trend",data=compdata_train).fit()
train_pred = model_train.predict(compdata_train)  # train_data prediction
train_resid  = train_pred - compdata_train.price    # train residual values 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))   # RMSE value for train data 
test_pred = model_train.predict(compdata_test)      # prediction on test data set 
test_resid  = test_pred - compdata_test.price       # test residual values 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))   # RMSE value for test data 


#  visualization
# Scatter plot between the variables along with histograms
sns.pairplot(compdata)
# Added varible plot 
sm.graphics.influence_plot(co1)
# Observed values VS Fitted values
plt.scatter(compdata.price,Profit1_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
# Residuals VS Fitted Values 
plt.scatter(Profit1_pred,co1.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
# Checking Residuals are normally distributed
st.probplot(co1.resid_pearson, dist="norm", plot=pylab)   

#
# Ridge Regression
# split dataset into x and y 
x = compdata.iloc[:,2:]
y = compdata['price']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# alpha values
alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,0.5, 1]
# alpha value apply
for a in alphas:
 model = Ridge(alpha=a, normalize=True).fit(x,y) 
 score = model.score(x, y)
 pred_y = model.predict(x)
 mse = mean_squared_error(y, pred_y) 
 print("Alpha:{0:.6f}, R2:{1:.3f}, MSE:{2:.2f}, RMSE:{3:.2f}"
    .format(a, score, mse, np.sqrt(mse)))
# X_train,X_test,Y_train & Y_train fitted into dataset
# print ypred,score,mse
ridge_mod=Ridge(alpha=0.01, normalize=True).fit(X_train,Y_train)
ypred = ridge_mod.predict(X_test)
score = model.score(X_test,Y_test)
mse = mean_squared_error(Y_test,ypred)
print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
   .format(score, mse,np.sqrt(mse))) 
# plot scatter plot
x_ax = range(len(X_test))
plt.scatter(x_ax, Y_test, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()











































