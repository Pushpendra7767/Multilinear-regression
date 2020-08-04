# import packages
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
# import dataset
startup = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Multi linear regression\\50Startups.csv")
startup.head(40)
startup.corr()
# rename column
startup.columns
startup = startup.rename(columns={"R&D":"RnD","Marketing Spend":"Marketing"})
# Getting coefficients of variables                   
st1 = smf.ols('Profit~RnD+Administration+Marketing',data=startup).fit() 
st1.params
st1.summary()
st_v=smf.ols('Profit~RnD',data = startup).fit()  
st_v.summary() 
st_w=smf.ols('Profit~Administration',data = startup).fit()  
st_w.summary() 
st_wv=smf.ols('Profit~Marketing',data = startup).fit()  
st_wv.summary()
final_st= smf.ols('Profit~RnD+Administration+Marketing',data = startup_new).fit()
final_st.params
final_st.summary() 
st_new = smf.ols('Profit~RnD+Administration+Marketing',data=startup_new).fit()    
st_new.params
st_new.summary() 
print(st_new.conf_int(0.01))
# drop row
startup_new=startup.drop(startup.index[[45,49,48]],axis=0)
startup_new.head()
# Predicted values of profit  
Profit2_pred = final_st.predict(startup_new)
Profit1_pred = st_new.predict(startup_new[['RnD','Administration','Marketing']])
Profit1_pred
# calculating VIF's values of independent variables
rsq_RnD = smf.ols('RnD~Administration+Marketing',data=startup_new).fit().rsquared  
vif_RnD = 1/(1-rsq_RnD) 
rsq_Administration = smf.ols('Administration~RnD+Marketing',data=startup_new).fit().rsquared  
vif_Administration = 1/(1-rsq_Administration) 
rsq_Marketing = smf.ols('Marketing~RnD+Administration',data=startup_new).fit().rsquared  
vif_Marketing = 1/(1-rsq_Marketing)          
s1 = {'Variables':['RnD','Administration','Marketing'],'VIF':[vif_RnD,vif_Administration,vif_Marketing]}
Vif_frame = pd.DataFrame(s1)  
Vif_frame

# split dataset into ttrain & test
startup_train,startup_test  = train_test_split(startup_new,test_size = 0.2)

# preparing the model on train data 
model_train = smf.ols("Profit~RnD+Administration+Marketing",data=startup_train).fit()
train_pred = model_train.predict(startup_train)
train_resid  = train_pred - startup_train.Profit
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
test_pred = model_train.predict(startup_test)
test_resid  = test_pred - startup_test.Profit
test_rmse = np.sqrt(np.mean(test_resid*test_resid))

#  visualization
sns.pairplot(startup)
sm.graphics.influence_plot(st1)
sm.graphics.plot_partregress_grid(st_new)
sm.graphics.plot_partregress_grid(final_st)
plt.scatter(startup_new.Profit,Profit1_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
plt.scatter(Profit2_pred,final_st.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
plt.hist(final_st.resid_pearson) 
st.probplot(final_st.resid_pearson, dist="norm", plot=pylab)
plt.scatter(Profit2_pred,final_st.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")







































































