# imporrt packages
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
# print dataset
toyo = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Multi linear regression\\ToyotaCorolla1.csv",encoding='latin1')
print(toyo.head())
toyota1 = toyo.iloc[:,[2,3,6,8,12,13,15,16,17]]
# rename column name
toyota1 = toyota1.rename(columns={"Age_08_04":"Age","Quarterly_Tax":"QuarterlyTax"})

startup.corr()
# Getting coefficients of variables 
to1 = smf.ols('Price~Age+KM+HP+cc+Doors+Gears+QuarterlyTax+Weight',data=toyota1).fit() 
to1.params
to1.summary()
to2_v=smf.ols('Price~Age',data = toyota1).fit()  
to2_v.summary() 
to3_w=smf.ols('Price~KM',data = toyota1).fit()  
to3_w.summary() 
to4_wv=smf.ols('Price~HP',data = toyota1).fit()  
to4_wv.summary()
to5_pq=smf.ols('Price~QuarterlyTax',data = toyota1).fit()  
to5_wv.summary()
to6_pw=smf.ols('Price~Weight',data = toyota1).fit()  
to6_wv.summary()
toyo2 = smf.ols('Price~Age+KM+HP+cc+Doors+Gears+QuarterlyTax+Weight',data=toyota2).fit()    
toyo2.params
toyo2.summary() 
print(toyo2.conf_int(0.01))
final_t= smf.ols('Price~Age+KM+HP+cc+Doors+Gears+QuarterlyTax+Weight',data = toyota2).fit()
final_t.params
final_t.summary() 
# drop row
toyota2=toyota1.drop(toyota1.index[[80,221,960,601]],axis=0)
toyota2.head()
# Predicted values of profit 
Profit1_pred = toyo2.predict(toyota2[['Age','KM','HP','cc','Doors','Gears','QuarterlyTax','Weight']])
Profit1_pred
Profit2_pred = final_t.predict(toyota2)
# calculating VIF's values of independent variables
rsq_Age = smf.ols('Age~KM+HP+cc+Weight',data=toyota2).fit().rsquared  
vif_Age = 1/(1-rsq_Age) 
rsq_QuarterlyTax = smf.ols('QuarterlyTax~Age+HP+KM+Weight',data=toyota2).fit().rsquared  
vif_QuarterlyTax = 1/(1-rsq_QuarterlyTax)
rsq_Weight = smf.ols('Weight~KM+HP+Doors+Gears',data=toyota2).fit().rsquared  
vif_Weight = 1/(1-rsq_Weight)  
rsq_KM = smf.ols('KM~Age+HP+Gears+Weight',data=toyota2).fit().rsquared  
vif_KM = 1/(1-rsq_KM) 
t1 = {'Variables':['Age','QuarterlyTax','Weight','KM'],'VIF':[vif_Age,vif_QuarterlyTax,vif_Weight,vif_KM]}
Vif_frame = pd.DataFrame(t1)  
Vif_frame
# calculating VIF's values of independent variables
model_train = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+QuarterlyTax+Weight",data=startup_train).fit()
train_pred = model_train.predict(startup_train)
train_resid  = train_pred - startup_train.Price
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
test_pred = model_train.predict(startup_test)
test_resid  = test_pred - startup_test.Price
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
#  visualization
sm.graphics.influence_plot(to1)
sm.graphics.plot_partregress_grid(toyo2)
sm.graphics.plot_partregress_grid(final_t)
plt.scatter(toyota2.Price,Profit1_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
plt.scatter(Profit2_pred,final_t.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
plt.hist(final_t.resid_pearson) 
st.probplot(final_t.resid_pearson, dist="norm", plot=pylab)
startup_train,startup_test  = train_test_split(toyota2,test_size = 0.2)









