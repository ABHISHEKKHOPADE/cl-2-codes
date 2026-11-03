"""Python code exported from BI_Assignment_6_Agricultural_Genomics_Crop_Improvement.ipynb."""

# %% Cell 2
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# %% Cell 3
rng=np.random.default_rng(42)
n_plants,n_markers=500,100
X=rng.integers(0,3,size=(n_plants,n_markers))
markers=[f"Marker_{i+1:03d}" for i in range(n_markers)]
effects=np.zeros(n_markers); effects[[5,18,37,64,82]]=[1.8,-1.4,1.2,1.6,-1.0]
yield_value=50+X@effects+rng.normal(0,2.5,n_plants)
df=pd.DataFrame(X,columns=markers); df["Yield"]=yield_value
display(df.head()); print("Shape:",df.shape)

# %% Cell 4
print("Missing values:",df.isna().sum().sum())
display(df.Yield.describe().to_frame("Yield"))
plt.figure(figsize=(9,4)); plt.hist(df.Yield,bins=25,edgecolor="black")
plt.title("Crop Yield Distribution"); plt.xlabel("Yield"); plt.ylabel("Frequency"); plt.tight_layout(); plt.show()

# %% Cell 5
# Marker-Trait Association Screening
corr=df[markers].corrwith(df.Yield)
assoc=pd.DataFrame({"Marker":corr.index,"Correlation":corr.values,"Absolute_Correlation":np.abs(corr.values)}).sort_values("Absolute_Correlation",ascending=False)
display(assoc.head(15))
top=assoc.head(15).sort_values("Correlation")
plt.figure(figsize=(9,6)); plt.barh(top.Marker,top.Correlation)
plt.title("Top Marker-Trait Associations"); plt.xlabel("Correlation with Yield"); plt.tight_layout(); plt.show()

# %% Cell 6
# Random Forest Model Traning and Prediction.
X_train,X_test,y_train,y_test=train_test_split(df[markers],df.Yield,test_size=.2,random_state=42)
model=RandomForestRegressor(n_estimators=300,max_features="sqrt",random_state=42)
model.fit(X_train,y_train); pred=model.predict(X_test)
mae=mean_absolute_error(y_test,pred); rmse=np.sqrt(mean_squared_error(y_test,pred)); r2=r2_score(y_test,pred)
print(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")

# %% Cell 7
importance=pd.Series(model.feature_importances_,index=markers).sort_values(ascending=False)
display(importance.head(15).to_frame("Importance"))
importance.head(15).sort_values().plot(kind="barh",figsize=(9,6))
plt.title("Top Markers for Yield Prediction"); plt.xlabel("Random Forest Importance"); plt.tight_layout(); plt.show()

# %% Cell 8
plt.figure(figsize=(7,6)); plt.scatter(y_test,pred,alpha=.7)
lo=min(y_test.min(),pred.min()); hi=max(y_test.max(),pred.max())
plt.plot([lo,hi],[lo,hi],"--")
plt.title("Observed vs Predicted Crop Yield"); plt.xlabel("Observed Yield"); plt.ylabel("Predicted Yield"); plt.tight_layout(); plt.show()
