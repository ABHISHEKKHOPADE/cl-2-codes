"""Python code exported from BI_Assignment_5_Machine_Learning_for_Genomic_Data.ipynb."""

# %% Cell 1
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,ConfusionMatrixDisplay

# %% Cell 2
rng=np.random.default_rng(42)
n_samples,n_markers=600,80
X=rng.integers(0,3,size=(n_samples,n_markers))
signal=1.2*X[:,3]+1.0*X[:,11]+0.9*X[:,27]-1.1*X[:,42]+0.8*X[:,58]+rng.normal(0,1.2,n_samples)
y=(signal>np.median(signal)).astype(int)
markers=[f"SNP_{i+1:03d}" for i in range(n_markers)]
df=pd.DataFrame(X,columns=markers); df["Class"]=y
display(df.head()); print("Shape:",df.shape); display(df.Class.value_counts().to_frame("count"))

# %% Cell 3
print("Missing values:",df.isna().sum().sum())
display(df[markers].describe().T.head(10))
plt.figure(figsize=(9,4)); df.Class.value_counts().sort_index().plot(kind="bar")
plt.title("Class Distribution"); plt.xlabel("Class"); plt.ylabel("Samples"); plt.xticks(rotation=0); plt.tight_layout(); plt.show()

# %% Cell 4
X_train,X_test,y_train,y_test=train_test_split(df[markers],df.Class,test_size=.2,random_state=42,stratify=df.Class)
print("Training:",X_train.shape,"Testing:",X_test.shape)

# %% Cell 5
# SVM Model Training
svm=Pipeline([("scaler",StandardScaler()),("model",SVC(kernel="rbf",random_state=42))])
svm.fit(X_train,y_train); svm_pred=svm.predict(X_test)
print(classification_report(y_test,svm_pred,digits=4))

# %% Cell 6
# Random Forest Model Training
rf=RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced")
rf.fit(X_train,y_train); rf_pred=rf.predict(X_test)
print(classification_report(y_test,rf_pred,digits=4))

# %% Cell 7
# Comparison of Models
def metric_row(y,p):
    return [accuracy_score(y,p),precision_score(y,p,zero_division=0),recall_score(y,p,zero_division=0),f1_score(y,p,zero_division=0)]
comparison=pd.DataFrame([metric_row(y_test,svm_pred),metric_row(y_test,rf_pred)],
 index=["SVM","Random Forest"],columns=["Accuracy","Precision","Recall","F1"])
display(comparison.round(4))
comparison.plot(kind="bar",figsize=(10,5)); plt.ylim(0,1.05); plt.title("Genomic Classification Model Comparison"); plt.xticks(rotation=0); plt.tight_layout(); plt.show()

# %% Cell 8
fig,ax=plt.subplots(1,2,figsize=(10,4))
ConfusionMatrixDisplay.from_predictions(y_test,svm_pred,ax=ax[0]); ax[0].set_title("SVM")
ConfusionMatrixDisplay.from_predictions(y_test,rf_pred,ax=ax[1]); ax[1].set_title("Random Forest")
plt.tight_layout(); plt.show()

# %% Cell 9
importance=pd.Series(rf.feature_importances_,index=markers).sort_values(ascending=False)
display(importance.head(15).to_frame("Importance"))
importance.head(15).sort_values().plot(kind="barh",figsize=(9,5))
plt.title("Top Genomic Markers — Random Forest"); plt.xlabel("Feature Importance"); plt.tight_layout(); plt.show()
