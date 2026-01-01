import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,
classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import layers,models
RANDOM_STATE = 42
df = pd.read_csv("train.csv")
print("shape:", df.shape)
df.head()
print("\nColumns, dtype, missing, nunique:")
display(pd.DataFrame({
 "dtype": df.dtypes.astype(str),
 "n_missing": df.isna().sum(),
 "n_unique": df.nunique()
}))
if "id" in df.columns:
 df = df.drop(columns=["id"])
 print("Dropped 'id' column.")
data = df.copy()
bin_map = {"Yes": 1, "No": 0, "yes": 1, "no": 0, "YES": 1, "NO": 0}
for c in ["Stage_fear", "Drained_after_socializing"]:
 if c in data.columns:
 data[c] = data[c].astype(str).str.strip().replace({"nan": np.nan})
 data[c] = data[c].map(bin_map).astype(pd.Float64Dtype())
numeric_candidates = [
 "Time_spent_Alone","Social_event_attendance","Going_outside",
 "Friends_circle_size","Post_frequency"
]
for c in numeric_candidates:
 if c in data.columns:
 data[c] = pd.to_numeric(data[c], errors="coerce")
TARGET = "Personality"
if TARGET not in data.columns:
 raise ValueError(f"Target column '{TARGET}' not present. Columns:
{data.columns.tolist()}")
for c in ["Stage_fear", "Drained_after_socializing"]:
 if c in data.columns:
 print(f"\nValue counts for {c}:")
 display(data[c].value_counts(dropna=False))
le_target = LabelEncoder()
data["_y"] = le_target.fit_transform(data[TARGET].astype(str))
print("Target mapping:", dict(zip(le_target.classes_,
le_target.transform(le_target.classes_))))
eng = pd.DataFrame()
eng["Time_spent_Alone"] = data["Time_spent_Alone"]
eng["Social_event_attendance"] = data["Social_event_attendance"]
eng["Going_outside"] = data["Going_outside"]
eng["Friends_circle_size"] = data["Friends_circle_size"]
eng["Post_frequency"] = data["Post_frequency"]
# Feature engineering
eng["Outdoors_Adjusted_Frequency"] = (data["Going_outside"] *
data["Friends_circle_size"]).clip(0,50)
eng["Digital_Extroversion_Score"] = (data["Post_frequency"] *
data["Friends_circle_size"]).clip(0,150)
eng["Social_Engagement_Index"] = (data["Social_event_attendance"] *
data["Friends_circle_size"]).clip(0,150)
# FIX: do NOT divide for energy drain
eng["Energy_Drain_Indicator"] =
data["Drained_after_socializing"].fillna(0).astype(float)
X = eng.copy()
y = data["_y"]
X.describe()
X.head()
X = X.fillna(X.median())
X_train, X_test, y_train, y_test =
train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
models = {
 "LR": LogisticRegression(max_iter=500),
 "RF": RandomForestClassifier(),
 "SVM RBF": SVC(),
 "XGB": XGBClassifier(eval_metric='logloss')
}
print("\nCLASSICAL MODELS F1:\n")
for name,model in models.items():
 model.fit(X_train,y_train)
 preds = model.predict(X_test)
 print(name, "MacroF1:", f1_score(y_test,preds,average="macro"))
from sklearn.model_selection import GridSearchCV
param_lr = {
 "C":[0.01,0.1,1,5,10],
 "penalty":["l2"],
 "solver":["lbfgs"]
}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_lr,
 scoring="f1_macro", cv=3, n_jobs=-1, verbose=1)
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
pred_lr = best_lr.predict(X_test)
print("\n==== LOGISTIC REGRESSION ====")
print("MacroF1:", f1_score(y_test,pred_lr,average="macro"))
print("Accuracy:", accuracy_score(y_test,pred_lr))
print("Best LR Params:", grid_lr.best_params_)
param_svm = {
 "C":[0.1,1,5,10],
 "gamma":[0.01,0.05,0.1,0.2]
}
grid_svm = GridSearchCV(SVC(probability=True), param_svm,
 scoring="f1_macro", cv=3, n_jobs=-1, verbose=1)
grid_svm.fit(X_train, y_train)
best_svm = grid_svm.best_estimator_
pred_svm = best_svm.predict(X_test)
print("\n==== SVM ====")
print("MacroF1:", f1_score(y_test,pred_svm,average="macro"))
print("Accuracy:", accuracy_score(y_test,pred_svm))
print("Best SVM Params:", grid_svm.best_params_)
param_xgb = {
 "max_depth":[3,5,7],
 "learning_rate":[0.01,0.05,0.1],
 "n_estimators":[100,200]
}
grid_xgb = GridSearchCV(
 XGBClassifier(eval_metric='logloss', use_label_encoder=False),
 param_xgb,
 scoring="f1_macro", cv=3, n_jobs=-1, verbose=1
)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
pred_xgb = best_xgb.predict(X_test)
print("\n==== XGBOOST ====")
print("MacroF1:", f1_score(y_test,pred_xgb,average="macro"))
print("Accuracy:", accuracy_score(y_test,pred_xgb))
print("Best XGB Params:", grid_xgb.best_params_)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import numpy as np
from sklearn.metrics import f1_score, classification_report
import joblib
import os
RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
try:
 X_train.shape
 X_test.shape
 y_train.shape
 y_test.shape
except Exception as e:
 raise RuntimeError("X_train/X_test/y_train/y_test not found in workspace.
Run preprocessing + split first.") from e
from sklearn.utils import class_weight
cw = class_weight.compute_class_weight(class_weight='balanced',
classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: w for cls, w in zip(np.unique(y_train), cw)}
print("Class weights:", class_weight_dict)
# small helper to build GELU layer (tf has GELU activation, fallback to
tf.nn.gelu)
def gelu_activation(x):
 try:
 return tf.keras.activations.gelu(x) # tf 2.8+
 except Exception:
 return tf.nn.gelu(x)
# small helper to build GELU layer (tf has GELU activation, fallback to
tf.nn.gelu)
def gelu_activation(x):
 try:
 return tf.keras.activations.gelu(x) # tf 2.8+
 except Exception:
 return tf.nn.gelu(x)
def build_dnn(input_dim, units_list=[64,32], dropout=0.15, lr=1e-3):
 inp = layers.Input(shape=(input_dim,))
 x = inp
 for u in units_list:
 x = layers.Dense(u)(x)
 x = layers.Activation(gelu_activation)(x)
 x = layers.Dropout(dropout)(x)
 x = layers.Dense(1, activation="sigmoid")(x)
 model = models.Model(inputs=inp, outputs=x)
 model.compile(optimizer=optimizers.Adam(learning_rate=lr),
 loss="binary_crossentropy",
 metrics=["accuracy"])
 return model
input_dim = X_train.shape[1]
dnn_shallow = build_dnn(input_dim, units_list=[64,32], dropout=0.15, lr=1e-3)
dnn_deep = build_dnn(input_dim, units_list=[128,64,32], dropout=0.15,
lr=5e-4)
es = callbacks.EarlyStopping(monitor="val_loss", patience=6,
restore_best_weights=True)
cb_list = [es]
BATCH_SIZE = 32
EPOCHS = 50
VERBOSE = 2
print("\n=== Training DNN (shallow) ===")
hist_shallow = dnn_shallow.fit(
 X_train, y_train,
 validation_split=0.15,
 epochs=EPOCHS,
 batch_size=BATCH_SIZE,
 class_weight=class_weight_dict,
 callbacks=cb_list,
 verbose=VERBOSE
)
y_pred_shallow = (dnn_shallow.predict(X_test, batch_size=BATCH_SIZE) >
0.5).astype(int).ravel()
f1_shallow = f1_score(y_test, y_pred_shallow, average="macro")
print("\nDNN (shallow) Macro F1:", f1_shallow)
print("Classification report (shallow):\n", classification_report(y_test,
y_pred_shallow))
print("\n=== Training DNN (deep) ===")
hist_deep = dnn_deep.fit(
 X_train, y_train,
 validation_split=0.15,
 epochs=EPOCHS,
 batch_size=BATCH_SIZE,
 class_weight=class_weight_dict,
 callbacks=cb_list,
 verbose=VERBOSE
)
y_pred_deep = (dnn_deep.predict(X_test, batch_size=BATCH_SIZE) >
0.5).astype(int).ravel()
f1_deep = f1_score(y_test, y_pred_deep, average="macro")
print("\nDNN (deep) Macro F1:", f1_deep)
print("Classification report (deep):\n", classification_report(y_test,
y_pred_deep))
rf = RandomForestClassifier(random_state=RANDOM_STATE,
class_weight="balanced")
param_grid = {
 "n_estimators":[200,400],
 "max_depth":[4,6,8],
 "min_samples_split":[2,5]
}
grid = GridSearchCV(
 rf,
 param_grid,
 scoring="f1_macro",
 cv=3,
 n_jobs=-1,
 verbose=2
)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
print("Best RF Params:", grid.best_params_)
preds = best_rf.predict(X_test)
print("Tuned RF MacroF1:", f1_score(y_test, preds, average="macro"))
print("\nClassification report:")
print(classification_report(y_test,preds))
import shap
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test) # shape (n_samples, n_features)
global_importance = np.mean(np.abs(shap_values), axis=0)
feat_importance = (
 pd.DataFrame({
 "feature": X.columns,
 "mean_abs_shap": global_importance
 }).sort_values("mean_abs_shap", ascending=False)
)
print(feat_importance)
# rebuild SVM with probability=True and same params:
best_svm = SVC(
 C=grid_svm.best_params_["C"],
 gamma=grid_svm.best_params_["gamma"],
 probability=True
)
best_svm.fit(X_train, y_train)
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(
 estimators=[('lr', best_lr), ('svm', best_svm), ('xgb', best_xgb)],
 voting='soft',
 n_jobs=-1
)
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
y_proba = voting_clf.predict_proba(X_test)[:,1]
print("Soft Voting Macro F1:", f1_score(y_test,y_pred,average="macro"))
print("Accuracy:", accuracy_score(y_test,y_pred))
print("ROC AUC:", roc_auc_score(y_test,y_proba))
import pandas as pd
results = {
 "Model":[
 "Logistic Regression",
 "SVM (RBF)",
 "XGBoost",
 "Soft Voting Ensemble",
 "DNN (Shallow)",
 "DNN (Deep)"
 ],
 "Type":[
 "Classical ML",
 "Classical ML",
 "Classical ML",
 "Ensemble",
 "Deep Learning",
 "Deep Learning"
 ],
 "Macro F1":[
 0.9622173810245025,
 0.9629665462676298,
 0.9619311124180581,
 0.9629665462676298,
 0.9616201594597384,
 0.9623179747422886
 ],
 "Accuracy":[
 0.9708502024291498,
 0.9713900134952766,
 0.9705802968960864,
 0.9713900134952766,
 0.9700000000000000, # shallow
 0.9700000000000000 # deep
 ],
 "Best Params":[
 "{C:0.1, penalty:l2, solver:lbfgs}",
 "{C:1, gamma:0.2}",
 "{max_depth:3, learning_rate:0.1, n_estimators:100}",
 "LR + SVM Soft Voting",
 "2 Dense Layers (Shallow) + EarlyStop",
 "3 Dense Layers (Deep) + EarlyStop"
 ]
}
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
import matplotlib.pyplot as plt
feat_importance_sorted = feat_importance.sort_values("mean_abs_shap",
ascending=True)
plt.figure(figsize=(8,5))
plt.barh(feat_importance_sorted["feature"],
feat_importance_sorted["mean_abs_shap"])
plt.xlabel("Mean |SHAP| Value")
plt.ylabel("Feature")
plt.title("Global SHAP Feature Importance - Horizontal Bar")
plt.tight_layout()
plt.show()
