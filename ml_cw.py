# %% [markdown]

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# %%
import tensorflow as tf

# %%
import joblib
import keras_tuner as kt

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# ### Data Collection and Cleaning

# %% [markdown]
# On initial glance the csv files are seperated with `;` instead of `,`. And there happens to be seperated `bank` and `bank-addtional` files 

# %%
bank = pd.read_csv("bank+marketing/bank/bank.csv", sep=';')
bank

# %%
bank_full = pd.read_csv("bank+marketing/bank/bank-full.csv", sep=';')
bank_full

# %%
bank_additional = pd.read_csv("bank+marketing/bank-additional/bank-additional.csv", sep=';')
bank_additional

# %%
bank_additional_full = pd.read_csv("bank+marketing/bank-additional/bank-additional-full.csv", sep=';')
bank_additional_full

# %%
bank.isna().sum(), bank_full.isna().sum(), bank_additional.isna().sum(), bank_additional_full.isna().sum()

# %% [markdown]
# There are no na or null values present in the 4 dataframes

# %%
common_rows_bank = bank_full.merge(bank)
common_rows_bank

# %%
common_rows_bank_additional = bank_additional_full.merge(bank_additional)
common_rows_bank_additional

# %% [markdown]
# Every row in both of `bank` and `bank-additional` are present in `bank-full` and `bank-additional-full`, the representations are 10% of the original datasets. Therefore we can safely ignore the smaller dataset and proceed with the larger `bank-full` and `bank-additional-full`.

# %%
bank_additional_full.columns

# %% [markdown]
# There are additioanl 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed' in the `bank-additional` dataset

# %%
common_columns = bank_additional_full.columns.intersection(bank_full.columns)
common_columns_dataframe = bank_additional_full[common_columns]
common_columns_dataframe

# %%
common_rows_bank_all = common_columns_dataframe.merge(bank_full)
common_columns_dataframe

# %% [markdown]
# Every row in the in `bank-full` is in `bank-additional-full`, Therefore Project will the done with the use of `bank-additional-full` dataset because of more data.

# %%
bank_additional_full.drop_duplicates(inplace=True)
bank_additional_full.shape

# %%
bank_additional_full['y'].value_counts()

# %% [markdown]
# ### Data Visualization and Exploratory Data Analysis

# %%
bank_additional_full.info()

# %%
bank_additional_full.head()

# %%
bank_additional_full['job'].value_counts()

# %%
bank_additional_full.describe()

# %%
sns.pairplot(bank_additional_full, hue='y')

# %% [markdown]
# Data is appeared to be heavily clustered and some outliers can be seen

# %%
sns.boxplot(bank_additional_full['duration'])

# %% [markdown]
# There appears to be some outliers present in the `duration` column (visualized in the box plot above), by using a 0.05 - 0.95 confidence interval we can eleminate the outliers

# %%
def remove_outliers_iqr(data:pd.DataFrame, columns):
    data_copy = data.copy()
    for column in columns:
        Q1 = data_copy[column].quantile(0.05)
        Q3 = data_copy[column].quantile(0.95)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        data_copy.drop(data_copy[(data_copy[column] < lower_bound) | (data_copy[column] > upper_bound)].index, inplace=True)

    return data_copy

# %%
bank_additional_full_rm_outliers = remove_outliers_iqr(bank_additional_full, ['duration'])
sns.boxplot(bank_additional_full_rm_outliers['duration'])

# %%
bank_additional_full.shape[0] - bank_additional_full_rm_outliers.shape[0]

# %% [markdown]
# 93 records were removed from the dataset

# %% [markdown]
# #### Class imbalance fixing

# %%
y_value_count = bank_additional_full_rm_outliers['y'].value_counts()
minority_class_len = y_value_count[1]
y_value_count, minority_class_len

# %% [markdown]
# As we can see in the dataset there is a significant imbalance between yes and no classes. Deep Neural Networks tend to perform worse with the class imbalances therefore class imbalance need to be fixed. Due to nature of dataset undersampling is selected to fix the class imbalance.

# %%
minority = bank_additional_full_rm_outliers[bank_additional_full_rm_outliers['y'] == 'yes']
majority = bank_additional_full_rm_outliers[bank_additional_full_rm_outliers['y'] == 'no']
majority_undersampled = majority.sample(minority_class_len, random_state=42)
majority_minority_combined = pd.concat([minority, majority_undersampled])
majority_minority_combined

# %% [markdown]
# ### Encoding

# %%
rf_data_copy = majority_minority_combined.copy()

# %%
rf_data_copy.dtypes

# %%
rf_data_copy_encoded = rf_data_copy.copy()

# %%
label_encoders = {}
for column in rf_data_copy_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    rf_data_copy_encoded[column] = le.fit_transform(rf_data_copy_encoded[column])
    label_encoders[column] = le

# %%
label_encoders

# %%
rf_data_copy_encoded

# %%
plt.figure(figsize=(14, 8))
sns.heatmap(rf_data_copy_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

# %% [markdown]
# There are 20 features in the dataset Dimentionality reduction can be performed to reduce the number of features. Principal Component Analysis will be used.

# %%
y = rf_data_copy_encoded['y']
rf_data_copy_encoded.drop(columns=['y'], inplace=True)
rf_data_copy_encoded

# %%
scaler = StandardScaler()
rf_data_copy_encoded_scaled = scaler.fit_transform(rf_data_copy_encoded)
rf_data_copy_encoded_scaled = pd.DataFrame(rf_data_copy_encoded_scaled, columns=rf_data_copy_encoded.columns)
rf_data_copy_encoded_scaled

# %% [markdown]
# #### PCA

# %%
pca = PCA()
pca_data = pca.fit(rf_data_copy_encoded_scaled)

# %%
np.cumsum(pca.explained_variance_ratio_)

# %%
pca_17_components = PCA(n_components=17)
pca_17_component_data = pca_17_components.fit_transform(rf_data_copy_encoded_scaled)

# %%
pca_columns = [f"PC{i+1}" for i in range(pca_17_components.n_components)]
X_pca_df = pd.DataFrame(pca_17_component_data, columns=pca_columns)

# %%
X_pca_df

# %%
# Saving objects
joblib.dump(pca, 'pca.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

# %% [markdown]
# ## Training

# %%
X_train, X_test, y_train, y_test = train_test_split(X_pca_df, y, test_size=0.2, random_state=42)

# %% [markdown]
# ### Random Forest 

# %%
randomforest = RandomForestClassifier(n_jobs=-1, n_estimators=200)
randomforest.fit(X_train, y_train)

# %%
rf_predictions = randomforest.predict(X_test)

# %%
print(confusion_matrix(y_pred=rf_predictions, y_true=y_test))
print(classification_report(y_pred=rf_predictions, y_true=y_test))

# %%
from sklearn.metrics import make_scorer, roc_auc_score
auc_scoring = make_scorer(roc_auc_score)

# %%
param_grid = {
    'n_estimators': range(200, 1000, 200),
    'max_features': ['sqrt', 'log2'],
    'max_depth': range(1, 51, 2),
    'min_samples_split': range(2, 15, 2),
    'min_samples_leaf': [1, 2, 4, 6],
    'criterion': ['gini', 'entropy']
}

rf = RandomForestClassifier()
rf_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, verbose=2, scoring=auc_scoring, cv=2, n_iter=100, n_jobs=-1)
rf_search.fit(X_train, y_train)

print("Best parameters found: ", rf_search.best_params_)
print("Best precision found: ", rf_search.best_score_)

# %%
optimized_rf_model = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=6,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=29,
    criterion='entropy',
    n_jobs=-1
)

optimized_rf_model.fit(X_train, y_train)

optimized_rf_predictions = optimized_rf_model.predict(X_test)

print(confusion_matrix(y_pred=optimized_rf_predictions, y_true=y_test))
print(classification_report(y_pred=optimized_rf_predictions, y_true=y_test))

# %%
joblib.dump(optimized_rf_model, 'optimized_rf_model.pkl')

# %% [markdown]
# ### Tensorflow

# %% [markdown]
# #### Tensor Conversion

# %%
X_train_bal = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_val_bal = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train_bal = tf.convert_to_tensor(y_train.values, dtype=tf.int32)
y_val_bal = tf.convert_to_tensor(y_test.values, dtype=tf.int32)

# %%
BATCH_SIZE = 32

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_bal, y_train_bal)).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_bal, y_val_bal)).batch(BATCH_SIZE)

# %% [markdown]
# #### Model Building

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=17, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
], name='bank_marketing_model')

model.summary()

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)

callbacks = [early_stopping, tensorboard_callback, model_checkpoint]

# %%
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %% [markdown]
# #### Model Training

# %%
model.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=callbacks)

# %%
model.evaluate(val_dataset)

# %%
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu',
        input_shape=(X_train_bal.shape[1],)
    ))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            activation='relu'
        ))
        model.add(tf.keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# %%
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=2,
    directory='kt_dir',
    project_name='bank_marketing_tuning'
)

tuner.search(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[early_stopping])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

rs_dl_model = tuner.hypermodel.build(best_hps)
rs_dl_history = rs_dl_model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[early_stopping])

# %%
rs_dl_model.summary()

# %%
print("Best hyperparameters:")
for param in best_hps.values.keys():
    print(f"{param}: {best_hps.get(param)}")

# %%
dl_predictions = rs_dl_model.predict(val_dataset)
dl_predictions = tf.round(dl_predictions).numpy().astype(int).flatten()

# %%
rs_dl_model.save('rs_dl_model.keras')

# %% [markdown]
# ## Evaluations

# %% [markdown]
# A Detailed evaluation with comparison of the `Random Forest` and the `Deep learning model` 

# %%
rf_classification_report = classification_report(y_test, optimized_rf_predictions, target_names=['no', 'yes'])
print("Random Forest Model Classification Report:\n", rf_classification_report)

dl_classification_report = classification_report(y_test, dl_predictions, target_names=['no', 'yes'])
print("Deep Learning Model Classification Report:\n", dl_classification_report)

# %% [markdown]
# ## Data Pipeline and Experimentation

# %%
saved_pca = joblib.load('pca.pkl')
saved_label_encoders = joblib.load('label_encoders.pkl')
saved_scaler = joblib.load('scaler.pkl')
saved_model = tf.keras.models.load_model('rs_dl_model.keras')
saved_rf_model = joblib.load('optimized_rf_model.pkl')

# %%
from imblearn.over_sampling import SMOTE

# %%
X_exp = bank_additional_full_rm_outliers.drop(columns=['y'])
y_exp = bank_additional_full_rm_outliers['y']

label_encoders_exp = {}

X_exp_encoded = X_exp.copy()
for column in X_exp_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X_exp_encoded[column] = le.fit_transform(X_exp_encoded[column])
    label_encoders_exp[column] = le

y_exp_encoded = label_encoders['y'].transform(y_exp)

smote = SMOTE(random_state=42)
X_exp_resampled, y_exp_resampled = smote.fit_resample(X_exp_encoded, y_exp_encoded)

bank_additional_full_smote = pd.DataFrame(X_exp_resampled, columns=X_exp_encoded.columns)
bank_additional_full_smote['y'] = y_exp_resampled

bank_additional_full_smote['y'].value_counts()

# %%
X_smote = bank_additional_full_smote.drop(columns=['y'])
y_smote = bank_additional_full_smote['y']

X_smote_scaled = scaler.transform(X_smote)
X_smote_scaled = pd.DataFrame(X_smote_scaled, columns=X_smote.columns)

X_smote_pca = pca_17_components.transform(X_smote_scaled)
X_smote_pca_df = pd.DataFrame(X_smote_pca, columns=pca_columns)

X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(X_smote_pca_df, y_smote, test_size=0.3, random_state=42)

X_exp_train = tf.convert_to_tensor(X_exp_train, dtype=tf.float32)
y_exp_train = tf.convert_to_tensor(y_exp_train, dtype=tf.int32)

X_exp_test = tf.convert_to_tensor(X_exp_test, dtype=tf.float32)
y_exp_test = tf.convert_to_tensor(y_exp_test, dtype=tf.float32)

train_exp_tensor = tf.data.Dataset.from_tensor_slices((X_exp_train, y_exp_train)).batch(BATCH_SIZE)
val_exp_tensor = tf.data.Dataset.from_tensor_slices((X_exp_test, y_exp_test)).batch(BATCH_SIZE)

# %%
rf_exp = RandomForestClassifier(n_jobs=-1, n_estimators=200)
rf_exp.fit(X_exp_train, y_exp_train)

rf_exp_predictions = randomforest.predict(X_exp_test)

# %%
model_exp = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=17, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_exp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

tensorboard_callback_exp = tf.keras.callbacks.TensorBoard(log_dir='./logs/exp')
model_checkpoint_exp = tf.keras.callbacks.ModelCheckpoint('best_model_exp.keras', monitor='val_accuracy', save_best_only=True)

callbacks = [early_stopping, tensorboard_callback_exp, model_checkpoint_exp]

model_exp.fit(train_exp_tensor, validation_data=val_exp_tensor, epochs=100, callbacks=callbacks)

model_exp.evaluate(val_dataset)

# %%
exp_dl_pred = model_exp.predict(val_exp_tensor)
exp_dl_pred = tf.round(exp_dl_pred).numpy().astype(int).flatten()

# %%
rf_confusion_matrix = confusion_matrix(y_exp_test, rf_exp_predictions)
rf_classification_report = classification_report(y_exp_test, rf_exp_predictions, target_names=['no', 'yes'])

print("Random Forest Model Confusion Matrix:\n", rf_confusion_matrix)
print("Random Forest Model Classification Report:\n", rf_classification_report)

dl_confusion_matrix = confusion_matrix(y_exp_test, exp_dl_pred)
dl_classification_report = classification_report(y_exp_test, exp_dl_pred, target_names=['no', 'yes'])

print("Deep Learning Model Confusion Matrix:\n", dl_confusion_matrix)
print("Deep Learning Model Classification Report:\n", dl_classification_report)

# %% [markdown]
# Both random forest and the neural networks shows improvements over the undersampling

# %%
def preprocess_data_rf(data, label_encoders, scaler, pca):
    data_copy = data.copy()
    for column in data_copy.select_dtypes(include='object').columns:
        le = label_encoders.get(column)
        data_copy[column] = le.transform(data_copy[column])
    data_scaled = scaler.transform(data_copy)
    data_pca = pca.transform(data_scaled)
    return pd.DataFrame(data_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)])

def preprocess_data_dl(data, label_encoders, scaler, pca):
    rf_data = preprocess_data_rf(data, label_encoders, scaler, pca)
    return tf.convert_to_tensor(rf_data, dtype=tf.float32)

def random_forest_prediction(data, label_encoders, scaler, model, pca):
    preprocessed_data = preprocess_data_rf(data, label_encoders, scaler, pca)
    predictions = model.predict(preprocessed_data)
    return predictions

def deep_learning_prediction(data, label_encoders, scaler, model, pca):
    preprocessed_data = preprocess_data_dl(data, label_encoders, scaler, pca)
    predictions = model.predict(preprocessed_data)
    predictions = tf.round(predictions).numpy().astype(int).flatten()
    return predictions

# %%
def get_user_input():
    user_data = {}
    user_data['age'] = int(input("Enter age: "))
    user_data['job'] = input("Enter job: ")
    user_data['marital'] = input("Enter marital status: ")
    user_data['education'] = input("Enter education: ")
    user_data['default'] = input("Enter default status: ")
    user_data['housing'] = input("Enter housing status: ")
    user_data['loan'] = input("Enter loan status: ")
    user_data['contact'] = input("Enter contact type: ")
    user_data['month'] = input("Enter month: ")
    user_data['day_of_week'] = input("Enter day of the week: ")
    user_data['duration'] = int(input("Enter duration: "))
    user_data['campaign'] = int(input("Enter campaign: "))
    user_data['pdays'] = int(input("Enter pdays: "))
    user_data['previous'] = int(input("Enter previous: "))
    user_data['poutcome'] = input("Enter poutcome: ")
    user_data['emp.var.rate'] = float(input("Enter emp.var.rate: "))
    user_data['cons.price.idx'] = float(input("Enter cons.price.idx: "))
    user_data['cons.conf.idx'] = float(input("Enter cons.conf.idx: "))
    user_data['euribor3m'] = float(input("Enter euribor3m: "))
    user_data['nr.employed'] = float(input("Enter nr.employed: "))
    
    return pd.DataFrame([user_data])


