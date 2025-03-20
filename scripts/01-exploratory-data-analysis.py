
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/bonisvc/telcoChurnPrediction

credentials = {}
with open('/content/drive/MyDrive/credentials.txt', 'r') as f:
    for line in f:
        key, value = line.strip().split('=')
        credentials[key] = value

username=credentials['username']
email=credentials['email']
token=credentials['token']
import os

credentials = {}
try:
    with open('/content/drive/MyDrive/credentials.txt', 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            credentials[key] = value

    username = credentials['username']
    email = credentials['email']
    token = credentials['token']

    if not all(key in credentials for key in ('username', 'email', 'token')):
        raise ValueError("Credenciais incompletas no arquivo credentials.txt")

    try:
        %cd telcoChurnPrediction
    except Exception as e:
        print(f"Erro ao mudar de diretório: {e}")

    try:
        !git config --global user.name '$username'
        !git config --global user.email '$email'
        !git remote set-url origin https://$username:$token@github.com/$username/telcoChurnPrediction.git
        !git push origin main
        !git remote set-url origin https://github.com/$username/telcoChurnPrediction.git #remove o token da url

    except Exception as git_error:
        print(f"Erro ao executar comandos Git: {git_error}")

except Exception as overall_error:
    print(f"Erro geral: {overall_error}")
!pip install lifelines

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import pointbiserialr
from lifelines import KaplanMeierFitter

path= f'https://raw.githubusercontent.com/{username}/telcoChurnPrediction/refs/heads/main/data/raw/telco-customer-churn.csv' #raw data path

try:
  df = pd.read_csv(path)
  print(df.sample(5))
except Exception as e:
  print(e)

df.info()

df['TotalCharges'] = df['TotalCharges'].apply(lambda x: 0 if x == ' ' else float(x))

df.describe()

print(df.isnull().sum())
print(df.duplicated().sum())

df['Churn'].value_counts(normalize=True)

categorical_cols = df.select_dtypes(include='object').columns
categorical_cols = categorical_cols.drop(['customerID', 'Churn'])

num_cols = 3
num_graphs = len(categorical_cols)
num_rows = math.ceil(num_graphs / num_cols)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, hue='Churn', data=df, ax=axes[i])
    axes[i].set_title(f'Churn Rate by {col}')
    axes[i].tick_params(axis='x', rotation=45)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

numerical_cols = df.select_dtypes(include='number').columns

df.groupby('Churn')[numerical_cols].agg(['mean', 'median', 'std', 'min', 'max'])

correlations = {}
for col in numerical_cols:
    corr, pval = pointbiserialr(df[col], df['Churn'].map({'Yes': 1, 'No': 0}))
    correlations[col] = corr

sns.heatmap(pd.DataFrame(correlations, index=['Correlation']).T, annot=True, cmap='coolwarm')
plt.show()
for col in categorical_cols:
    cross_tab = pd.crosstab(df[col], df['Churn'], normalize='index')
    print(f"Crosstab for {col}:")
    print(cross_tab)
    print("-" * 20)

import matplotlib.pyplot as plt
import math
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()

kmf.fit(df['tenure'], event_observed=df['Churn'].map({'Yes': 1, 'No': 0}))

plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Tenure (Months)')
plt.ylabel('Survival Probability')
plt.show()

num_cols = 3
num_graphs = len(categorical_cols)
num_rows = math.ceil(num_graphs / num_cols)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    for category in df[col].unique():
        subset = df[df[col] == category]
        if subset.empty:
            continue

        kmf.fit(subset['tenure'], event_observed=subset['Churn'].map({'Yes': 1, 'No': 0}), label=category)

        kmf.plot_survival_function(ax=axes[i])

    axes[i].set_title(f'Survival Curve by {col}')
    axes[i].set_xlabel('Tenure (Months)')
    axes[i].set_ylabel('Survival Probability')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
from sklearn.preprocessing import OneHotEncoder

df_aux = df.copy()

for col in categorical_cols:
  encoder = OneHotEncoder(sparse_output=False, drop='first')
  encoded_cor = encoder.fit_transform(df[col].values.reshape(-1, 1))
  encoded_cor_df = pd.DataFrame(encoded_cor, columns=encoder.get_feature_names_out([col]))
  df_aux = pd.concat([df_aux, encoded_cor_df], axis=1)
  df_aux = df_aux.drop(col, axis=1)

df_aux['Churn'] = df_aux['Churn'].astype(bool)
df_aux.drop('customerID', axis=1, inplace=True)

df_aux.head()
df_aux.info()plt.figure(figsize=(10, 8))
sns.heatmap(df_aux.drop('Churn', axis=1).corr(), annot=False, cmap='coolwarm')
plt.show()from sklearn.preprocessing import StandardScaler
import pickle
import os

folder = '/content/telcoChurnPrediction/models'

if not os.path.exists(folder):
  os.makedirs(folder)

def modelSaver(model, path):
  with open(path, 'wb') as file:
    pickle.dump(model, file)
  print(f'Modelo salvo em: {path}')

for col in numerical_cols:
  scaler = StandardScaler()
  df_aux[col] = scaler.fit_transform(df_aux[col].values.reshape(-1, 1))
  modelSaver(scaler, f'{folder}/{col}_scaler.pkl')
from sklearn.model_selection import train_test_split

X = df_aux.drop('Churn', axis=1)
y = df_aux['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)import json

folder_path = '/content/telcoChurnPrediction/'

if not os.path.exists(f'{folder_path}data/processed'):
  os.makedirs(f'{folder_path}data/processed')

if not os.path.exists(f'{folder_path}scripts'):
  os.makedirs(f'{folder_path}scripts')


X_train.to_csv(f'{folder_path}data/processed/X_train.csv', index=False)
print('Salvo')
X_test.to_csv(f'{folder_path}data/processed/X_test.csv', index=False)
print('Salvo')
y_train.to_csv(f'{folder_path}data/processed/y_train.csv', index=False)
print('Salvo')
y_test.to_csv(f'{folder_path}data/processed/y_test.csv', index=False)
print(f'Todos os arquivos salvos em {folder_path}data/processed/')

path = '/content/drive/MyDrive/01-exploratory-data-analysis.ipynb'
with open(path, 'r') as f:
  notebook_content = json.load(f)

codigo = ""
for cell in notebook_content['cells']:
  if cell['cell_type'] == 'code':
      for line in cell['source']:
          codigo += line

arquivo_py = os.path.join('/content/telcoChurnPrediction/scripts', "01-exploratory-data-analysis.py")

with open(arquivo_py, 'w') as f:
  f.write(codigo)

print(f"Script salvo em: {arquivo_py}")

!git config --global user.name '$username'
!git config --global user.email '$email'

!git add /content/telcoChurnPrediction
!git pull origin main
!git commit -m 'exploratory analysis script & processed data'
!git push https://$username:$token@github.com/$username/telcoChurnPrediction.git main
!rm -rf /content/telcoChurnPrediction