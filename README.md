# Predictive Risk Scoring - Risk Modelling

The goal of this project is to predictively score the risk of heart disease in an individual. 

## Describing the Dataset

The dataset was obtained from Kaggle, and consists of 10,000 rows.


```python
df = pd.read_json("data.json")
display(df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>weight</th>
      <th>sex</th>
      <th>height</th>
      <th>sys_bp</th>
      <th>smoker</th>
      <th>nic_other</th>
      <th>num_meds</th>
      <th>occup_danger</th>
      <th>ls_danger</th>
      <th>...</th>
      <th>addiction</th>
      <th>major_surgery_num</th>
      <th>diabetes</th>
      <th>hds</th>
      <th>cholesterol</th>
      <th>asthma</th>
      <th>immune_defic</th>
      <th>family_cancer</th>
      <th>family_heart_disease</th>
      <th>family_cholesterol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>219</td>
      <td>m</td>
      <td>74</td>
      <td>136</td>
      <td>n</td>
      <td>n</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>n</td>
      <td>0</td>
      <td>n</td>
      <td>y</td>
      <td>203</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>66</td>
      <td>242</td>
      <td>m</td>
      <td>73</td>
      <td>111</td>
      <td>n</td>
      <td>n</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>y</td>
      <td>0</td>
      <td>n</td>
      <td>n</td>
      <td>228</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31</td>
      <td>197</td>
      <td>f</td>
      <td>65</td>
      <td>112</td>
      <td>n</td>
      <td>n</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>y</td>
      <td>3</td>
      <td>n</td>
      <td>y</td>
      <td>183</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>244</td>
      <td>f</td>
      <td>69</td>
      <td>127</td>
      <td>n</td>
      <td>n</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>n</td>
      <td>2</td>
      <td>n</td>
      <td>y</td>
      <td>228</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>4</th>
      <td>93</td>
      <td>183</td>
      <td>f</td>
      <td>63</td>
      <td>91</td>
      <td>y</td>
      <td>n</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>y</td>
      <td>2</td>
      <td>n</td>
      <td>n</td>
      <td>169</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



```python
display(df.describe())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>weight</th>
      <th>height</th>
      <th>sys_bp</th>
      <th>num_meds</th>
      <th>occup_danger</th>
      <th>ls_danger</th>
      <th>drinks_aweek</th>
      <th>major_surgery_num</th>
      <th>cholesterol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.00000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.0000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>64.63570</td>
      <td>214.725500</td>
      <td>67.237100</td>
      <td>126.482400</td>
      <td>4.590500</td>
      <td>1.996800</td>
      <td>2.0056</td>
      <td>9.983400</td>
      <td>4.170900</td>
      <td>199.736100</td>
    </tr>
    <tr>
      <th>std</th>
      <td>23.19368</td>
      <td>35.624989</td>
      <td>3.897127</td>
      <td>14.279162</td>
      <td>5.512372</td>
      <td>0.819425</td>
      <td>0.8166</td>
      <td>5.556601</td>
      <td>2.964013</td>
      <td>35.633212</td>
    </tr>
    <tr>
      <th>min</th>
      <td>25.00000</td>
      <td>97.000000</td>
      <td>53.000000</td>
      <td>67.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>83.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>46.00000</td>
      <td>190.000000</td>
      <td>64.000000</td>
      <td>117.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.0000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>175.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>64.00000</td>
      <td>214.000000</td>
      <td>67.000000</td>
      <td>126.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.0000</td>
      <td>9.000000</td>
      <td>4.000000</td>
      <td>199.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>82.00000</td>
      <td>238.000000</td>
      <td>70.000000</td>
      <td>136.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>3.0000</td>
      <td>13.000000</td>
      <td>6.000000</td>
      <td>223.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>120.00000</td>
      <td>366.000000</td>
      <td>82.000000</td>
      <td>180.000000</td>
      <td>53.000000</td>
      <td>3.000000</td>
      <td>3.0000</td>
      <td>37.000000</td>
      <td>16.000000</td>
      <td>351.000000</td>
    </tr>
  </tbody>
</table>
</div>


## Visualizing the Data


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_json('data.json')
sns.set(style='whitegrid')
```


```python
# 1. Age distribution by health disease status (hds)
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='age', hue='hds', kde=True, palette='Set1', bins=30)
plt.title('Age Distribution by Health Disease Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
```


    
![png](output_8_0.png)
    



```python
# 2. Sex distribution by health disease status
plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='hds', data=df, palette='Set2')
plt.title('Sex Distribution by Health Disease Status')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()
```


    
![png](output_9_0.png)
    



```python
# 3. Cholesterol levels by health disease status (boxplot)
plt.figure(figsize=(8,5))
sns.boxplot(x='hds', y='cholesterol', data=df, palette='pastel')
plt.title('Cholesterol Levels by Health Disease Status')
plt.xlabel('Health Disease Status (hds)')
plt.ylabel('Cholesterol')
plt.show()
```

    /var/folders/68/7k4yhjwn6hz02wv392bx_2b80000gn/T/ipykernel_17586/3021393648.py:3: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.boxplot(x='hds', y='cholesterol', data=df, palette='pastel')



    
![png](output_10_1.png)
    



```python
# 4. Correlation heatmap of selected numerical features
num_features = ['age', 'weight', 'height', 'sys_bp', 'cholesterol', 'drinks_aweek', 'major_surgery_num']
plt.figure(figsize=(10,8))
corr = df[num_features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
```


    
![png](output_11_0.png)
    



```python
# 5. Pairplot of numerical variables colored by health disease status
sns.pairplot(df[num_features + ['hds']], hue='hds', corner=True, diag_kind='kde', palette='Set1')
plt.suptitle('Pairwise Plot of Numerical Variables by Health Disease Status', y=1.02)
plt.show()
```


    
![png](output_12_0.png)
    



```python
# 6. Stacked bar chart for addiction by health disease status
addiction_counts = df.groupby(['addiction', 'hds']).size().unstack().fillna(0)
addiction_counts.plot(kind='bar', stacked=True, figsize=(7,5), colormap='Paired')
plt.title('Addiction Status by Health Disease Status')
plt.xlabel('Addiction')
plt.ylabel('Count')
plt.show()
```


    
![png](output_13_0.png)
    



```python
# 7. Systolic blood pressure by smoking status (boxplot)
plt.figure(figsize=(8,5))
sns.boxplot(x='smoker', y='sys_bp', data=df, palette='Set3')
plt.title('Systolic Blood Pressure by Smoking Status')
plt.xlabel('Smoker')
plt.ylabel('Systolic Blood Pressure')
plt.show()
```

    /var/folders/68/7k4yhjwn6hz02wv392bx_2b80000gn/T/ipykernel_17586/2546053192.py:3: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.boxplot(x='smoker', y='sys_bp', data=df, palette='Set3')



    
![png](output_14_1.png)
    



```python
# 8. Number of medications distribution by health disease status
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='num_meds', hue='hds', multiple='stack', bins=15, palette='Set2')
plt.title('Number of Medications Distribution by Health Disease Status')
plt.xlabel('Number of Medications')
plt.ylabel('Count')
plt.show()
```


    
![png](output_15_0.png)
    


## Finding the Best Model for the Dataset

### Random Forest Classifier


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_json("data.json")

target = 'hds'
cat_cols = ['sex', 'smoker', 'nic_other', 'cannabis', 'opioids', 'other_drugs', 'addiction', 
            'diabetes', 'hds', 'asthma', 'immune_defic', 'family_cancer', 'family_heart_disease', 
            'family_cholesterol']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

features = df.columns.drop(target)

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred_proba = clf.predict_proba(X_test)[:,1]
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.75      0.77      0.76      1633
               1       0.72      0.69      0.70      1367
    
        accuracy                           0.74      3000
       macro avg       0.73      0.73      0.73      3000
    weighted avg       0.73      0.74      0.73      3000
    


### Using XGBoost


```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

df = pd.read_json("data.json")

target = 'hds'
cat_cols = ['sex', 'smoker', 'nic_other', 'cannabis', 'opioids', 'other_drugs', 'addiction', 
            'diabetes', 'hds', 'asthma', 'immune_defic', 'family_cancer', 'family_heart_disease', 
            'family_cholesterol']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

features = df.columns.drop(target)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)

xgb = XGBClassifier(eval_metric='logloss', random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid = GridSearchCV(xgb, param_grid, scoring='f1', cv=3)
grid.fit(X_train, y_train)

y_pred = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.76      0.79      0.77      1633
               1       0.74      0.71      0.72      1367
    
        accuracy                           0.75      3000
       macro avg       0.75      0.75      0.75      3000
    weighted avg       0.75      0.75      0.75      3000
    


### Using LightGBM


```python
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

df = pd.read_json('data.json')

cat_cols = ['sex', 'smoker', 'nic_other', 'cannabis', 'opioids', 'other_drugs', 'addiction',
            'diabetes', 'hds', 'asthma', 'immune_defic', 'family_cancer', 'family_heart_disease',
            'family_cholesterol']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

target = 'hds'
features = df.columns.drop(target)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)

lgbm = LGBMClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid = GridSearchCV(lgbm, param_grid, scoring='f1', cv=3)
grid.fit(X_train, y_train)
```

<style>#sk-container-id-11 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-11 {
  color: var(--sklearn-color-text);
}

#sk-container-id-11 pre {
  padding: 0;
}

#sk-container-id-11 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-11 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-11 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-11 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-11 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-11 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-11 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-11 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-11 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-11 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-11 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-11 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-11 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-11 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-11 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-11 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-11 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-11 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-11 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-11 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-11 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-11 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-11 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-11 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-11 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-11 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-11 div.sk-label label.sk-toggleable__label,
#sk-container-id-11 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-11 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-11 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-11 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-11 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-11 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-11 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-11 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-11 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-11 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-11 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-11 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-11 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-11" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=3, estimator=LGBMClassifier(random_state=42),
             param_grid={&#x27;learning_rate&#x27;: [0.01, 0.1, 0.2],
                         &#x27;max_depth&#x27;: [3, 5, 7], &#x27;n_estimators&#x27;: [100, 200]},
             scoring=&#x27;f1&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-27" type="checkbox" ><label for="sk-estimator-id-27" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=3, estimator=LGBMClassifier(random_state=42),
             param_grid={&#x27;learning_rate&#x27;: [0.01, 0.1, 0.2],
                         &#x27;max_depth&#x27;: [3, 5, 7], &#x27;n_estimators&#x27;: [100, 200]},
             scoring=&#x27;f1&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-28" type="checkbox" ><label for="sk-estimator-id-28" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: LGBMClassifier</div></div></label><div class="sk-toggleable__content fitted"><pre>LGBMClassifier(max_depth=3, random_state=42)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-29" type="checkbox" ><label for="sk-estimator-id-29" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LGBMClassifier</div></div></label><div class="sk-toggleable__content fitted"><pre>LGBMClassifier(max_depth=3, random_state=42)</pre></div> </div></div></div></div></div></div></div></div></div>




```python
y_pred = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.76      0.79      0.78      1633
               1       0.74      0.71      0.72      1367
    
        accuracy                           0.75      3000
       macro avg       0.75      0.75      0.75      3000
    weighted avg       0.75      0.75      0.75      3000
    


### Stacking / Ensembling


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

df = pd.read_json('data.json')
cat_cols = ['sex', 'smoker', 'nic_other', 'cannabis', 'opioids', 'other_drugs', 'addiction',
            'diabetes', 'hds', 'asthma', 'immune_defic', 'family_cancer', 'family_heart_disease',
            'family_cholesterol']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
target = 'hds'
features = df.columns.drop(target)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)
```


```python
estimators = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('xgb', XGBClassifier(eval_metric='logloss', random_state=42)),
    ('lgbm', LGBMClassifier(random_state=42))
]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3)
stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)
```

    [LightGBM] [Info] Number of positive: 3285, number of negative: 3715
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000763 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 731
    [LightGBM] [Info] Number of data points in the train set: 7000, number of used features: 23
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.469286 -> initscore=-0.123012
    [LightGBM] [Info] Start training from score -0.123012
    [LightGBM] [Info] Number of positive: 2190, number of negative: 2476
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000759 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 698
    [LightGBM] [Info] Number of data points in the train set: 4666, number of used features: 23
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.469353 -> initscore=-0.122743
    [LightGBM] [Info] Start training from score -0.122743
    [LightGBM] [Info] Number of positive: 2190, number of negative: 2477
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000794 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 704
    [LightGBM] [Info] Number of data points in the train set: 4667, number of used features: 23
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.469252 -> initscore=-0.123147
    [LightGBM] [Info] Start training from score -0.123147
    [LightGBM] [Info] Number of positive: 2190, number of negative: 2477
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000739 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 696
    [LightGBM] [Info] Number of data points in the train set: 4667, number of used features: 23
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.469252 -> initscore=-0.123147
    [LightGBM] [Info] Start training from score -0.123147



```python
print("Stacking Classifier Results:")
print(classification_report(y_test, y_pred_stack))
```

    Stacking Classifier Results:
                  precision    recall  f1-score   support
    
               0       0.76      0.78      0.77      1633
               1       0.73      0.70      0.72      1367
    
        accuracy                           0.75      3000
       macro avg       0.74      0.74      0.74      3000
    weighted avg       0.74      0.75      0.74      3000
    


### Neural Networks


```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

nn_eval = model.evaluate(X_test, y_test, verbose=0)
print(f"Neural Network Accuracy: {nn_eval[1]:.4f}")

y_pred_nn = (model.predict(X_test) > 0.5).astype("int32")
print("Neural Network Results:")
print(classification_report(y_test, y_pred_nn))
```

    /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/keras/src/layers/core/dense.py:92: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    Neural Network Accuracy: 0.7470
    [1m94/94[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 391us/step
    Neural Network Results:
                  precision    recall  f1-score   support
    
               0       0.73      0.85      0.78      1633
               1       0.77      0.63      0.69      1367
    
        accuracy                           0.75      3000
       macro avg       0.75      0.74      0.74      3000
    weighted avg       0.75      0.75      0.74      3000
    


## Summary of Model Fitting

Even after all the attempts to increase accuracy, the best value is 75%. The best model to proceed with is LightGBM, because it gives a good balance between accuracy, speed, and resource usage, which is helpful during deployment and real-time risk scoring.

## Implementing the LightGBM Model into Risk Scoring Algorithm

### Importing Libraries


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
```

### Pre-processing the Dataset


```python
df = pd.read_json('data.json')
cat_cols = ['sex', 'smoker', 'nic_other', 'cannabis', 'opioids', 'other_drugs', 'addiction',
            'diabetes', 'hds', 'asthma', 'immune_defic', 'family_cancer', 'family_heart_disease',
            'family_cholesterol']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

target = 'hds'
features = df.columns.drop(target)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)
```

### Training the Model


```python
# Train LightGBM model
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)
```

    [LightGBM] [Info] Number of positive: 3285, number of negative: 3715
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.001866 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 731
    [LightGBM] [Info] Number of data points in the train set: 7000, number of used features: 23
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.469286 -> initscore=-0.123012
    [LightGBM] [Info] Start training from score -0.123012





<style>#sk-container-id-12 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-12 {
  color: var(--sklearn-color-text);
}

#sk-container-id-12 pre {
  padding: 0;
}

#sk-container-id-12 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-12 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-12 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-12 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-12 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-12 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-12 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-12 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-12 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-12 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-12 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-12 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-12 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-12 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-12 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-12 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-12 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-12 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-12 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-12 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-12 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-12 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-12 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-12 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-12 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-12 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-12 div.sk-label label.sk-toggleable__label,
#sk-container-id-12 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-12 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-12 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-12 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-12 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-12 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-12 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-12 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-12 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-12 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-12 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-12 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-12 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-12" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LGBMClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-30" type="checkbox" checked><label for="sk-estimator-id-30" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LGBMClassifier</div></div><div><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>LGBMClassifier(random_state=42)</pre></div> </div></div></div></div>



### Risk Scoring Logic


```python
y_pred_proba = model.predict_proba(X_test)[:, 1]
risk_scores = np.round(y_pred_proba * 100, 2)
```

### Creating a DataFrame showing risk score alongside true label


```python
risk_df = X_test.copy()
risk_df['true_label'] = y_test.values
risk_df['risk_score'] = risk_scores

print(risk_df[['risk_score', 'true_label']].head())
```

          risk_score  true_label
    6252       25.34           0
    4684       63.31           1
    1731        9.00           0
    4742       43.74           1
    4521       97.51           1


### Defining the `get_risk_score` function for New Data Points


```python
def get_risk_score(single_data_point):
    prob = model.predict_proba(single_data_point)[:, 1][0]
    score = round(prob * 100, 2)
    return score
```
