import pandas as pd
import numpy as np

# -----------------------------
# 0. Load raw dataset
# -----------------------------
file_path = "original_dataset[old].xlsx"   # change path if needed
df = pd.read_excel(file_path)
start_rows = len(df)

# -----------------------------
# 1. Initial inspection (optional)
# -----------------------------
print("Initial shape:", df.shape)
print(df.head())

# -----------------------------
# 2. Convert datatypes
# -----------------------------
for col in ['Age','Salary','PerformanceScore']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['JoinDate'] = pd.to_datetime(
    df['JoinDate'],
    dayfirst=True,
    infer_datetime_format=True,
    errors='coerce'
)

# -----------------------------
# 3. Clean text fields
# -----------------------------
df['Name'] = df['Name'].replace('', np.nan).astype('object')
df['Name'] = df['Name'].str.strip().str.title()

df['Department'] = df['Department'].replace('', np.nan).astype('object')
df['Department'] = df['Department'].str.strip().str.title()

def clean_gender(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in ('m','male'): return 'Male'
    if s in ('f','female'): return 'Female'
    return np.nan

df['Gender'] = df['Gender'].apply(clean_gender)

# -----------------------------
# 4. Fix invalid ranges
# -----------------------------
df.loc[(df['Age'] < 15) | (df['Age'] > 100), 'Age'] = np.nan
df.loc[df['Salary'] <= 0, 'Salary'] = np.nan

# -----------------------------
# 5. Deduplicate
# -----------------------------
df = df.drop_duplicates(subset=['ID'], keep='first').reset_index(drop=True)

# -----------------------------
# 6. Outlier flags
# -----------------------------
def flag_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)

df['Salary_outlier'] = flag_outliers_iqr(df['Salary'])
df['Age_outlier'] = flag_outliers_iqr(df['Age'])

# -----------------------------
# 7. Impute missing values (fixed with transform)
# -----------------------------
df['Age'] = df['Age'].fillna(df.groupby('Department')['Age'].transform('median'))
df['Age'] = df['Age'].fillna(df['Age'].median())

df['Salary'] = df['Salary'].fillna(df.groupby('Department')['Salary'].transform('median'))
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

df['PerformanceScore'] = df['PerformanceScore'].fillna(df['PerformanceScore'].median())
df['PerformanceScore'] = df['PerformanceScore'].round().astype('Int64')

df.loc[df['Name'].isna(), 'Name'] = 'Unknown_' + df.loc[df['Name'].isna(), 'ID'].astype(str)

# -----------------------------
# 8. Derived columns
# -----------------------------
today = pd.Timestamp('today').normalize()
df['TenureDays'] = (today - df['JoinDate']).dt.days
df['TenureYears'] = (df['TenureDays'] / 365).round(2)

# -----------------------------
# 9. Final encoding
# -----------------------------
df['GenderCode'] = df['Gender'].map({'Male':1,'Female':0}).astype('Int64')
df['Department'] = df['Department'].astype('category')

# -----------------------------
# 10. Save outputs
# -----------------------------
clean_path = "cleaned_dataset.xlsx"
df.to_excel(clean_path, index=False)

summary = {
    'rows_before': [start_rows],
    'rows_after': [len(df)],
    'salary_median': [df['Salary'].median()],
    'age_median': [df['Age'].median()],
    'missing_after': [df.isna().sum().sum()]
}
pd.DataFrame(summary).T.to_csv("cleaning_summary.csv", header=False)

log = []
log.append(f"initial_rows: {start_rows}")
log.append(f"rows_after_dedup: {len(df)}")
log.append(f"missing_after: {df.isna().sum().sum()}")
pd.Series(log, name='steps').to_csv("cleaning_log.txt", index=False)

print("✅ Cleaning complete!")
print("Saved cleaned file:", clean_path)
