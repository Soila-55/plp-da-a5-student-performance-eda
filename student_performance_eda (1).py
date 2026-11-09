import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Ensure directories exist
os.makedirs('figures', exist_ok=True)

# 2. Generate Student Performance Dataset (Seed 55)
np.random.seed(55)
n_students = 1200

counties = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Kiambu']
school_types = ['Public', 'Private']
genders = ['Male', 'Female']

data = {
    'student_id': [f'STU_{i:04d}' for i in range(1, n_students + 1)],
    'gender': np.random.choice(genders, size=n_students),
    'school_type': np.random.choice(school_types, size=n_students, p=[0.6, 0.4]),
    'county': np.random.choice(counties, size=n_students),
    'attendance_rate': np.random.uniform(50, 100, size=n_students),
    'study_hours': np.random.gamma(shape=3, scale=4, size=n_students),
    'math_score': np.random.normal(65, 12, size=n_students),
    'english_score': np.random.normal(70, 10, size=n_students),
    'science_score': np.random.normal(62, 15, size=n_students)
}

df = pd.DataFrame(data)

# Ensure scores remain within realistic bounds (0 to 100)
for col in ['math_score', 'english_score', 'science_score']:
    df[col] = np.clip(df[col], 0, 100).round(1)
df['attendance_rate'] = df['attendance_rate'].round(1)
df['study_hours'] = df['study_hours'].round(1)

# Save dataset
df.to_csv('student_performance.csv', index=False)
print("Saved dataset to 'student_performance.csv'\\n")

# SECTION 1: Structure
print("--- SECTION 1: STRUCTURE ---")
print(f"Dataset Shape: {df.shape}")
print("Grain of Data: One row represents an individual student's performance metrics.")
print("\\nVariables and Types:")
print(df.dtypes)
print("\\nMissing values count per column:")
print(df.isnull().sum())
print("\\nData types confirmation: Structure checks out perfectly without null values.")

# SECTION 2: Univariate
print("\\n--- SECTION 2: UNIVARIATE ---")
print("\\nSummary of Numerical Variables:")
print(df[['attendance_rate', 'study_hours', 'math_score', 'english_score', 'science_score']].describe())

print("\\nSummary of Categorical Variables (Gender):")
print(df['gender'].value_counts())
print(df['gender'].value_counts(normalize=True) * 100)

print("\\nSummary of Categorical Variables (School Type):")
print(df['school_type'].value_counts())
print(df['school_type'].value_counts(normalize=True) * 100)

print("\\nSummary of Categorical Variables (County):")
print(df['county'].value_counts())
print(df['county'].value_counts(normalize=True) * 100)

# Outliers in study_hours (using standard IQR method)
q1 = df['study_hours'].quantile(0.25)
q3 = df['study_hours'].quantile(0.75)
iqr = q3 - q1
upper_limit = q3 + 1.5 * iqr
outliers = df[df['study_hours'] > upper_limit]
print(f"\\nStudy Hours IQR: {iqr:.2f}, Upper Limit for Outliers: {upper_limit:.2f}")
print(f"Number of outliers in study_hours: {len(outliers)}")

# Create average_score and performance_band
df['average_score'] = df[['math_score', 'english_score', 'science_score']].mean(axis=1).round(1)
bins = [0, 59.9, 74.9, 100]
labels = ['At Risk', 'Meeting', 'Exceeding']
df['performance_band'] = pd.cut(df['average_score'], bins=bins, labels=labels, include_lowest=True)

print("\\nPerformance Band Distribution:")
print(df['performance_band'].value_counts())
print(df['performance_band'].value_counts(normalize=True) * 100)

# Chart 1: Distribution of Average Scores
plt.figure(figsize=(8, 5))
df['average_score'].plot(kind='hist', bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Average Student Scores')
plt.xlabel('Average Score')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figures/distribution_average_score.png')
plt.close()

# SECTION 3: Bivariate
print("\\n--- SECTION 3: BIVARIATE ---")
numerical_cols = ['attendance_rate', 'study_hours', 'math_score', 'english_score', 'science_score', 'average_score']
correlation_matrix = df[numerical_cols].corr()
print("\\nCorrelation Matrix:")
print(correlation_matrix.round(3))

print("\\nAverage Score by Gender:")
print(df.groupby('gender')['average_score'].mean().round(2))
print("\\nAverage Score by School Type:")
print(df.groupby('school_type')['average_score'].mean().round(2))
print("\\nAverage Score by County:")
print(df.groupby('county')['average_score'].mean().round(2))

county_band = pd.crosstab(df['county'], df['performance_band'], normalize='index') * 100
print("\\nPerformance Band Share by County (%):")
print(county_band.round(2))
highest_at_risk_county = county_band['At Risk'].idxmax()
print(f"\\nCounty with the highest share of At Risk learners: {highest_at_risk_county} ({county_band.loc[highest_at_risk_county, 'At Risk']:.2f}%)")

# Chart 2: Average Score by County
plt.figure(figsize=(8, 5))
df.groupby('county')['average_score'].mean().sort_values().plot(kind='bar', color='coral', edgecolor='black')
plt.title('Average Student Score across Counties')
plt.ylabel('Mean Average Score')
plt.xlabel('County')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figures/average_score_by_county.png')
plt.close()

# SECTION 4: Multivariate
print("\\n--- SECTION 4: MULTIVARIATE ---")
df['attendance_band'] = pd.cut(df['attendance_rate'], bins=[0, 75, 90, 100], labels=['Low', 'Medium', 'High'])
df['study_hours_quartile'] = pd.qcut(df['study_hours'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

print("\\nPivot Table 1: School Type Gap across Attendance Bands")
pivot_attendance = df.pivot_table(values='average_score', index='attendance_band', columns='school_type', aggfunc='mean')
pivot_attendance['gap'] = pivot_attendance['Private'] - pivot_attendance['Public']
print(pivot_attendance.round(2))

print("\\nPivot Table 2: School Type Gap across Study Hour Quartiles")
pivot_study = df.pivot_table(values='average_score', index='study_hours_quartile', columns='school_type', aggfunc='mean')
pivot_study['gap'] = pivot_study['Private'] - pivot_study['Public']
print(pivot_study.round(2))

# Chart 3: Average Scores by Performance Band (Box Plot)
plt.figure(figsize=(8, 5))
df.boxplot(column='average_score', by='performance_band', grid=True)
plt.title('Distribution of Average Scores across Performance Bands')
plt.suptitle('')
plt.ylabel('Average Score')
plt.tight_layout()
plt.savefig('figures/average_score_boxplot.png')
plt.close()

# SECTION 5: Trends
print("\\n--- SECTION 5: TRENDS ---")
slope, intercept = np.polyfit(df['study_hours'], df['average_score'], 1)
y_pred = slope * df['study_hours'] + intercept
y_mean = np.mean(df['average_score'])
ss_res = np.sum((df['average_score'] - y_pred) ** 2)
ss_tot = np.sum((df['average_score'] - y_mean) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"\\nTrend line parameters: slope = {slope:.4f}, R-squared (R²) = {r_squared:.4f}")
print(f"\\nInterpretation: A slope of {slope:.4f} indicates that for each additional hour a student studies, their average score changes on average by {slope:.4f} marks.")
print("Limitation: R² indicates that study hours explain only a small fraction of score variation. Other factors are not included.")

# FINAL REPORT OUTLINE (corrected to match generated data)
print("\\n" + "="*60)
print("FINDINGS")
print(f"1. The overall mean average score is {df['average_score'].mean():.2f} marks.")
print(f"2. {(df['performance_band'] == 'At Risk').mean()*100:.2f}% of learners are classified as At Risk.")
print(f"3. Attendance and average score correlation is {df['attendance_rate'].corr(df['average_score']):.3f}, indicating a very weak linear relationship.")
print(f"4. Study hours and average score correlation is {df['study_hours'].corr(df['average_score']):.3f}, indicating a very weak linear relationship.")
print(f"5. {highest_at_risk_county} County has the highest proportion of At Risk learners ({county_band.loc[highest_at_risk_county, 'At Risk']:.2f}%).")
print(f"6. Study hours explain approximately {r_squared*100:.3f}% of variation in average scores in this linear model.")

print("\\nRECOMMENDATIONS")
print("1. Investigate additional variables that may influence student performance.")
print("2. Prioritize learner support in counties with higher At Risk proportions.")
print("3. Avoid causal conclusions from correlations in this synthetic dataset.")

print("\\nLIMITATIONS")
print("This analysis uses synthetic data with independently generated variables.")
print("It does not intentionally model relationships between attendance, study hours, and achievement.")
print("Findings cannot be generalized to actual Kenyan students or establish causation.")
print("="*60)
