import pandas as pd

df = pd.read_parquet("aml-for-cl-project\data\arqmath_questions.parquet")
print(df.shape)
print(df.dtypes)
df.head()