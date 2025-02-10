import numpy as np  

# Replace 0 values in important columns (if applicable)
cols_to_replace = ['Column1', 'Column2']  # Replace with actual column names
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)

# Fill missing values with median
df.fillna(df.median(), inplace=True)
