import pandas as pd
import numpy as np

sales = {'account': ['Jones LLC', 'Alpha Co', 'Blue Inc'],
         'Jan': [150, 200, 50],
         'Feb': [None, 210, None],
         'Mar': [140, 215, 95]}
df = pd.DataFrame.from_dict(sales)
feb = df['Feb']
for i in range(len(feb)):
	if np.isnan(feb[i]):
		print(feb[i])
		df.at[i, 'Feb'] = 99999
print(df)
