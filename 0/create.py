import pandas as pd
test_data_config = ['data/1/A.csv','data/1/B.csv','data/1/C.csv','data/1/D.csv','data/1/E.csv']
for i in test_data_config:
    df = pd.read_csv(i)
    df1 = df.head(20)
    df1.to_csv(f'test_{i}', index=False, encoding='utf-8-sig')
