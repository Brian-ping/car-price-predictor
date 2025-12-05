import pandas as pd

try:
    # Try to read just the first 5 rows
    df = pd.read_csv('data/raw/vehicles.csv', nrows=5)
    print('✅ Successfully loaded vehicles.csv')
    print(f'Shape: {df.shape}')
    print(f'\n📋 Column names:')
    for i, col in enumerate(df.columns, 1):
        print(f'  {i:2}. {col}')
    
    print(f'\n📊 First 3 rows:')
    print(df.head(3).to_string())
    
    # Check for price column
    price_cols = [col for col in df.columns if 'price' in col.lower()]
    if price_cols:
        print(f'\n💰 Price column(s) found: {price_cols}')
    else:
        print(f'\n⚠️ No price column found. Available columns with numeric data:')
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            print(f'  - {col}')
            
except Exception as e:
    print(f'❌ Error: {e}')
    print('\nTrying with different encoding...')
    try:
        df = pd.read_csv('data/raw/vehicles.csv', nrows=5, encoding='latin-1')
        print('✅ Successfully loaded with latin-1 encoding')
        print(f'Columns: {list(df.columns)}')
    except Exception as e2:
        print(f'❌ Still error: {e2}')