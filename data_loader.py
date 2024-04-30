import pandas as pd
import xarray as xr

def load_data(file_paths):
    #Empty data frame created for storing the concatenated df
    df = pd.DataFrame()

    # Iterate over each file path in the list
    for file_path in file_paths:
        # Open the dataset
        ds = xr.open_dataset(file_path)

        #Relevant variables are used to convert it to a dataframe
        df_temp = ds[["TREFMXAV_U", "FLNS", "FSNS", "PRECT", "PRSN", "QBOT", "UBOT", "VBOT"]].to_dataframe().reset_index().dropna()
        
        # Concatenate the DataFrame to the main DataFrame
        df = pd.concat([df, df_temp])

    #Index reset
    df = df.reset_index(drop=True)
    
    print(df.isnull().any())

    # Checking the lat and lon values in df
    unique_lat_values = df['lat'].unique()
    print("Unique latitude values:", unique_lat_values)
    
    unique_lon_values = df['lon'].unique()
    print("Unique longitude values:", unique_lon_values)

    # Filtering Manchester using lat and lon
    manchester = df[(df['lat'] > 53) & (df['lat'] < 54) & (df['lon'] > 357) & (df['lon'] < 358)]

    # Changing the time to datetime format
    manchester['time'] = pd.to_datetime(manchester['time'], format='%Y-%m-%d %H:%M:%S')

    # Replacing negative values with 0
    manchester.loc[manchester['PRECT'] < 0, 'PRECT'] = 0
    manchester.loc[manchester['PRSN'] < 0, 'PRSN'] = 0

    return df, manchester

