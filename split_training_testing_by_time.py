

def split_data_by_time(data, start_date='2050-01-01', end_date='2080-12-31'):
 
    # Filtering train and test data by the time
    train_data = data[~((data['time'] >= start_date) & (data['time'] <= end_date))]
    test_data = data[(data['time'] >= start_date) & (data['time'] <= end_date)]
    
    return train_data, test_data