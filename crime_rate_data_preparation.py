import pandas as pd
import numpy as np

SP_crime_weather_data = pd.read_csv('datasets/SP_Monthly_weather_crime_data_2001_2021.csv', index_col=0)
SP_population_data = pd.read_csv('datasets/population_by_city.csv', index_col=0)

# Use the melt function to convert the DataFrame to long format
SP_population_data = pd.melt(SP_population_data, id_vars=['Cidade'], var_name='Year', value_name='Population')

# Rename the columns for clarity
SP_population_data.rename(columns={'Cidade': 'Municipality'}, inplace=True)

SP_population_data['Municipality'] = SP_population_data['Municipality'].str.strip()

annual_crime_count_data = SP_crime_weather_data.groupby(['Year', 'Region','Municipality', 'Crime Type'], as_index=False)['Monthly Crime Count'].sum()
annual_crime_count_data['Municipality'] = annual_crime_count_data ['Municipality'].str.strip()
SP_annual_crime_count_data = annual_crime_count_data.rename(columns={'Monthly Crime Count': 'Crime Count'})


def find_common_and_unique_elements(array1, array2):
    """
    Find common elements and elements unique to two arrays, and create a mapping dictionary.

    Args:
        array1 (numpy.ndarray): The first input array.
        array2 (numpy.ndarray): The second input array.

    Returns:
        tuple: A tuple containing:
               1. common_elements (numpy.ndarray): Common elements between array1 and array2.
               2. mapping_dict (dict): A dictionary mapping elements unique to array1 to elements unique to array2.
               3. unique_to_array1 (numpy.ndarray): Elements unique to array1.
               4. unique_to_array2 (numpy.ndarray): Elements unique to array2.
    """
    # Find common elements (values that exist in both arrays)
    common_elements = np.intersect1d(array1, array2)

    # Find the elements unique to each array
    unique_to_array1 = np.setdiff1d(array1, array2)
    unique_to_array2 = np.setdiff1d(array2, array1)

    # Create a mapping dictionary where keys are from unique_to_array1 and values are from unique_to_array2
    mapping_dict = {val1: val2 for val1, val2 in zip(unique_to_array1, unique_to_array2)}

    return common_elements, mapping_dict, unique_to_array1, unique_to_array2

crime_df_unique = SP_annual_crime_count_data['Municipality'].unique()
pop_df_unique = SP_population_data['Municipality'].unique()
common_elements, mapping_dict, unique_to_array1, unique_to_array2 = find_common_and_unique_elements(crime_df_unique, pop_df_unique)

for key, value in mapping_dict.items():
  SP_population_data['Municipality'] = SP_population_data['Municipality'].str.replace(value, key)

# Convert the 'Year' column in SP_population_data to integer
SP_population_data['Year'] = SP_population_data['Year'].astype('int')

# Merge crime data and population data on 'Year' and 'Municipality' using an inner join
crime_data_by_year = SP_annual_crime_count_data.merge(SP_population_data, on=['Year', 'Municipality'], how='inner')

# Calculate the 'Crime Rate' as (Crime Count / Population) * 100
crime_data_by_year['Crime Rate'] = (crime_data_by_year['Crime Count'] / crime_data_by_year['Population']) * 100

# Group the data by 'Year', 'Region', 'Municipality', and 'Population' while summing 'Crime Count'
crime_data_by_year_without_crime_type = crime_data_by_year.groupby(['Year', 'Region', 'Municipality', 'Population'], as_index=False)['Crime Count'].sum()

# Calculate the 'Crime Rate' for this aggregated data
crime_data_by_year_without_crime_type['Crime Rate'] = (crime_data_by_year_without_crime_type['Crime Count'] / crime_data_by_year_without_crime_type['Population']) * 100

crime_data_by_year_without_crime_type.to_csv('crime_data_by_year_without_crime_type.csv')
crime_data_by_year.to_csv('crime_data_by_year.csv')