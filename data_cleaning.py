# Import libraries
import pandas as pd
from pathlib import Path

# Path to the folder containing gage data
folder = Path(r"C:\Users\jxjh1\Downloads\IA1\gage")

# Get all CSV files
files1 = sorted(folder.glob("primary-*.csv"))

# Get gage data from each file
gage_data = []

for file in files1:
    df = pd.read_csv(file)
    gage_data.append(df)

# Combine them
gage = pd.concat(gage_data, ignore_index=True)

# Keep needed columns
gage = gage[["time", "value"]]

# Rename columns
gage = gage.rename(columns={
    "time": "date",
    "value": "gage height (ft)"
})

# Clean and prepare
gage["date"] = pd.to_datetime(gage["date"], utc=True) #change format
gage = gage.sort_values("date") #sort by date
gage = gage.drop_duplicates() #drop duplicates if any
gage = gage.reset_index(drop=True) #fixes any gap that may have been created by removing duplicates

# Save the output
gage.to_csv(folder / "gage_combined.csv", index=False)

#====

# Path to the folder containing discharge files
folder = Path(r"C:\Users\jxjh1\Downloads\IA1\discharge")

# Get all CSV files
files2 = sorted(folder.glob("primary*.csv"))

# Get Discharge data from each file
discharge_data = []

for file in files2:
    df2 = pd.read_csv(file)
    discharge_data.append(df2)

# Combine them
discharge = pd.concat(discharge_data, ignore_index=True)

# Keep needed columns
discharge = discharge[["time", "value"]]

# Rename columns
discharge = discharge.rename(columns={
    "time": "date",
    "value": "discharge (ft^3/s)"
})

# Clean and prepare
discharge["date"] = pd.to_datetime(discharge["date"], utc=True)
discharge = discharge.sort_values("date")
discharge = discharge.drop_duplicates()
discharge = discharge.reset_index(drop=True)

# Save
discharge.to_csv(folder / "discharge.csv", index=False)

#========

#  Path to the folder containing precipitation files
folder = Path(r"C:\Users\jxjh1\Downloads\IA1\precipitation")

# Get all CSV files
files3 = sorted(folder.glob("primary-*.csv"))

# Get Precipitation data from each file
precipitation_data = []

for file in files3:
    df3 = pd.read_csv(file)
    precipitation_data.append(df3)

# Combine them
precipitation = pd.concat(precipitation_data, ignore_index=True)

# Keep needed columns
precipitation = precipitation[["time", "value"]]

# Rename columns
precipitation = precipitation.rename(columns={
    "time": "date",
    "value": "precipitation (in)"
})

# Clean and prepare
precipitation["date"] = pd.to_datetime(precipitation["date"], utc=True)
precipitation = precipitation.sort_values("date")
precipitation = precipitation.drop_duplicates()
precipitation = precipitation.reset_index(drop=True)

# Save
precipitation.to_csv(folder / "precipitation_combined.csv", index=False)

#===
# Check the format and the size of each dataframe
print(gage.dtypes)
print(gage.shape)
print(discharge.dtypes)
print(discharge.shape)
print(precipitation.dtypes)
print(precipitation.shape)

# Combine dataframes
combined = (
    gage
    .merge(discharge, on="date", how="inner")   #inner join guarantees that only the exact dates from all three data frames will be combined
    .merge(precipitation, on="date", how="inner")
)

# Save
output_folder = Path(r"C:\Users\jxjh1\Downloads\IA1")
combined.to_csv(output_folder / "combined.csv", index=False)


# Sort by date
combined = combined.sort_values("date")
combined = combined.reset_index(drop=True)

# Check the size of the combined data frame
print(combined.shape)



