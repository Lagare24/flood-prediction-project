import pandas as pd
import numpy as np

def calculate_flood_status(row):
    # Initialize score
    score = 0
    
    # Rainfall scoring (mm/hr)
    if row['Rainfall'] > 30: score += 5  # Very High
    elif 15 <= row['Rainfall'] <= 30: score += 4  # High
    elif 7.5 <= row['Rainfall'] < 15: score += 3  # Medium
    elif 2.5 <= row['Rainfall'] < 7.5: score += 2  # Low
    else: score += 1  # Very Low
    
    # Water Level scoring (meters)
    if row['Water Level'] > 4500: score += 5  # Very High
    elif 4300 <= row['Water Level'] <= 4400: score += 4  # High
    elif 4200 <= row['Water Level'] < 4300: score += 3  # Medium
    elif 4000 <= row['Water Level'] < 4200: score += 2  # Low
    else: score += 1  # Very Low
    
    # Elevation scoring (meters)
    if 0 <= row['Elevation'] <= 5: score += 5  # Very High
    elif 6 <= row['Elevation'] <= 20: score += 4  # High
    elif 21 <= row['Elevation'] <= 50: score += 3  # Medium
    elif 51 <= row['Elevation'] <= 150: score += 2  # Low
    else: score += 1  # Very Low
    
    # Slope scoring (degrees)
    if 0 <= row['Slope'] <= 3: score += 5  # Very High
    elif 3 < row['Slope'] <= 8: score += 4  # High
    elif 8 < row['Slope'] <= 18: score += 3  # Medium
    elif 18 < row['Slope'] <= 30: score += 2  # Low
    else: score += 1  # Very Low
    
    # Distance from River scoring (meters)
    if 0 <= row['Distance from River'] <= 100: score += 5  # Very High
    elif 100 < row['Distance from River'] <= 200: score += 4  # High
    elif 200 < row['Distance from River'] <= 300: score += 3  # Medium
    elif 300 < row['Distance from River'] <= 400: score += 2  # Low
    else: score += 1  # Very Low
    
    # Calculate final flood status based on average score
    avg_score = score / 5  # 5 parameters
    
    if avg_score >= 4: return 2  # High flood risk
    elif avg_score >= 3: return 1  # Medium flood risk
    else: return 0  # Low flood risk

# Read the CSV file
df = pd.read_csv('data/THESIS - GIS DATA - FLOOD SCENARIOS.csv')

# Calculate flood status for each row
df['Flood Status'] = df.apply(calculate_flood_status, axis=1)

# Save the updated CSV file
df.to_csv('data/THESIS - GIS DATA - FLOOD SCENARIOS.csv', index=False)

print("Flood status values have been updated in the CSV file.")
print("\nSummary of flood status values:")
print(df['Flood Status'].value_counts().sort_index())
