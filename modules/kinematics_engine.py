import numpy as np
import pandas as pd

def calculate_approach_angles(df_statcast: pd.DataFrame) -> pd.DataFrame:
    """
    Computes exact Vertical and Horizontal Approach Angles (VAA / HAA) 
    at the front edge of home plate (y = 1.417 ft) from raw tracking vectors.
    """
    y_plate = 1.417
    y0 = 50.0
    
    vy0 = df_statcast["vy0"].values
    ay = df_statcast["ay"].values
    vz0 = df_statcast["vz0"].values
    az = df_statcast["az"].values
    vx0 = df_statcast["vx0"].values
    ax = df_statcast["ax"].values
    
    # Quadratic time-of-flight solver for y-plane
    discriminant = vy0**2 - 2 * ay * (y0 - y_plate)
    t_plate = (-vy0 - np.sqrt(np.maximum(0.0, discriminant))) / ay
    
    # Terminal velocity vectors at the plate
    vy_plate = -np.sqrt(np.maximum(0.0, vy0**2 + 2 * ay * (y_plate - y0)))
    vz_plate = vz0 + az * t_plate
    vx_plate = vx0 + ax * t_plate
    
    # Angular resolution in degrees
    vaa = np.arctan(vz_plate / -vy_plate) * (180.0 / np.pi)
    haa = np.arctan(vx_plate / -vy_plate) * (180.0 / np.pi)
    
    df_statcast["vaa"] = np.round(vaa, 2)
    df_statcast["haa"] = np.round(haa, 2)
    
    if "plate_z" in df_statcast.columns:
        expected_vaa = -0.80 * df_statcast["plate_z"].values - 4.5
        df_statcast["vaa_above_expected"] = np.round(vaa - expected_vaa, 2)
        
    return df_statcast
