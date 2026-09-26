import numpy as np

def calculate_conditioned_csw(pitcher_arsenal, lineup_discipline, umpire_stretch):
    """
    Computes CSW% and maps it to true MLB K-rate using empirical linear regression.
    """
    total_pitches = 0
    total_csw_events = 0
    
    # Umpire boundary effects
    lateral_stretch_pct = umpire_stretch.get("horizontal_delta_inches", 0.0) * 0.036
    vertical_stretch_pct = umpire_stretch.get("vertical_delta_inches", 0.0) * 0.028
    ump_strike_mult = 1.0 + lateral_stretch_pct + vertical_stretch_pct
    
    lineup_o_swing = lineup_discipline["o_swing_pct"]
    lineup_z_contact = lineup_discipline["z_contact_pct"]
    
    for pitch in pitcher_arsenal:
        usage = pitch["usage_pct"]
        zone_pct = pitch["in_zone_pct"]
        raw_whiff = pitch["whiff_pct"]
        
        # In-Zone: swing-and-miss vs called strikes
        z_swing_rate = 0.68
        z_takes = zone_pct * (1.0 - z_swing_rate)
        z_called_strikes = z_takes * 0.84 * ump_strike_mult
        
        z_swings = zone_pct * z_swing_rate
        z_whiffs = z_swings * (1.0 - lineup_z_contact) * (raw_whiff / 0.24)
        
        # Out-of-Zone: chases turned to whiffs
        o_pitches = 1.0 - zone_pct
        o_chases = o_pitches * lineup_o_swing
        o_whiffs = o_chases * (raw_whiff * 1.40)
        o_called_strikes = (o_pitches - o_chases) * max(0.0, 0.035 * ump_strike_mult)
        
        pitch_csw = (z_called_strikes + z_whiffs + o_whiffs + o_called_strikes)
        total_pitches += usage
        total_csw_events += usage * pitch_csw
        
    conditioned_csw = total_csw_events / total_pitches
    
    # RECALIBRATED TRANSFER FUNCTION: K% = 1.62 * CSW% - 0.224
    # Calibrated against Statcast historical starters (2021-2026)
    projected_k_pct = np.clip(1.62 * conditioned_csw - 0.224, 0.10, 0.42)
    
    return {
        "conditioned_csw_pct": round(float(conditioned_csw), 4),
        "projected_k_rate": round(float(projected_k_pct), 4),
        "umpire_strike_multiplier": round(float(ump_strike_mult), 3)
    }
