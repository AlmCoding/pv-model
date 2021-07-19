import pandas as pd
import numpy as np


def flatten(l: list):
    res = []
    for i in l:
        if isinstance(i, list):
            res += flatten(i)
        else:
            res.append(i)
    return res


if __name__ == '__main__':
    pv_inclination = 10  # 10 grad
    pv_rotation = 25  # 25 grad

    minimal_effective_sun_altitude = 1
    pv_efficiency = 0.2
    e0_space = 1361  # [W/m2]
    atmospheric_attenuation_zenith = 0.73    # ~70% direct + ~3% scatter
    atmospheric_attenuation_horizon = 0.22   # ?

    df = pd.read_csv('straubing.csv', sep=';')
    df.columns = ["date"] + flatten([[f"{i}:00 alt", f"{i}:00 rot"] for i in range(24)]) + ['none']
    del df['none']
    df = df.replace(r'--', 0.0)
    print(df.head)

    df_orig_sun_altitude = pd.concat([df['date']], axis=1, keys=['date'])       # Original csv file sun altitude
    dfn_sun_altitude = pd.concat([df['date']], axis=1, keys=['date'])           # Normalized sun altitude
    dfn_atm_attenuation = pd.concat([df['date']], axis=1, keys=['date'])        # Atmospheric attenuation

    for i in range(24):
        sun_rot_column = 2 * i + 2
        sun_rotation = pd.to_numeric(df.iloc[:, sun_rot_column])
        # Sun rotation normalized pv inclination
        effective_pv_inclination = pv_inclination * np.cos(np.deg2rad(sun_rotation - pv_rotation))

        sun_alt_column = 2 * i + 1
        sun_altitude = pd.to_numeric(df.iloc[:, sun_alt_column])
        sun_altitude[sun_altitude < 0.0] = 0.0

        df_orig_sun_altitude[df.columns[sun_alt_column]] = sun_altitude

        # Atmospheric attenuation
        atmospheric_attenuation = atmospheric_attenuation_horizon + \
                                 (atmospheric_attenuation_zenith - atmospheric_attenuation_horizon) * \
                                 np.sin(np.deg2rad(sun_altitude))
        atmospheric_attenuation[sun_altitude <= 0.0] = 0.0

        # PV inclination normalized sun altitude
        effective_sun_altitude = sun_altitude + effective_pv_inclination
        # Considers too small (zero and neg.) sun inclinations lifted into positive range by pv inclination
        effective_sun_altitude[sun_altitude <= minimal_effective_sun_altitude] = 0.0
        # Considers neg. effective sun inclinations due to pv inclination
        effective_sun_altitude[effective_sun_altitude <= minimal_effective_sun_altitude] = 0.0

        # Drop zero columns
        if sum(effective_sun_altitude) > 0.0:
            column_name = df.columns[sun_alt_column]
            dfn_sun_altitude[column_name] = effective_sun_altitude
            dfn_atm_attenuation[column_name] = atmospheric_attenuation

    # Load sun duration averages
    df_hours = pd.read_csv('sunnydays_straubing.csv', sep='\t', decimal=",")

    # Compute monthly average over the years
    df_hours['Year/Month'] = df_hours['Year/Month'].map(lambda d: d.split('/')[-1])
    df_hours = df_hours.rename(columns={'Year/Month': 'Month'})
    df_hours = df_hours.apply(pd.to_numeric)

    df_mean_hours = pd.DataFrame(columns=df_hours.columns)
    months = set(df_hours.iloc[:, 0])
    for month in months:
        total = df_hours.loc[df_hours['Month'] == month]
        total = total.sum() / total.shape[0]
        df_mean_hours = df_mean_hours.append(total, ignore_index=True)

    # Create sun map

    for i in range(dfn_sun_altitude.shape[0]):
        day = dfn_sun_altitude.iloc[i, :]
        date = day[0]
        inclinations = day[1:]
        a = 12
