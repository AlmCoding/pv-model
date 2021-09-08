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


class Converter:
    def __init__(self, df_sunny_hours: pd.DataFrame):
        self.df_sunny_hours = df_sunny_hours
        self.fill_order = [13, 12, 14, 11, 15, 10, 16, 9, 17, 8]

    """
    def get_month(self, ds: pd.Series):
        return int(ds[0].split("-")[1])

    def sunny_day_hours(self, month: int):
        return self.df_sunny_hours.iloc[month-1]["SunDuration"]
    """

    def convert(self, ds: pd.Series):
        month = int(ds[0].split("-")[1])
        daly_sun_hours = self.df_sunny_hours.iloc[month-1]["SunDuration"] / 30

        ds_sun = ds.copy()
        ds_sun[1:] = 0.0

        for i in self.fill_order:
            idx = i + 1
            if daly_sun_hours >= 1.0:
                ds_sun[idx] = 1.0
                daly_sun_hours -= 1
            elif daly_sun_hours > 0.0:
                ds_sun[idx] = daly_sun_hours
                break

        return ds_sun


if __name__ == '__main__':
    # pv_area = 30    # m2
    pv_inclination = 10  # 10 grad
    pv_rotation = 25  # 25 grad

    minimal_effective_sun_altitude = 1
    pv_efficiency = 0.2
    e0_space = 1.361  # [kW/m2]
    atmospheric_attenuation_zenith = 0.73    # ~70% direct + ~3% scatter
    atmospheric_attenuation_horizon = 0.20   # ?

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
        # if sum(effective_sun_altitude) > 0.0:
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

    # Create weather attenuation
    conv = Converter(df_mean_hours)
    dfn_weather_attenuation = dfn_atm_attenuation.apply(conv.convert, axis=1)

    dfn_total_attenuation = dfn_atm_attenuation.iloc[:, 1:] * dfn_weather_attenuation.iloc[:, 1:]

    dfn_energy = e0_space * dfn_total_attenuation * np.sin(np.deg2rad(dfn_sun_altitude.iloc[:, 1:])) * pv_efficiency
    dfn_energy.insert(0, "date", dfn_atm_attenuation.iloc[:, 0])

    dfn_energy_days = dfn_energy.iloc[:, 1:].sum(axis=1)
    dfn_energy_days = pd.DataFrame({"date": dfn_atm_attenuation.iloc[:, 0], "energy": dfn_energy_days})
    total_energy = dfn_energy_days.iloc[:, 1:].sum()

    energy_months = {}
    months = ["Jan", "Feb", "Mar", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    for index, row in dfn_energy_days.iterrows():
        month = int(row["date"].split("-")[1]) - 1
        if months[month] in energy_months.keys():
            energy_months[months[month]][0] += row["energy"]
        else:
            energy_months[months[month]] = [row["energy"]]
    dfn_energy_months = pd.DataFrame(energy_months)

    import seaborn as sns
    import matplotlib.pyplot as plt
    # sns.set(style="whitegrid", color_codes=True)

    pal = sns.color_palette("ch:start=.2,rot=-.3", 12)
    rank = np.argsort(-dfn_energy_months.iloc[0, :]).argsort()  # http://stackoverflow.com/a/6266510/1628638
    p = sns.barplot(x=dfn_energy_months.columns, y=dfn_energy_months.iloc[0, :], palette=np.array(pal[::-1])[rank])
    p.set_title("Energy/m2")
    p.set_ylabel("Energy [kWh]")
    plt.show()

    a = 12
