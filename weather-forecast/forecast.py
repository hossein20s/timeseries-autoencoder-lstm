import weather


class Forecast:
    def __init__(self):
        weather.maybe_download_and_extract()
        df = weather.load_resampled_data()
        self.df_sampled = df.copy()
        df.drop(('Esbjerg', 'Pressure'), axis=1, inplace=True)
        df.drop(('Roskilde', 'Pressure'), axis=1, inplace=True)
        self.df_filtered = df.copy()
        df['Various', 'Day'] = df.index.dayofyear
        df['Various', 'Hour'] = df.index.hour
        self.df = df.copy()
