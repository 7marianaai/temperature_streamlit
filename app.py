from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


weather_url = 'https://api.openweathermap.org/data/2.5/weather'

month_to_season = {12: 'winter', 1: 'winter', 2: 'winter',
                   3: 'spring', 4: 'spring', 5: 'spring',
                   6: 'summer', 7: 'summer', 8: 'summer',
                   9: 'autumn', 10: 'autumn', 11: 'autumn'}

season_order = ['winter', 'spring', 'summer', 'autumn']
season_names = {
    'winter': 'Зима',
    'spring': 'Весна',
    'summer': 'Лето',
    'autumn': 'Осень',
}


@st.cache_data
def data_load(file):
    data = pd.read_csv(file)
    columns = ['city', 'timestamp', 'temperature', 'season']

    if not set(columns).issubset(data.columns):
        raise ValueError('В CSV нужны столбцы city, timestamp, temperature, season.')

    data = data[columns].copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data['temperature'] = pd.to_numeric(data['temperature'], errors='coerce')
    data['city'] = data['city'].astype('string').str.strip()
    data['season'] = data['season'].astype('string').str.strip()

    if data.empty or data.isna().any().any():
        raise ValueError('Файл пуст или содержит пропуски, неверные даты или температуры.')
    if (data['city'] == '').any() or not np.isfinite(data['temperature']).all():
        raise ValueError('Названия городов не должны быть пустыми, температуры — бесконечными.')
    if not data['season'].isin(season_order).all():
        raise ValueError('Допустимые сезоны: winter, spring, summer, autumn.')
    if (data['season'] != data['timestamp'].dt.month.map(month_to_season)).any():
        raise ValueError('Сезоны должны соответствовать месяцам, как в данных задания.')

    data['timestamp'] = data['timestamp'].dt.normalize()
    data = data.sort_values(['city', 'timestamp']).reset_index(drop=True)
    if data.duplicated(['city', 'timestamp']).any():
        raise ValueError('Для одного города должна быть только одна температура за день.')

    day_step = data.groupby('city')['timestamp'].diff().dropna()
    if (day_step != pd.Timedelta(days=1)).any():
        raise ValueError('В каждом городе даты должны идти с шагом в один день без пропусков.')

    return data


def roll_stat_get(city_data):
    res = city_data.sort_values('timestamp').copy()
    res['rolling_mean'] = res['temperature'].rolling(window=30).mean()
    res['rolling_std'] = res['temperature'].rolling(window=30).std()
    return res


def season_stat_get(data):
    res = data.groupby(['city', 'season'])['temperature'].agg(['mean', 'std'])
    res = res.reset_index()
    res = res.rename(columns={'mean': 'season_mean', 'std': 'season_std'})
    return res


def season_anom_add(data, stats):
    res = data.merge(stats, on=['city', 'season'])
    res['season_lower'] = res['season_mean'] - 2 * res['season_std']
    res['season_upper'] = res['season_mean'] + 2 * res['season_std']
    res['season_anomaly'] = (
        (res['temperature'] < res['season_lower'])
        | (res['temperature'] > res['season_upper'])
    )
    return res


def roll_anom_add(data):
    res = data.copy()
    res['rolling_lower'] = res['rolling_mean'] - 2 * res['rolling_std']
    res['rolling_upper'] = res['rolling_mean'] + 2 * res['rolling_std']
    res['rolling_anomaly'] = (
        (res['temperature'] < res['rolling_lower'])
        | (res['temperature'] > res['rolling_upper'])
    )
    return res


def analyze_city(city_data):
    res = roll_stat_get(city_data)
    stats = season_stat_get(city_data)
    res = season_anom_add(res, stats)
    res = roll_anom_add(res)
    return res


def time_series_plot(city_data, sel_city, anomaly_column, anomaly_name):
    anomalies = city_data[city_data[anomaly_column]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=city_data['timestamp'],
            y=city_data['temperature'],
            mode='lines',
            name='Температура',
            line=dict(color='royalblue', width=1),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=city_data['timestamp'],
            y=city_data['rolling_mean'],
            mode='lines',
            name='Скользящее среднее за 30 дней',
            line=dict(color='orange', width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=anomalies['timestamp'],
            y=anomalies['temperature'],
            mode='markers',
            name=anomaly_name,
            marker=dict(color='red', size=7),
        )
    )

    if anomaly_column == 'season_anomaly':
        lower_column, upper_column = 'season_lower', 'season_upper'
    else:
        lower_column, upper_column = 'rolling_lower', 'rolling_upper'

    for column, name in [(lower_column, 'Нижняя граница нормы'),
                         (upper_column, 'Верхняя граница нормы')]:
        fig.add_trace(go.Scatter(
            x=city_data['timestamp'],
            y=city_data[column],
            mode='lines',
            name=name,
            line=dict(color='gray', width=1, dash='dot'),
        ))

    fig.update_layout(
        title=f'Временной ряд температуры: {sel_city}',
        xaxis_title='Дата',
        yaxis_title='Температура, °C',
        hovermode='x unified',
    )

    return fig


def season_profile_plot(season_stats, sel_city):
    profile = season_stats.set_index('season').reindex(season_order)

    fig = go.Figure(
        go.Bar(
            x=[season_names[season] for season in season_order],
            y=profile['season_mean'],
            error_y=dict(type='data', array=profile['season_std']),
            name='Средняя температура',
            marker_color='seagreen',
        )
    )

    fig.update_layout(
        title=f'Сезонный профиль: {sel_city}',
        xaxis_title='Сезон',
        yaxis_title='Средняя температура, °C',
    )

    return fig


def trend_plot(city_data, sel_city):
    trend_data = city_data.copy()
    trend_data['year'] = trend_data['timestamp'].dt.year
    annual_temp = trend_data.groupby('year')['temperature'].mean().reset_index()
    annual_temp = annual_temp.rename(columns={'temperature': 'year_mean'})

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=annual_temp['year'],
            y=annual_temp['year_mean'],
            mode='lines+markers',
            name='Средняя температура за год',
        )
    )

    if len(annual_temp) >= 2:
        slope, intercept = np.polyfit(
            annual_temp['year'], annual_temp['year_mean'], 1
        )
        annual_temp['trend'] = slope * annual_temp['year'] + intercept

        fig.add_trace(
            go.Scatter(
                x=annual_temp['year'],
                y=annual_temp['trend'],
                mode='lines',
                name=f'Линия тренда ({slope:.3f} °C в год)',
            )
        )

    fig.update_layout(
        title=f'Долгосрочный тренд температуры: {sel_city}',
        xaxis_title='Год',
        yaxis_title='Средняя температура, °C',
        hovermode='x unified',
    )

    return fig


@st.cache_data(ttl=300, show_spinner=False)
def weather_sync_get(city, api_key):
    params = {'q': city, 'appid': api_key, 'units': 'metric'}

    try:
        response = requests.get(weather_url, params=params, timeout=10)
        weather = response.json()
    except requests.RequestException:
        return None, {'message': 'Не удалось выполнить запрос. Проверьте соединение и повторите.'}
    except ValueError:
        return None, {'message': 'API вернул ответ не в формате JSON.'}

    if response.status_code != 200:
        return None, weather

    local_time = pd.to_datetime(weather['dt'] + weather['timezone'], unit='s')
    return {
        'city': city,
        'local_time': local_time,
        'current_temp': weather['main']['temp'],
    }, None


def season_get(timestamp):
    return month_to_season[timestamp.month]


def current_temp_compare(weather, stats):
    city = weather['city']
    season = season_get(weather['local_time'])
    city_stats = stats[(stats['city'] == city) & (stats['season'] == season)]

    if city_stats.empty:
        raise ValueError('В исторических данных нет текущего сезона для выбранного города.')

    city_stat = city_stats.iloc[0]
    if pd.isna(city_stat['season_std']):
        raise ValueError('Для оценки сезонной нормы нужно хотя бы два наблюдения за сезон.')

    lower = city_stat['season_mean'] - 2 * city_stat['season_std']
    upper = city_stat['season_mean'] + 2 * city_stat['season_std']
    normal = lower <= weather['current_temp'] <= upper

    return {
        'city': city,
        'local_time': weather['local_time'],
        'season': season,
        'current_temp': weather['current_temp'],
        'season_mean': city_stat['season_mean'],
        'season_lower': lower,
        'season_upper': upper,
        'status': 'нормальная' if normal else 'аномальная',
    }


def main():
    st.set_page_config(page_title='Анализ температуры', layout='wide')
    st.title('Анализ температурных данных')
    st.write('Исторические температуры и сравнение текущей погоды с сезонной нормой.')

    st.sidebar.header('Настройки')
    uploaded_file = st.sidebar.file_uploader('Загрузите CSV с историческими данными', type=['csv'])

    try:
        if uploaded_file is not None:
            temp_data = data_load(uploaded_file)
        else:
            file = Path(__file__).with_name('temperature_data.csv')
            if not file.exists():
                st.info('Загрузите CSV, чтобы начать анализ.')
                return
            temp_data = data_load(file)
            st.sidebar.info('Используется temperature_data.csv из проекта.')
    except (ValueError, OSError) as error:
        st.error(str(error))
        return

    cities = sorted(temp_data['city'].unique())
    sel_city = st.sidebar.selectbox('Выберите город', cities)
    city_data = temp_data[temp_data['city'] == sel_city].copy()
    analysis_res = analyze_city(city_data)
    analysis_res = analysis_res.sort_values('timestamp').reset_index(drop=True)
    season_statistics = season_stat_get(city_data)

    st.header(f'Исторические данные: {sel_city}')
    first_date = city_data['timestamp'].min().date()
    last_date = city_data['timestamp'].max().date()
    date_column, rows_column = st.columns(2)
    start_date = date_column.date_input(
        'Начальная дата таблицы', value=first_date,
        min_value=first_date, max_value=last_date, format='DD.MM.YYYY',
    )
    if start_date is None:
        start_date = first_date

    preview_data = city_data[city_data['timestamp'] >= pd.Timestamp(start_date)]
    max_rows = min(30, len(preview_data))
    row_count = rows_column.number_input(
        'Количество строк в таблице', min_value=1, max_value=max_rows,
        value=min(5, max_rows), step=1,
    )
    preview_data = preview_data.head(row_count).copy()
    preview_data['timestamp'] = preview_data['timestamp'].dt.strftime('%d.%m.%Y')
    st.dataframe(preview_data, hide_index=True)
    st.subheader('Описательная статистика')
    st.dataframe(city_data['temperature'].describe().round(2).to_frame('Температура, °C'))

    col1, col2 = st.columns(2)
    col1.metric('Сезонных аномалий', int(analysis_res['season_anomaly'].sum()))
    col2.metric('Аномалий по скользящему окну', int(analysis_res['rolling_anomaly'].sum()))

    st.subheader('Временной ряд и аномалии')
    anomaly_method = st.selectbox(
        'Какие аномалии показать?',
        ['Аномалии по скользящему окну', 'Сезонные аномалии'],
    )
    if anomaly_method == 'Сезонные аномалии':
        anomaly_column = 'season_anomaly'
    else:
        anomaly_column = 'rolling_anomaly'

    st.write('Аномалия — температура за пределами среднего ± 2 стандартных отклонения.')
    st.write('Для скользящего окна используются текущий и 29 предыдущих дней. '
             'В первые 29 дней этот показатель не определяется.')
    st.plotly_chart(time_series_plot(analysis_res, sel_city, anomaly_column, anomaly_method))

    if st.checkbox('Показать найденные аномалии'):
        columns = ['timestamp', 'temperature', 'season']
        anomalies = analysis_res.loc[analysis_res[anomaly_column], columns]
        st.dataframe(anomalies.round(2), hide_index=True)

    st.subheader('Сезонный профиль')
    st.write('На графике показаны среднее и ± 1 стандартное отклонение. '
             'Для проверки аномалий используется диапазон ± 2 стандартных отклонения.')
    st.plotly_chart(season_profile_plot(season_statistics, sel_city))
    season_table = season_statistics.set_index('season').reindex(season_order)
    season_table = season_table[['season_mean', 'season_std']]
    season_table.index = [season_names[season] for season in season_order]
    season_table = season_table.rename(columns={
        'season_mean': 'Средняя температура, °C',
        'season_std': 'Стандартное отклонение, °C',
    })
    st.dataframe(season_table.round(2))

    st.subheader('Долгосрочный тренд')
    st.plotly_chart(trend_plot(analysis_res, sel_city))
    st.write('Годовые средние рассчитаны по доступным дням каждого года. '
             'Для линии тренда нужны данные минимум за два года.')

    st.header('Текущая температура')
    try:
        api_key = st.secrets.get('OPENWEATHER_API_KEY', '').strip()
    except FileNotFoundError:
        api_key = ''

    if api_key:
        if st.button('Обновить текущую температуру'):
            weather_sync_get.clear(sel_city, api_key)
        st.caption('Погода загружается автоматически. Ответ сохраняется на 5 минут; '
                   'кнопка позволяет запросить свежие данные сразу.')
    else:
        with st.form('weather_form'):
            api_key = st.text_input('API-ключ OpenWeatherMap', type='password')
            weather_button = st.form_submit_button('Получить текущую температуру')

        api_key = api_key.strip()
        if not api_key:
            st.info('Введите API-ключ, чтобы увидеть текущую температуру.')
            return
        if not weather_button:
            return
        weather_sync_get.clear(sel_city, api_key)

    weather, error = weather_sync_get(sel_city, api_key)
    if error is not None:
        st.error('Не удалось получить текущую погоду:')
        st.json(error)
        return

    st.metric(f'Текущая температура: {sel_city}', f"{weather['current_temp']:.2f} °C")
    st.write(f"Местное время данных о погоде: {weather['local_time']:%d.%m.%Y %H:%M}")
    try:
        comparison = current_temp_compare(weather, season_statistics)
    except ValueError as error:
        st.warning(str(error))
        return

    season = season_names[comparison['season']]
    st.write(f"Текущий сезон: {season}. Средняя температура сезона: {comparison['season_mean']:.2f} °C.")
    st.write(f"Нормальный диапазон: {comparison['season_lower']:.2f} … {comparison['season_upper']:.2f} °C.")
    if comparison['status'] == 'нормальная':
        st.success('Текущая температура в пределах сезонной нормы.')
    else:
        st.warning('Текущая температура аномальная: выходит за пределы сезонной нормы.')
    st.write('Сравнение выполнено с историческими среднесуточными температурами. '
             'Сезоны определены по календарю из задания для всех городов.')


if __name__ == '__main__':
    main()
