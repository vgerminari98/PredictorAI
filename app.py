import os
import json
import pandas as pd
import requests_cache
import openmeteo_requests

from pprint import pprint
from retry_requests import retry
from geopy.geocoders import Nominatim
from flask import Flask, render_template, request, redirect, url_for
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    city_name = request.form.get('city')
    print(f"Received city name: {city_name}")


@app.route('/submit-city', methods=['GET'])
def submit_city():
    city_name = request.values.get('city')
    if not city_name:
        return redirect(url_for('index'))

    coordinates = obter_coordenadas(city_name)
    if not coordinates:
        return f"Coordenadas para {city_name} não encontradas.", 404

    latitude, longitude = coordinates

    try:
        weather_info = meteorological_data(latitude, longitude)
    except Exception as e:
        print(f"Erro ao obter dados meteorológicos: {e}")
        return "Erro ao obter dados meteorológicos.", 500
    
    modelo_resposta = """"
    {
        'Previsão Resumida': '', 
        'Sugestão de Roupa': 
            {
                'Peças': '', 
                'Cor Mais Apropriada': ''
            }, 
        'Recomendação de Atividade': {
            'Tipo': '', 
            'Detalhes': ''
        }
    }
    """

    prompt = f"""
    Atue como um assistente de estilo e previsão do tempo. Analise os dados meteorológicos fornecidos abaixo e gere um objeto JSON que contenha uma Previsão Resumida, Sugestão de Roupa (incluindo peças e a Cor Mais Apropriada), e uma Recomendação de Atividade (especificando se é melhor Ao Ar Livre ou Em Lugar Coberto)
    cidade de {city_name}:
    
    {weather_info}

    retorne apenas o json no seguinte formato:

    {modelo_resposta}
    """

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Aviso: GOOGLE_API_KEY não configurada. Configure via variável de ambiente.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=1.0
    )

    messages = [HumanMessage(content=prompt)]

    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"Erro ao chamar o LLM: {e}")
        return "Erro ao chamar o assistente de estilo.", 500


    try:
        if hasattr(response, "content") and response.content is not None:
            raw = response.content
        elif isinstance(response, dict) and "content" in response:
            raw = response["content"]
        elif hasattr(response, "text"):
            raw = response.text
        else:
            raw = str(response)

        raw = raw.strip() if isinstance(raw, str) else ""

        if not raw:
            print("Resposta vazia do LLM:", response)
            return "Resposta vazia do assistente.", 500

        try:
            style_data = json.loads(raw)
        except json.JSONDecodeError:
            import re
            m = re.search(r'(\{(?:.|\n)*\}|\[(?:.|\n)*\])', raw)
            if m:
                try:
                    style_data = json.loads(m.group(1))
                except Exception as e2:
                    print("Falha ao decodificar JSON extraído:", e2)
                    print("JSON extraído bruto:", m.group(1))
                    return "Erro ao processar a resposta do assistente (JSON inválido).", 500
            else:
                print("Conteúdo retornado não é JSON:", raw)
                return "Assistente retornou texto não-JSON.", 500

        print(f"""
        ==============================================
        Resposta do LLM (parseada): {style_data}
        ==============================================
        """)
    except Exception as e:
        print(f"Erro ao processar resposta do LLM: {e}")
        return "Erro ao processar a resposta do assistente de estilo.", 500

    return render_template('weather_result.html', city=city_name, data=style_data)


def obter_coordenadas(nome_cidade):
    """
    Recebe o nome de uma cidade e retorna suas coordenadas (latitude, longitude).
    """
    try:
        geolocator = Nominatim(user_agent="meu_aplicativo_geocoding")
        location = geolocator.geocode(nome_cidade)
        
        if location:
            latitude = location.latitude
            longitude = location.longitude
            
            print(f"Cidade: {nome_cidade}")
            print(f"Latitude: {latitude}")
            print(f"Longitude: {longitude}")
            
            return (latitude, longitude)
        else:
            print(f"Coordenadas não encontradas para: {nome_cidade}")
            return None
            
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None


def meteorological_data(latitude, longitude):

    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": f'{latitude}',
        "longitude": f'{longitude}',
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability", "precipitation", "rain", "showers", "snowfall", "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "visibility", "evapotranspiration", "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m", "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m", "wind_gusts_10m", "temperature_80m", "temperature_120m", "temperature_180m", "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm"],
        "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "is_day", "precipitation", "rain", "showers", "snowfall", "weather_code", "cloud_cover", "pressure_msl", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
        "timezone": "America/Sao_Paulo",
    }

    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    current = response.Current()
    current_temperature_2m = current.Variables(0).Value()
    current_apparent_temperature = current.Variables(2).Value()
    current_relative_humidity_2m = current.Variables(1).Value()
    current_precipitation = current.Variables(4).Value()
    current_weather_code = current.Variables(8).Value()
    current_wind_gusts_10m = current.Variables(14).Value()
    current_relative_humidity_2m = current.Variables(1).Value()
    current_cloud_cover = current.Variables(9).Value()


    print("\nCurrent Weather:")
    print(f"  Temperature: {current_temperature_2m} °C")
    print(f"  Apparent Temperature: {current_apparent_temperature} °C")
    print(f"  Relative Humidity: {current_relative_humidity_2m} %")
    print(f"  Precipitation: {current_precipitation} mm")
    print(f"  Weather Code: {current_weather_code}")
    print(f"  Wind Gusts at 10m: {current_wind_gusts_10m} km/h")
    print(f"  Cloud Cover: {current_cloud_cover} %")

    response_data = {
        "temperature_2m": current_temperature_2m,
        "apparent_temperature": current_apparent_temperature,
        "relative_humidity_2m": current_relative_humidity_2m,
        "precipitation": current_precipitation,
        "weather_code": current_weather_code,
        "wind_gusts_10m": current_wind_gusts_10m,
        "cloud_cover": current_cloud_cover
    }

    return response_data



if __name__ == '__main__':
    app.run(debug=True)