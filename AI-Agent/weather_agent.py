from langchain_core.tools import tool
from typing_extensions import Annotated


@tool
def weather_agent(city_name:Annotated[str,'this is a city name']):
    ''' 
    This is an agent to inform about the weather of the given city

    Parameters:
    city_name(str):this is a city name.

    Return:
    str: The weather in the given city
    
    
    '''
    weather_text = f' Weather in {city_name} is Nice and Sunny'
    return weather_text