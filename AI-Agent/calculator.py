from langchain_core.tools import tool
from typing_extensions import Annotated
import re


@tool
def calculator_agent(num1:Annotated[int,'it is integer'], num2:Annotated[int,'it is integer']):
    ''' 
    This is a calculator agent.

    Parameters:
    - inputs (int): two integers `num1` and `num2`.

    Returns:
    - str: A message showing the sum of num1 and num2.
    '''
    
    result = f'The sum of {num1} and {num2} is {num1 + num2}'
    return result

def handle_query(query: str):
    # Extract numbers from the query
    numbers = list(map(int, re.findall(r'\d+', query)))
    if len(numbers) == 2:
        # Pass numbers to the calculator tool
        return calculator_agent.invoke({"num1": numbers[0], "num2": numbers[1]})
    else:
        return "Please provide exactly two numbers in your query."
