import logging
import os
import requests
import json
import asyncio
from dotenv import load_dotenv
from datetime import datetime, timedelta

from livekit import agents
from livekit.agents import Agent, AgentSession, function_tool, RunContext, BackgroundAudioPlayer
from livekit.agents.llm import ChatMessage, FunctionCall
from livekit.plugins import openai, elevenlabs, deepgram, silero

logger = logging.getLogger("flight_agent")
logger.setLevel(logging.INFO)

load_dotenv()

# --- SerpApi Configuration ---
SERP_API_KEY = os.getenv('SERP_API_KEY')
if not SERP_API_KEY:
    raise ValueError("SERP_API_KEY environment variable is required")

# --- Airport Code Knowledge Base ---
AIRPORT_CODES = {
    "toronto": "YYZ",
    "karachi": "KHI",
    "lahore": "LHE",
    "new york": "JFK",
    "london": "LHR",
    "dubai": "DXB",
    "frankfurt": "FRA",
    "paris": "CDG",
    "los angeles": "LAX",
    "chicago": "ORD",
    "tokyo": "HND",
    "sydney": "SYD",
    "mumbai": "BOM",
    "delhi": "DEL",
    "beijing": "PEK",
    "singapore": "SIN",
    "hong kong": "HKG",
    "san francisco": "SFO",
    "vancouver": "YVR",
    "montreal": "YUL",
    "calgary": "YYC",
    "halifax": "YHZ",
    "ottawa": "YOW",
    "lon": "LHR",
}

# --- SerpApi Google Flights API Function ---

def _format_duration_serpapi(minutes: int) -> str:
    """Helper to format duration in minutes into a readable string."""
    if minutes is None:
        return "N/A"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    if hours > 0 and remaining_minutes > 0:
        return f"{hours} hours {remaining_minutes} minutes"
    elif hours > 0:
        return f"{hours} hours"
    elif remaining_minutes > 0:
        return f"{remaining_minutes} minutes"
    return "0 minutes"

def search_flights(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,
    return_date: str = None,
    adults: int = 1,
    children: int = 0,
    infants_in_seat: int = 0,
    infants_on_lap: int = 0,
    travel_class: str = "1",
    currency: str = None, 
    gl: str = None,
    hl: str = None,
    type: str = "1",
    multi_city_json: str = None,
    show_hidden: bool = False,
    deep_search: bool = False,
    sort_by: str = None,
    stops: str = "0",
    exclude_airlines: str = None,
    include_airlines: str = None,
    bags: int = 0,
    max_price: int = None,
    outbound_times: str = None,
    return_times: str = None,
    emissions: str = None,
    layover_duration: str = None,
    exclude_conns: str = None,
    max_duration: int = None,
    departure_token: str = None,
    booking_token: str = None,
) -> dict:
    """
    Searches for flight offers using the SerpApi Google Flights API.
    """
    url = "https://serpapi.com/search.json"
    effective_departure_id = AIRPORT_CODES.get(departure_id.lower(), departure_id.upper())
    effective_arrival_id = AIRPORT_CODES.get(arrival_id.lower(), arrival_id.upper())

    params = {
        "engine": "google_flights",
        "api_key": SERP_API_KEY,
    }
    
    if departure_id:
        params["departure_id"] = effective_departure_id
    if arrival_id:
        params["arrival_id"] = effective_arrival_id
    if outbound_date:
        params["outbound_date"] = outbound_date
    if return_date:
        params["return_date"] = return_date
    if adults is not None:
        params["adults"] = adults
    if children is not None:
        params["children"] = children
    if infants_in_seat is not None:
        params["infants_in_seat"] = infants_in_seat
    if infants_on_lap is not None:
        params["infants_on_lap"] = infants_on_lap
    if travel_class:
        params["travel_class"] = travel_class
    if currency:
        params["currency"] = currency
    if gl:
        params["gl"] = gl
    if hl:
        params["hl"] = hl
    if type:
        params["type"] = type
    if multi_city_json:
        params["multi_city_json"] = multi_city_json
    if show_hidden:
        params["show_hidden"] = "true"
    if deep_search:
        params["deep_search"] = "true"
    if sort_by:
        params["sort_by"] = sort_by
    if stops:
        params["stops"] = stops
    if exclude_airlines:
        params["exclude_airlines"] = exclude_airlines
    if include_airlines:
        params["include_airlines"] = include_airlines
    if bags is not None:
        params["bags"] = bags
    if max_price is not None:
        params["max_price"] = max_price
    if outbound_times:
        params["outbound_times"] = outbound_times
    if return_times:
        params["return_times"] = return_times
    if emissions:
        params["emissions"] = emissions
    if layover_duration:
        params["layover_duration"] = layover_duration
    if exclude_conns:
        params["exclude_conns"] = exclude_conns
    if max_duration is not None:
        params["max_duration"] = max_duration
    if departure_token:
        params["departure_token"] = departure_token
    if booking_token:
        params["booking_token"] = booking_token

    logger.info(f"DEBUGGING SerpApi FLIGHTS Request Params: {params}")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        if 'error' in results:
            logger.error(f"SerpApi Error Response: {results['error']}")
            return {"error": results['error']}
        
        return results

    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching flights with SerpApi: {e}")
        return {"error": "The flight search service is temporarily unavailable. Please try again later."}


def search_cheapest_flights_in_range(
    departure_id: str,
    arrival_id: str,
    start_date: str,
    end_date: str,
    adults: int = 1,
    travel_class: str = "1",
    currency: str = None,
    stops: str = "0",
) -> dict:
    """
    Finds the cheapest flight within a specified date range.
    """
    
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        return {"error": f"Invalid date format provided: {e}"}

    today = datetime.now()
    if start_dt < today:
        start_dt = today
    if end_dt < start_dt:
        end_dt = end_dt.replace(year=start_dt.year + 1) if start_dt.month > end_dt.month else end_dt.replace(year=start_dt.year)
        if end_dt < start_dt:
             end_dt = end_dt + timedelta(days=365)

    cheapest_flight_details = None
    cheapest_price = float('inf')
    cheapest_date = None
    
    current_month = start_dt.replace(day=1)
    
    while current_month <= end_dt:
        search_start_day = max(start_dt, current_month)
        
        logger.info(f"Searching for cheapest flight in month starting from {search_start_day.strftime('%Y-%m-%d')}")
        
        month_results = search_flights(
            departure_id=departure_id,
            arrival_id=arrival_id,
            outbound_date=search_start_day.strftime("%Y-%m-%d"),
            adults=adults,
            travel_class=travel_class,
            currency=currency,
            stops=stops,
            type="2", # Force one-way search for calendar data
        )
        
        if 'error' in month_results:
            logger.warning(f"Skipping month due to error: {month_results['error']}")
            current_month = current_month + timedelta(days=32)
            continue
            
        calendar_data = month_results.get('best_flights_calendar', [])
        
        if calendar_data:
            for day in calendar_data:
                day_date_str = day.get('date')
                day_price = day.get('price')
                if not day_date_str or day_price is None:
                    continue
                day_date = datetime.strptime(day_date_str, "%Y-%m-%d")
                
                if start_dt <= day_date <= end_dt:
                    if day_price < cheapest_price:
                        cheapest_price = day_price
                        cheapest_date = day_date_str
        else:
            flight_offers = month_results.get('best_flights', [])
            if flight_offers:
                current_price = flight_offers[0].get('price', float('inf'))
                current_date_str = search_start_day.strftime("%Y-%m-%d")
                if current_price is not None and current_price < cheapest_price:
                    cheapest_price = current_price
                    cheapest_date = current_date_str
                    
        current_month = current_month.replace(day=1) + timedelta(days=32)
        current_month = current_month.replace(day=1)
        
    if cheapest_date and cheapest_price != float('inf'):
        final_flight_details = search_flights(
            departure_id=departure_id,
            arrival_id=arrival_id,
            outbound_date=cheapest_date,
            type="2", # Force one-way search for specific details
            adults=adults,
            travel_class=travel_class,
            currency=currency,
            stops=stops,
        )
        
        if 'error' in final_flight_details:
            return {"error": "Found a cheapest date, but could not retrieve specific flight details."}
        
        best_flight_offer = final_flight_details.get('best_flights', [])
        if not best_flight_offer:
            return {"error": f"Could not find flight details for {cheapest_date}."}
        
        cheapest_flight_details = best_flight_offer[0]
        
        price = cheapest_flight_details.get('price', 'N/A')
        total_duration = cheapest_flight_details.get('total_duration', 'N/A')
        airline_name = "Various Airlines"
        if cheapest_flight_details.get('flights') and cheapest_flight_details['flights'][0].get('airline'):
            airline_name = cheapest_flight_details['flights'][0]['airline']
        
        layover_info = ""
        if cheapest_flight_details.get('layovers'):
            layover_count = len(cheapest_flight_details['layovers'])
            if layover_count > 0:
                layover_airports = [layover['id'] for layover in cheapest_flight_details['layovers']]
                layover_info = f" with {layover_count} stop(s) in {', '.join(layover_airports)}"
            else:
                layover_info = " (direct flight)"
        else:
            layover_info = " (direct flight)"

        currency_str = currency if currency else 'USD'
        message = (
            f"I have found the cheapest flight between {start_date} and {end_date}. "
            f"The best price is on **{cheapest_date}** for **{price} {currency_str}**. "
            f"The flight is on {airline_name}, with a total travel time of "
            f"approximately {_format_duration_serpapi(total_duration)}{layover_info}."
        )
        return {"message": message}
    else:
        return {
            "message": (
                f"I could not find any flights between {start_date} and {end_date}. "
                "It's possible that flight schedules for those dates are not yet available. "
                "Please check the dates or try a different destination."
            )
        }

# --- Agent Class and Main Entrypoint ---

class FlightAssistant(Agent):
    def __init__(self, tools: list) -> None:
        super().__init__(
            tools=tools,
            instructions="""You are a helpful voice AI assistant specialized in finding flights.
Keep your questions and responses as short and concise as possible.
You can search for flights by providing a departure city/airport, an arrival city/airport, and a departure date.
I have a knowledge base of common airport codes, so you can often use city names like "Toronto" or "Karachi", or direct IATA codes like "YYZ" or "KHI".
You can also find the cheapest flight for a user within a specified date range.
When a user provides a date without a year, always assume the current year or the next available year to ensure the date is in the future.
For example, if the current date is August 8th, 2025 and the user says "August 18th", you should assume "August 18th, 2025".
If the user says "February 1st" in August 2025, you should assume "February 1st, 2026".
Always confirm the user's request before performing a search.
You must always ask the user for their preferred currency (CAD, USD, or PKR) if they do not provide one.
You can also specify the number of adults, travel class (Economy, Premium economy, Business, First), and whether it's a non-stop flight.

When a user asks for the cheapest flight "on or after" a certain date without providing an end date, **automatically set the end date to two months after the specified start date.** Before performing the search, state the date range you will be checking. For example, "Okay, I'll check for the cheapest flights from [start date] until [end date]."

When presenting flight results, be conversational and highlight key information like:
- Total price in the requested currency
- Total flight duration
- Number of layovers and where they are (if any)
- The main airline
"""
        )
        self.background_audio = BackgroundAudioPlayer()

    async def on_tool_call(self, tool_call: FunctionCall) -> dict:
        """Handles calls from the LLM to defined tools (API functions)."""
        
        await self.background_audio.start(room=self.room, agent_session=self.session)
        audio_file = "waiting song.mp3"
        audio_path = os.path.join(os.path.dirname(__file__), "assistant", audio_file)
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
        
        handle = self.background_audio.play(audio=audio_path, loop=True) if os.path.exists(audio_path) else None
        
        stop_task = None
        if handle:
            async def stop_audio_after_delay():
                await asyncio.sleep(15)
                if not handle.stopped:
                    handle.stop()
            
            stop_task = asyncio.create_task(stop_audio_after_delay())

        try:
            if tool_call.function.name == "search_flights":
                result = await asyncio.to_thread(search_flights, **tool_call.function.args)
            elif tool_call.function.name == "search_cheapest_flights_in_range":
                result = await asyncio.to_thread(search_cheapest_flights_in_range, **tool_call.function.args)
            else:
                raise ValueError(f"Unknown tool: {tool_call.function.name}")
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_call.function.name}: {e}")
            return {"error": "An internal error occurred while searching for flights. Please try again."}
        finally:
            if handle and not handle.stopped:
                handle.stop()
            if stop_task:
                stop_task.cancel()
            await self.background_audio.aclose()


async def entrypoint(ctx: agents.JobContext):
    """Main entrypoint for the flight assistant agent."""
    
    search_flights_schema = {
        "name": "search_flights",
        "description": "Search for available flights between two airports/cities on a specific date. Can accept IATA airport codes or common city names. Supports one-way and round-trip searches.",
        "parameters": {
            "type": "object",
            "properties": {
                "departure_id": {
                    "type": "string",
                    "description": "The IATA airport code or common city name for the departure location (e.g., 'YYZ' or 'Toronto'). This is required."
                },
                "arrival_id": {
                    "type": "string",
                    "description": "The IATA airport code or common city name for the arrival location (e.g., 'KHI' or 'Karachi'). This is required."
                },
                "outbound_date": {
                    "type": "string",
                    "description": "The departure date in YYYY-MM-DD format. This is required."
                },
                "return_date": {
                    "type": "string",
                    "description": "The return date for a round-trip search in YYYY-MM-DD format. Optional for one-way flights."
                },
                "adults": {
                    "type": "integer",
                    "description": "Number of adult passengers. Defaults to 1.",
                    "default": 1
                },
                "children": {
                    "type": "integer",
                    "description": "Number of children passengers. Defaults to 0.",
                    "default": 0
                },
                "infants_in_seat": {
                    "type": "integer",
                    "description": "Number of infants who will occupy a seat. Defaults to 0.",
                    "default": 0
                },
                "infants_on_lap": {
                    "type": "integer",
                    "description": "Number of infants on a lap. Defaults to 0.",
                    "default": 0
                },
                "travel_class": {
                    "type": "string",
                    "enum": ["1", "2", "3", "4"],
                    "description": "The travel class as a numerical string (1: Economy, 2: Premium economy, 3: Business, 4: First). Defaults to 1.",
                    "default": "1"
                },
                "currency": {
                    "type": "string",
                    "description": "Currency code for prices (e.g., 'USD', 'CAD'). The agent will ask for this if not provided."
                },
                "gl": {
                    "type": "string",
                    "description": "Two-letter country code to use for the search (e.g., 'us')."
                },
                "hl": {
                    "type": "string",
                    "description": "Two-letter language code for the search (e.g., 'en')."
                },
                "type": {
                    "type": "string",
                    "enum": ["1", "2", "3"],
                    "description": "The type of flight search. '1' for round-trip, '2' for one-way, '3' for multi-city. Defaults to '1'.",
                    "default": "1"
                },
                "multi_city_json": {
                    "type": "string",
                    "description": "A JSON string containing an array of flight information objects for multi-city searches. Only used when type is '3'."
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Set to true to include hidden flight results. Defaults to false.",
                    "default": False
                },
                "deep_search": {
                    "type": "boolean",
                    "description": "Set to true to enable deep search for more precise results. Defaults to false.",
                    "default": False
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["1", "2", "3", "4", "5", "6"],
                    "description": "The sorting order of the results. '1' for Top flights, '2' for Price, '3' for Departure time, '4' for Arrival time, '5' for Duration, '6' for Emissions. Defaults to '1'."
                },
                "stops": {
                    "type": "string",
                    "enum": ["0", "1", "2", "3"],
                    "description": "The number of stops allowed. '0' for nonstop only, '1' for 1 stop or fewer, '2' for 2 stops or fewer, '3' for any stops. Defaults to '0'.",
                    "default": "0"
                },
                "exclude_airlines": {
                    "type": "string",
                    "description": "Comma-separated IATA airline codes to be excluded from the search."
                },
                "include_airlines": {
                    "type": "string",
                    "description": "Comma-separated IATA airline codes or alliances to be included in the search. Cannot be used with exclude_airlines."
                },
                "bags": {
                    "type": "integer",
                    "description": "Number of carry-on bags. Defaults to 0.",
                    "default": 0
                },
                "max_price": {
                    "type": "integer",
                    "description": "The maximum price for a ticket."
                },
                "outbound_times": {
                    "type": "string",
                    "description": "Time range for the outbound flight departure/arrival. Format is two or four comma-separated numbers (e.g., '4,18' for departure between 4AM-7PM)."
                },
                "return_times": {
                    "type": "string",
                    "description": "Time range for the return flight departure/arrival. Format is two or four comma-separated numbers."
                },
                "emissions": {
                    "type": "string",
                    "enum": ["1"],
                    "description": "Set to '1' to only show flights with less emissions."
                },
                "layover_duration": {
                    "type": "string",
                    "description": "Layover duration range in minutes, as two comma-separated numbers (e.g., '90,330')."
                },
                "exclude_conns": {
                    "type": "string",
                    "description": "Comma-separated IATA airport codes to exclude as connecting airports."
                },
                "max_duration": {
                    "type": "integer",
                    "description": "Maximum total flight duration in minutes."
                },
                "multi_city_json": {
                    "type": "string",
                    "description": "A JSON string containing an array of flight information objects for multi-city searches. Only used when type is '3'."
                },
                "departure_token": {
                    "type": "string",
                    "description": "Token for retrieving returning flights in a round-trip."
                },
                "booking_token": {
                    "type": "string",
                    "description": "Token for retrieving booking options for selected flights."
                }
            },
            "required": ["departure_id", "arrival_id", "outbound_date"]
        }
    }
    
    search_cheapest_in_range_schema = {
        "name": "search_cheapest_flights_in_range",
        "description": "Searches for the single cheapest one-way flight between two airports/cities within a specified date range. This function is useful when a user asks for the cheapest day to fly between two dates.",
        "parameters": {
            "type": "object",
            "properties": {
                "departure_id": {
                    "type": "string",
                    "description": "The IATA airport code or common city name for the departure location (e.g., 'YYZ' or 'Toronto'). This is required."
                },
                "arrival_id": {
                    "type": "string",
                    "description": "The IATA airport code or common city name for the arrival location (e.g., 'KHI' or 'Karachi'). This is required."
                },
                "start_date": {
                    "type": "string",
                    "description": "The beginning date of the search range in YYYY-MM-DD format. This is required."
                },
                "end_date": {
                    "type": "string",
                    "description": "The end date of the search range in YYYY-MM-DD format. This is required."
                },
                "adults": { "type": "integer", "description": "Number of adult passengers. Defaults to 1.", "default": 1 },
                "travel_class": { "type": "string", "enum": ["1", "2", "3", "4"], "description": "The travel class as a numerical string (1: Economy, 2: Premium economy, 3: Business, 4: First). Defaults to 1.", "default": "1" },
                "currency": { "type": "string", "description": "Currency code for prices (e.g., 'USD', 'CAD'). The agent will ask for this if not provided." },
                "stops": {
                    "type": "string",
                    "enum": ["0", "1", "2", "3"],
                    "description": "The number of stops allowed. '0' for nonstop only, '1' for 1 stop or fewer, '2' for 2 stops or fewer, '3' for any stops. Defaults to '0'.",
                    "default": "0"
                },
            },
            "required": ["departure_id", "arrival_id", "start_date", "end_date"]
        }
    }

    async def search_flights_handler(raw_arguments: dict, context: RunContext):
        return await asyncio.to_thread(search_flights, **raw_arguments)

    async def search_cheapest_in_range_handler(raw_arguments: dict, context: RunContext):
        return await asyncio.to_thread(search_cheapest_flights_in_range, **raw_arguments)

    tools = [
        function_tool(search_flights_handler, raw_schema=search_flights_schema),
        function_tool(search_cheapest_in_range_handler, raw_schema=search_cheapest_in_range_schema)
    ]
    
    agent_instance = FlightAssistant(tools=tools)

    session = AgentSession(
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(
            voice_id="EXAVITQu4vr4xnSDxMaL",
            model="eleven_flash_v2_5"
        ),
        stt=deepgram.STT(language="en-US"),
        vad=ctx.proc.userdata["vad"],
    )
    
    await session.start(
        room=ctx.room,
        agent=agent_instance,
    )

    await session.generate_reply(
        instructions="Hey, what flights do you want me to search for?"
    )

def prewarm(proc: agents.JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)
    )

