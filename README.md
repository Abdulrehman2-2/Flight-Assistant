A Conversational AI Flight Assistant: A Multimodal Approach to Real-Time Flight Search ✈️
This paper presents a conversational AI agent built to assist users with real-time flight searches using a multimodal architecture. The system leverages a suite of specialized AI services and an external API to create a seamless, voice-enabled user experience. The core of the agent's functionality is its ability to interpret natural language requests and execute complex data-retrieval tasks.

System Architecture and Components
The agent's architecture is a pipeline that processes a user's voice input, performs a search, and generates a voice response in real-time. The key components and their roles are as follows:

Voice Activity Detection (VAD): The LiveKit platform is used to continuously monitor the audio stream to detect when a user is speaking, triggering the speech processing pipeline.

Speech-to-Text (STT): User speech is transcribed into text by the Deepgram plugin, enabling the agent to understand the verbal request.

Large Language Model (LLM): The transcribed text is sent to an OpenAI LLM (specifically, GPT-4o-mini). The LLM acts as the "brain" of the agent, interpreting user intent, identifying necessary parameters, and deciding which tool to use.

Function Calling: The LLM is provided with definitions for two specific functions: search_flights and search_cheapest_flights_in_range. Based on the user's request (e.g., "Find me a flight to London" vs. "What's the cheapest day to fly next month?"), the LLM calls the appropriate function with the correct arguments.

External API Integration: The search_flights and search_cheapest_flights_in_range functions interact with SerpApi, a third-party service that provides real-time flight data. This allows the agent to access up-to-date information without having to maintain its own database.

Text-to-Speech (TTS): After the flight search is complete, the LLM's response is converted back into natural-sounding speech using the ElevenLabs plugin.

Background Audio: The agent uses a background audio player to play a "waiting song" during the search, providing a better user experience by indicating that the system is actively working on their request and preventing awkward silence.

Core Functionality and Implementation
The agent’s intelligence lies in how it handles user requests by mapping them to specific functions.

search_flights: This function is designed for specific, single-date searches, handling both one-way and round-trip requests. It can handle a variety of parameters, including departure and arrival locations (using both city names and IATA codes), travel dates, passenger counts, travel class, and even preferences like non-stop flights. The agent's instructions ensure it confirms the request and asks for a preferred currency if one isn't provided.

search_cheapest_flights_in_range: This is a more complex function designed to find the lowest price within a date range. If a user provides only a start date, the agent is instructed to automatically set the end date to two months later, ensuring a meaningful search range. The function iterates through potential dates, using the SerpApi's calendar data, to identify the cheapest day to fly and then retrieves the detailed information for that specific flight.

The entire system is built using the LiveKit Agents framework, which simplifies the integration of these different services and handles the real-time, asynchronous nature of the conversation. The use of a function-calling LLM is crucial, as it offloads the complex task of natural language understanding and query formation to a powerful model, allowing the agent to focus on executing the tools effectively.

Conclusion
This project demonstrates a robust and scalable architecture for a conversational AI agent that solves a real-world problem. By combining real-time communication tools, advanced LLMs with function-calling capabilities, and external data sources, it creates a powerful and user-friendly service. This approach is highly flexible and can be adapted to various other domains that require real-time data retrieval, such as booking hotels, checking weather, or getting stock prices. Future improvements could include multi-step conversations to refine search criteria and support for multi-city itineraries.

 This isn't just about code it's about making travel planning more accessible and seamless for everyone.
