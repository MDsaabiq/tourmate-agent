import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import re

# Load environment variables
load_dotenv()

class TripRequest(BaseModel):
    request: str
    city: str
    interests: str
    duration: int

class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "the messages in conversation"]
    city: str
    interests: List[str]
    duration: int
    itinerary: List[dict]

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash"
)

itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert travel guide. Create a realistic and enjoyable {duration}-day itinerary for {city} for travelers interested in {interests}.

CRITICAL: Each day MUST be COMPLETELY DIFFERENT with unique locations, attractions, and activities. Do NOT repeat the same places or activities across different days.

For EACH day:
1. Focus on a DIFFERENT neighborhood, district, or area of {city}
2. Provide specific attractions, museums, parks, restaurants unique to that area
3. Create a time-based schedule for the ENTIRE day
4. Mix famous sites with local hidden gems

Day-by-Day Structure:
- Day 1: Explore [First distinct area] focusing on [specific interest]
- Day 2: Visit [Different distinct area] focusing on [different aspect]
- Day 3: Discover [Another distinct area] with [different activities]
(Continue pattern - each day in a NEW location/neighborhood)

IMPORTANT - Return ONLY valid JSON with NO extra text:
[
  {{
    "day": 1,
    "title": "Day 1 - [Area Name] - [Catchy title about unique experience]",
    "description": "8:00 AM - Start at [specific landmark in Area 1]\n10:00 AM - Visit [unique museum/attraction in Area 1]\n12:30 PM - Lunch at [local restaurant in Area 1]\n3:00 PM - Explore [neighborhood/park in Area 1]\n6:00 PM - [Evening activity in Area 1]\n8:00 PM - Dinner at [restaurant in Area 1]",
    "location": "[Specific area name/neighborhood - NOT the city]"
  }},
  {{
    "day": 2,
    "title": "Day 2 - [Different Area Name] - [Different experience title]",
    "description": "8:00 AM - Begin at [unique starting point in Area 2]\n10:00 AM - Visit [different type of attraction in Area 2]\n12:30 PM - Lunch at [different restaurant in Area 2]\n3:00 PM - Explore [different neighborhood/site in Area 2]\n6:00 PM - [Different evening activity in Area 2]\n8:00 PM - Dinner at [different restaurant type in Area 2]",
    "location": "[Completely different area/neighborhood - NOT same as Day 1]"
  }}
]

Critical Guidelines:
- VARIATION: No day should visit the same location, restaurant, or neighborhood as another day
- Use specific, real place names for EACH location (research {city} geography)
- Each day focuses on a DIFFERENT neighborhood or district
- Include realistic HH:MM AM/PM times for all activities
- Activities MUST align with {interests} but vary across days
- Separate activities with line breaks (\n)
- Make it practical and balanced (not exhausting)
- Return ONLY the JSON array, nothing else""",),
    ("human", "Create a {duration}-day itinerary for {city} for someone interested in {interests}. Make sure each day visits completely different areas and neighborhoods with unique activities."),
])

def set_city(state: PlannerState) -> PlannerState:
    return {
        **state,
        "messages": state['messages'] + [HumanMessage(content=f"City: {state['city']}")],
    }

def set_interests(state: PlannerState) -> PlannerState:
    return {
        **state,
        "messages": state['messages'] + [HumanMessage(content=f"Interests: {', '.join(state['interests'])}")],
    }

def create_itinerary(state: PlannerState) -> PlannerState:
    print(f"Creating a {state['duration']}-day itinerary for {state['city']} based on interests: {', '.join(state['interests'])} and itineray should be planned for whole day time wise")
    
    response = llm.invoke(itinerary_prompt.format_messages(
        city=state['city'], 
        interests=', '.join(state['interests']),
        duration=state['duration']
    ))
    
    print("\nAI Response:")
    print(response.content)
    
    # Parse the JSON response
    try:
        # Extract JSON from response
        content = response.content.strip()
        
        # Try to find JSON array in the response
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            # Replace actual newlines with escaped newlines for proper JSON parsing
            json_str = json_str.replace('\n', '\\n')
            itinerary_data = json.loads(json_str)
        else:
            # Fallback: try to parse the entire content
            content = content.replace('\n', '\\n')
            itinerary_data = json.loads(content)
        # Validate and ensure proper structure
        validated_itinerary = []
        for item in itinerary_data:
            validated_item = {
                "day": item.get("day", len(validated_itinerary) + 1),
                "title": item.get("title", f"Day {item.get('day', len(validated_itinerary) + 1)} Activities"),
                "description": item.get("description", "Activities planned for this day"),
                "location": item.get("location", state['city'])
            }
            validated_itinerary.append(validated_item)
        
        return {
            **state,
            "messages": state['messages'] + [AIMessage(content=response.content)],
            "itinerary": validated_itinerary,
        }
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing JSON: {e}")
        # Fallback itinerary
        fallback_itinerary = []
        for day in range(1, state['duration'] + 1):
            fallback_itinerary.append({
                "day": day,
                "title": f"Explore {state['city']} - Day {day}",
                "description": f"Discover the best of {state['city']} with activities related to {', '.join(state['interests'])}",
                "location": state['city']
            })
        
        return {
            **state,
            "messages": state['messages'] + [AIMessage(content="Generated fallback itinerary")],
            "itinerary": fallback_itinerary,
        }

workflow = StateGraph(PlannerState)

workflow.add_node("set_city", set_city)
workflow.add_node("set_interests", set_interests)
workflow.add_node("create_itinerary", create_itinerary)

workflow.set_entry_point('set_city')

workflow.add_edge("set_city", "set_interests")
workflow.add_edge('set_interests', 'create_itinerary')
workflow.add_edge('create_itinerary', END)

app_graph = workflow.compile()

def travel_planner(user_request: str, city: str, interests: List[str], duration: int):
    print(f"Initial request: {user_request}\n")
    state = {
        "messages": [HumanMessage(content=user_request)],
        "city": city,
        "interests": interests,
        "duration": duration,
        "itinerary": [],
    }
    final_state = None
    for output in app_graph.stream(state):
        final_state = output
    
    # Extract the final itinerary from the last state
    if final_state:
        for node_output in final_state.values():
            if 'itinerary' in node_output and node_output['itinerary']:
                return node_output['itinerary']
    
    return []

# FastAPI app
app = FastAPI(title="TourMate AI Itinerary Generator", version="1.0.0")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )

@app.get("/")
def read_root():
    return {"message": "TourMate AI Itinerary Generator", "status": "active"}

@app.post("/plan-trip")
def plan_trip(trip_request: TripRequest):
    try:
        interests_list = [interest.strip() for interest in trip_request.interests.split(",")]
        result = travel_planner(trip_request.request, trip_request.city, interests_list, trip_request.duration)
        return {"message": "Trip planning completed", "itinerary": result}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Failed to generate trip plan")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8881))
    uvicorn.run("main:app", host="0.0.0.0", port=port)


