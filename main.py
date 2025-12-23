# imports

import os
import sys
import asyncio
import datetime
import base64
import requests
import ee
import geemap
from typing import TypedDict, Optional
from dotenv import load_dotenv

# MCP & LangGraph
from mcp.server.fastmcp import FastMCP
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

# Load the .env file
load_dotenv() 

print(f"Loaded GOOGLE_MAPS_API_KEY: {os.environ.get('GOOGLE_MAPS_API_KEY', 'NOT SET')}")
print(f"Loaded EE_PROJECT_ID: {os.environ.get('EE_PROJECT_ID', 'NOT SET')}")


# Initialize Earth Engine
try:
    ee.Initialize(project=os.environ.get("EE_PROJECT_ID"))
except Exception:
    print(" Authenticating Earth Engine... (Follow the browser prompt)")
    ee.Authenticate()
    ee.Initialize(project=os.environ.get("EE_PROJECT_ID"))

# Initialize MCP
mcp = FastMCP("Geo-Sentinel-Agent")

# FUNCTIONS

def mask_s2_clouds(image):
    """Masks clouds in a Sentinel-2 image using the QA band."""
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)

def url_to_base64(url):
    """Download image from URL and convert to base64 for the LLM."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
    return None




# LANGGRAPH STATE & NODES

class AgentState(TypedDict):
    query: str
    location_name: str
    lat: float
    lon: float
    date_str: str
    image_url: Optional[str]
    summary: Optional[str]

def parse_intent(state: AgentState):
    """Simulate intent parsing."""
    print(f"Parsing Query: {state['query']}")
    # Default to today
    target_date = datetime.date.today().strftime("%Y-%m-%d")
    
    return {
        "location_name": state['query'], 
        "date_str": target_date
    }

def get_coordinates(state: AgentState):
    """Geocode location name using Google Maps API"""
    print(f" Geocoding: {state['location_name']}")
    
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key or "PASTE_YOUR" in api_key:
        print("Error: Google Maps API Key is missing.")
        return {"lat": 0.0, "lon": 0.0}

    # Direct Google Maps API call
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": state['location_name'],
        "key": api_key
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if data['status'] == 'OK':
            loc = data['results'][0]['geometry']['location']
            return {"lat": loc['lat'], "lon": loc['lng']}
        else:
            print(f" Geocoding Error: {data['status']}")
            return {"lat": 0.0, "lon": 0.0}
            
    except Exception as e:
        print(f"Request Failed: {e}")
        return {"lat": 0.0, "lon": 0.0}

def fetch_satellite_image(state: AgentState):
    """Fetch Sentinel-2 image from GEE."""
    if state['lat'] == 0.0: return {"image_url": None}

    print(f"  Fetching Sentinel-2 Data for ({state['lat']}, {state['lon']})...")
    
    point = ee.Geometry.Point([state['lon'], state['lat']])
    end_date = state['date_str']
    start_date = (datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.timedelta(days=90)).strftime("%Y-%m-%d")

    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(point)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .map(mask_s2_clouds))
    
    # Check if we have images
    count = collection.size().getInfo()
    if count == 0:
        print("No clear images found in the last 90 days.")
        return {"image_url": None}

    image = collection.median()
    
    vis_params = {
        'min': 0.0, 'max': 0.3, 'bands': ['B4', 'B3', 'B2'], # RGB
        'region': point.buffer(1500).bounds().getInfo()['coordinates'] # 1.5km buffer
    }
    
    url = image.getThumbURL(vis_params)
    print(f" Image URL Generated")
    return {"image_url": url}

def analyze_image(state: AgentState):
    """Send image to local Ollama"""
    url = state.get('image_url')
    if not url:
        return {"summary": "No image available to analyze."}

    print(f"Analyzing with Ollama...")
    
    # Convert to Base64 for local model stability
    img_b64 = url_to_base64(url)
    if not img_b64:
        return {"summary": "Failed to download image for analysis."}

    llm = ChatOllama(model="moondream", temperature=0.2)
    
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "You are a satellite image analyst. Describe the land use (urban, vegetation, water) and features in this image concisely."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
        ]
    )
    
    try:
        response = llm.invoke([msg])
        print("Analysis Complete")
        return {"summary": response.content}
    except Exception as e:
        return {"summary": f"Model Error: {e}"}

# BUILD GRAPH
workflow = StateGraph(AgentState)
workflow.add_node("parser", parse_intent)
workflow.add_node("locator", get_coordinates)
workflow.add_node("satellite", fetch_satellite_image)
workflow.add_node("analyst", analyze_image)

workflow.set_entry_point("parser")
workflow.add_edge("parser", "locator")
workflow.add_edge("locator", "satellite")
workflow.add_edge("satellite", "analyst")
workflow.add_edge("analyst", END)

app = workflow.compile()

# MCP TOOL DEFINITION

@mcp.tool()
async def get_satellite_analysis(query: str) -> str:
    """Retrieves and analyzes a satellite image based on a user query."""
    result = await app.ainvoke({"query": query})
    
    if not result.get('image_url'):
        return "Could not retrieve imagery. Check logs."


    return f"""
###  Satellite Report
**Target:** {result['location_name']} ({result['lat']}, {result['lon']})
**Image:** [Click to View]({result['image_url']})
**Analysis:** {result['summary']}
    """

# ENTRY POINT

if __name__ == "__main__":
    print("\n---RUNNING---")
    test_query = "Marina Bay Sands, Singapore"
    
    # Run async test loop
    async def run_test():
        print(f"Test Query: '{test_query}'")
        result = await app.ainvoke({"query": test_query})
        print("\n" + "="*30)
        print(f"FINAL OUTPUT:\n{result.get('summary')}")
        print(f"Image URL: {result.get('image_url')}")
        print("="*30 + "\n")
    
    asyncio.run(run_test())