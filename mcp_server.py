from fastapi import FastAPI, Request
from langchain_google_genai import ChatGoogleGenerativeAI
from educhain import Educhain, LLMConfig

# This creates our FastAPI app
app = FastAPI()

# Setting up Gemini (the LLM) with a model called "gemini-2.0-flash"
# Think of this as choosing which 'brain' you want to use
# You’ll need a valid Google API key here for it to actually work
gemini_flash = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # using the lightweight, fast model
    google_api_key="API_KEY" # Your API key to talk to Gemini
)

# Wrapping our Gemini model in a config object so Educhain knows how to use it
flash_config = LLMConfig(custom_model=gemini_flash)
client = Educhain(flash_config)

# Endpoint #1: Generate MCQs
# This route listens for POST requests at /generate_mcqs
@app.post("/generate_mcqs")
async def generate_mcqs(request: Request):
    data = await request.json()
    topic = data.get("topic") # Extract the topic (e.g., "Game dev")
    num = data.get("num", 5)  # Default to 5 MCQs if not specified 
    questions = client.qna_engine.generate_questions(topic=topic, num=num)
    return questions.model_dump()

# Endpoint #2: Generate Lesson Plan
# This one helps generate a structured lesson plan on any given topic
@app.post("/generate_lesson_plan")
async def generate_lesson_plan(request: Request):
    data = await request.json()
    topic = data.get("topic")
    lesson_plan = client.content_engine.generate_lesson_plan(topic=topic)
    return lesson_plan.model_dump()

# Flashcard Generator Endpoint
@app.post("/generate_flashcards")
async def generate_flashcards(request: Request):
    data = await request.json()
    topic = data.get("topic")
    num = data.get("num", 5)  # Default to 5 flashcards if not specified
    flashcards = client.content_engine.generate_flashcards(topic=topic, num=num)
    return flashcards.model_dump()
