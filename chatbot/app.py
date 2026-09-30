from fastapi import FastAPI
import uvicorn
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from logger import logging
from llm import get_llm
from prompt import get_prompt
from entity import ChatReq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Chatbot",
    version="1.0",
    description="A simple chatbot API server using LangChain and Groq"
)

logging.info("initialized FastAPI server")

# For a simple in-memory conversation history
history = []

# Initialize the LLM
llm = get_llm()
logging.info("initialized LLM")

# Initialize the output parser
parser = StrOutputParser()
logging.info("initialized output parser")


@app.get("/")
async def index():
    response_data = {
        "status": "success",
        "response": "Welcome to the Chatbot !!!"
    }
    return response_data


@app.post("/api/chatbot")
async def chatbot(chat_req: ChatReq):
    global history

    question = chat_req.question
    logging.info(f"Question: {question}")

    response_data = {
        "status": "success",
        "response": ""
    }

    try:
        # Load the prompt template
        prompt = get_prompt()
        logging.info("Load prompt template")

        # Create the LangChain pipeline
        chain = prompt | llm | parser
        logging.info("Create chain")

        # Keep the last four messages
        history = history[-4:]

        # Generate the response
        response = chain.invoke({
            "history": history, 
            "question": [HumanMessage(content=question)]
            })
        logging.info(f"Response: {response}")

        # Save the conversation
        history.extend([HumanMessage(content=question),
                        AIMessage(content=response)])

        response_data["response"] = response
        return response_data
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        response_data["response"] = "Sorry, an error occurred while processing your question."
        response_data["status"] = "fail"
        return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)