import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from IPython.display import display, HTML
import ipywidgets as widgets

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize Language Model using API keys from environment variables
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

# Define Memory for the Chatbot
memory = ConversationBufferMemory()

# Define Prompt Template for the conversation
prompt = PromptTemplate(
    input_variables=["history", "human_input"],
    template="This is a conversation:\n{history}\nHuman: {human_input}\nAI:"
)

# Create the Conversational Chain
chat_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

def chat_with_bot(user_input):
    """Handles user input and returns the chatbot response text using .invoke()."""
    result = chat_chain.invoke({"human_input": user_input})
    # Extract and return only the text response if available
    if isinstance(result, dict) and "text" in result:
        return result["text"]
    return result

def start_console_chat():
    """Starts an interactive chatbot conversation in the console."""
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chat_with_bot(user_input)
        print("Bot:", response)

def get_session_history(session_id):
    """
    Retrieve chat message history for a given session.
    For now, returns a new ChatMessageHistory instance.
    In a full implementation, you'd store and retrieve histories based on the session_id.
    """
    return ChatMessageHistory()

if __name__ == "__main__":
    # Start the console chat if this script is executed directly
    start_console_chat()
