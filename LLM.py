import streamlit as st
import openai
from openai import OpenAI

import boto3
from langchain_aws import ChatBedrock

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# Assume imports for other providers like Anthropic or Cohere are added as needed

import os
aws_id = os.getenv("aws_access_key_id")
aws_key = os.getenv("aws_secret_access_key")
api_key = os.getenv("api_key")

# Initialize the client for AWS Bedrock
client = boto3.client(
        service_name="bedrock-runtime", region_name="us-east-1",
        aws_access_key_id= aws_id ,  # Replace with your actual access key ID
        aws_secret_access_key= aws_key,  # Replace with your actual secret access key

    )

# chat_llm = ChatBedrock(client = client, model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0", model_kwargs = {"temperature": 0})

template = ChatPromptTemplate(
    messages = [
        SystemMessagePromptTemplate.from_template(template = """
                                                 You are a helpful math assistant. When a user submits a math problem, provide a detailed, step-by-step explanation that shows your full reasoning process leading to the final answer. Break down each part of the problem clearly, ensuring that every step is understandable. If you encounter a problem where you are uncertain or cannot determine the correct answer, simply reply with "I don't know" without making any guesses.
                                                 """
                                                 ),
        HumanMessage(content =
                     """
                    Solve for \\( x \\) in the equation: \\(2x + 3 = 11\\)
                    """
                    ),

        AIMessage(content =
                  """
                  **Step 1:** Write down the equation:
                    \\(2x + 3 = 11\\)
                    
                    **Step 2:** Subtract 3 from both sides to isolate the term with \\( x \\):
                    \\[
                    2x + 3 - 3 = 11 - 3 \\quad \\Longrightarrow \\quad 2x = 8
                    \\]
                    
                    **Step 3:** Divide both sides by 2 to solve for \\( x \\):
                    \\[
                    \\frac{2x}{2} = \\frac{8}{2} \\quad \\Longrightarrow \\quad x = 4
                    \\]
                    
                    **Final Answer:** \\( x = 4 \\)
                """
                 ),

        HumanMessagePromptTemplate.from_template(template = "{text}")
    ],

    input_variable = ["text"]
)

def LLM_invoke(text, chat_llm):
  messages = template.format_messages(text = text)
  try:
    result = chat_llm.invoke(messages).content
  except Exception as e:
      print(f"Model invocation failed: {e}")
      return None
  return result



# Sidebar for API key input and model selection
with st.sidebar:
    st.title("🔧 Settings")
    model_provider = st.selectbox("Select Language Model Provider", ["DeepSeek", "OpenAI GPT Series", "Anthropic Claude Series", "Meta Llama Series"])
    
    # Capture the API key based on the selected model provider
    if model_provider == "DeepSeek":
        st.info("DeepSeek is currently not supported due to regulations.")

    if model_provider == "OpenAI GPT Series":
        # api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        model_name = st.selectbox("Select OpenAI Model", ["gpt-4o-mini", "o1-mini"])
    
        # Information links
        st.markdown("[Get an API Key](https://platform.openai.com/account/api-keys)")\

    if model_provider == "Anthropic Claude Series":
        model_name = st.selectbox("Select Claude Model", ["Claude 3 Haiku", "Claude 3.5 Sonnet v1"])
        st.info("These models are supported by AWS Bedrock. Due to token limitation, It cannot be requested many times in a short period of time. Please use it wisely.")

    if model_provider == "Meta Llama Series":
        model_name = st.selectbox("Select Llama Model", ["Llama 3.2 3B", "Llama 3.3 70B"])
        st.info("These models are supported by AWS Bedrock. Due to token limitation, It cannot be requested many times in a short period of time. Please use it wisely.")

st.title("💬 Your math assistant")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input prompt and handle response generation
if prompt := st.chat_input():
    # if model_provider == "OpenAI" and not api_key:
    #     st.info("Please add your API key to continue.")
    #     st.stop()

    # Append user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Generate response based on selected model provider
    if model_provider == "DeepSeek":
        msg = "Sorry, DeepSeek is currently not supported due to regulations."
        # response = client_ds.chat.completions.create(
        #     model="deepseek-chat",
        #     messages=st.session_state.messages
        # )
        # msg = response.choices[0].message.content

    elif model_provider == "OpenAI GPT Series":
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=st.session_state.messages
        )
        msg = response.choices[0].message.content

    elif model_provider == "Anthropic Claude Series":
        # Anthropic API call
        if model_name == "Claude 3 Haiku":
            chat_llm = ChatBedrock(client = client, model_id = "anthropic.claude-3-haiku-20240307-v1:0", model_kwargs = {"temperature": 0})
        elif model_name =="Claude 3.5 Sonnet v1":
            chat_llm = ChatBedrock(client = client, model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0", model_kwargs = {"temperature": 0})
        msg = LLM_invoke(prompt, chat_llm)

    elif model_provider == "Meta Llama Series":
        if model_name == "Llama 3.2 3B":
            chat_llm = ChatBedrock(client = client, model_id = "meta.llama3-2-3b-instruct-v1:0", model_kwargs = {"temperature": 0})
        elif model_name =="Llama 3.3 70B":
            chat_llm = ChatBedrock(client = client, model_id = "meta.llama3-3-70b-instruct-v1:0", model_kwargs = {"temperature": 0})
        msg = LLM_invoke(prompt, chat_llm)


    # Append assistant's message to the chat history
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)