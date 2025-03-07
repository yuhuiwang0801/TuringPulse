import streamlit as st
import openai
from openai import OpenAI

import boto3
from langchain_aws import ChatBedrock

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import requests
import base64

# Assume imports for other providers like Anthropic or Cohere are added as needed

import os
aws_id = os.getenv("aws_access_key_id")
aws_key = os.getenv("aws_secret_access_key")
api_key = os.getenv("api_key")
grok_api_key = os.getenv("grok_api_key")


# Initialize the client for AWS Bedrock
client = boto3.client(
        service_name="bedrock-runtime", region_name="us-east-1",
        aws_access_key_id= aws_id ,  # Replace with your actual access key ID
        aws_secret_access_key= aws_key,  # Replace with your actual secret access key

    )

client_grok = OpenAI(
  api_key=grok_api_key,
  base_url="https://api.x.ai/v1",
)

# chat_llm = ChatBedrock(client = client, model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0", model_kwargs = {"temperature": 0})

template = ChatPromptTemplate(
    messages = [
        SystemMessagePromptTemplate.from_template(template = """
                                                 You are a helpful math assistant. When a user submits a math or logic reasoning problem, provide a detailed, step-by-step explanation that shows your full reasoning process leading to the final answer. Break down each part of the problem clearly, ensuring that every step is understandable. 
                                                 If you encounter a problem where you are uncertain or cannot determine the correct answer, simply reply with "I don't know" without making any guesses.

                                                Additionally, you must follow these instructions:
                                                
                                                1. Maintain a detailed chain-of-thought encapsulated in `<thinking> ... </thinking>` tags. This chain-of-thought should consist of 10-15 paragraphs that summarize your internal reasoning process.
                                                2. Present both interim and final results using `<output> ... </output>` tags. Use your interim results to derive the final answer, and ensure that your final answer is clearly highlighted within these tags.
                                                3. Your response should be structured so that the chain-of-thought (inside `<thinking>` tags) fully explains your reasoning step-by-step, and the final answer (inside `<output>` tags) is directly derived from that reasoning.
                                                4. Ensure that your explanation is complete and well-organized so that every step is understandable by the user.
                                                
                                                By following these guidelines, your answer will include both the internal reasoning process and the final result in a clear, structured manner.
                                                 """
                                                 ),
        HumanMessage(content =
                     """
                    question: "If the expression \(x^2 - 4x + 4\) is equal to 25, for which value(s) of \(x\) does the equation hold?"
                    """
                    ),

        AIMessage(content =
                  """
                  math_problem
                  steps:
                    - step_number: 1
                    explanation: >
                        Start by writing the equation: \(x^2 - 4x + 4 = 25\).
                    - step_number: 2
                    explanation: >
                        Move all terms to one side to set the equation to zero: 
                        \(x^2 - 4x + 4 - 25 = 0\), which simplifies to \(x^2 - 4x - 21 = 0\).
                    - step_number: 3
                    explanation: >
                        Factor the quadratic, if possible. Observe that 
                        \(-7 \times 3 = -21\) and \(-7 + 3 = -4\). 
                        So we can factor it as \((x - 7)(x + 3) = 0\).
                    - step_number: 4
                    explanation: >
                        Use the Zero Product Property: if \((x - 7)(x + 3) = 0\), 
                        then \(x - 7 = 0\) or \(x + 3 = 0\), leading to 
                        \(x = 7\) or \(x = -3\).
                answer: "The values of x that satisfy the equation are x = 7 and x = -3."
                """
                 ),

        HumanMessage(content =
                     """
                    Hat Puzzle:

                    Three people—Ann, Bob, and Chad—are each given a hat. The hats can be either black or white. They know there are exactly two black hats and one white hat in total. They stand in a line such that:

                    - Ann can see Bob and Chad.
                    - Bob can see Chad.
                    - Chad cannot see anyone.
                    They cannot see their own hats. Each is asked, in turn, if they know the color of their own hat:

                    1. Ann looks at Bob and Chad’s hats but says she does not know her own hat color.
                    2. Bob looks at Chad’s hat and also says he does not know his own hat color.
                    3. Chad then announces that he does know the color of his own hat.
                    Question: What color is Chad’s hat, and how does Chad figure it out?
                    """
                    ),

        AIMessage(content =
                  """
                reasoning_puzzle:
                steps:
                    - step_number: 1
                    title: "Initial Information"
                    explanation: >
                        There are 3 people—Ann, Bob, and Chad—each wearing either a black or white hat.
                        There are exactly 2 black hats and 1 white hat available. Ann can see Bob and
                        Chad, Bob can see Chad, and Chad sees no one.
                    - step_number: 2
                    title: "Ann’s Perspective"
                    explanation: >
                        Ann looks at Bob and Chad’s hats. If she saw two white hats, she would know her
                        hat must be black, because there is only one white hat in total. However, she
                        states she does not know the color of her hat. Therefore, Bob and Chad cannot
                        both be wearing white hats.
                    - step_number: 3
                    title: "Bob’s Perspective"
                    explanation: >
                        Bob sees Chad’s hat. If Bob sees a white hat on Chad, then Bob would deduce his
                        own hat must be black (to prevent Ann from seeing two white hats). That would
                        allow Bob to know his hat color. However, Bob also says he does not know his
                        hat color, implying Chad’s hat cannot be white.
                    - step_number: 4
                    title: "Conclusion So Far"
                    explanation: >
                        From the above two observations, it follows that Chad’s hat must be black, because
                        Bob’s uncertainty eliminates the possibility of Chad wearing a white hat.
                    - step_number: 5
                    title: "Chad’s Reasoning"
                    explanation: >
                        Chad hears Ann didn’t see two whites (so at least one black hat is among Bob or
                        Chad) and that Bob couldn’t deduce his own hat color even after seeing Chad’s. If
                        Chad’s hat had been white, Bob would have concluded his own hat was black. Because
                        Bob remains uncertain, Chad realizes his own hat must be black.

                answer: "Chad’s hat is black."
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


# read and  convert png to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_of_bin_file("logo1.png")

st.markdown(
    f"""
    <style>
        .wipe-image {{
            animation: wipe 2s ease-out 1;  /* 2 seconds wipe effect, runs once then stops */
            overflow: hidden;
        }}
        .custom-title {{
            background: linear-gradient(#7f7fd5, #86a8e7, #91eae4);  /* Blue to purple gradient */
            -webkit-background-clip: text;
            color: transparent!important;  /* Force blue color with !important */
            font-weight: bold;  /* Optional: make the text bold */
            font-size: 48px;  /* Optional: change font size */
        }}
        @keyframes wipe {{
            0% {{
                clip-path: inset(0 100% 0 0);  /* Fully hidden from the right side */
            }}
            100% {{
                clip-path: inset(0 0 0 0);  /* Fully visible */
            }}
        }}
    </style>
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_base64}" alt="Logo" width="120" class="wipe-image"/> 
        <h1 class="custom-title">TuringPulse</h1>
        <p>Hi, I'm TuringPulse, your personal math assisstant!👋</p>
    </div>
    """,
    unsafe_allow_html=True
)



# Sidebar for API key input and model selection
with st.sidebar:
    st.title("🔧 Settings")
    model_provider = st.selectbox("Select Language Model Provider", ["OpenAI Series", "Anthropic Claude Series", "xAI Grok", "DeepSeek"])
    
    # Capture the API key based on the selected model provider
    if model_provider == "DeepSeek":
        st.info("DeepSeek is currently not supported due to regulations.")

    if model_provider == "OpenAI Series":
        # api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        model_name = st.selectbox("Select OpenAI Model", ["gpt-4o-mini", "o1-mini"])
    
        # Information links
        st.markdown("[Get an API Key](https://platform.openai.com/account/api-keys)")\

    if model_provider == "Anthropic Claude Series":
        model_name = st.selectbox("Select Claude Model", ["Claude 3 Haiku", "Claude 3.5 Sonnet v1"])
        st.info("These models are supported by AWS Bedrock. Due to token limitation, It cannot be requested many times in a short period of time. Please use it wisely.")

    # if model_provider == "Meta Llama Series":
    #     model_name = st.selectbox("Select Llama Model", ["Llama 3.2 3B", "Llama 3.3 70B"])
    #     st.info("These models are supported by AWS Bedrock. Due to token limitation, It cannot be requested many times in a short period of time. Please use it wisely.")

    if model_provider == "xAI Grok":
        st.info("The current model loaded is the latest version of Grok-2.")
        model_name = "grok-2-latest"



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

    elif model_provider == "OpenAI Series":
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model=model_name,
            messages=st.session_state.messages
        )
        msg = response.choices[0].message.content

    elif model_provider == "xAI Grok":
        openai.api_key = grok_api_key
        response = client_grok.chat.completions.create(
            model=model_name,
            messages=st.session_state.messages,
            temperature = 0
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
