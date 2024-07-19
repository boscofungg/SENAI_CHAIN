import pandas as pd
import chainlit as cl

import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from translate import Translator

MODEL = "llama3"
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

parser = StrOutputParser()

chain = model | parser 

template = """
Answer the question based on the context below.
If you can't answer the question, reply "The answer to your question is not mentioned in the material you provided.".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
prompt.format(context="Here is some context", question="Here is a question")

chain = prompt | model | parser


translator= Translator(to_lang="zh-TW")
translation = translator.translate("I am trying to get rich and retire myself.")



@cl.on_chat_start
async def main():
    res = await cl.AskUserMessage(content="What is your Email?").send()
    email = res['output']
    df = pd.read_csv('USER_ID.csv')
    test = 0
    global context
    if email in df['Email'].values:
        # Retrieve the corresponding User ID
        user_id = df.loc[df['Email'] == email, 'User ID'].values[0]
        Parenting_Style = df.loc[df['Email'] == email, 'Parenting Styles'].values[0]
        strategies = df.loc[df['Email'] == email, 'Strategies'].values[0]
        coping_skill = df.loc[df['Email'] == email, 'Coping Skill'].values[0]
        sex = df.loc[df['Email'] == email, 'Child Sex'].values[0]
        age = df.loc[df['Email'] == email, "Child's Age"].values[0]
        test = 1
        context = f"Parenting Styles: {Parenting_Style}\nStrategies: {strategies}\nCoping Skill: {coping_skill}\nSex: {sex}\nAge: {age}\n"
        context += "Please provide some solutions based on the context given and please do not mention the parenting styles of the parent in your answer but state the child's age and sex."
    else:
        test = 0
    if test == 1:
        await cl.Message(
            content=f"Welcome back {user_id} !!",
        ).send()
    elif test == 0:
        await cl.Message(
            content=f"Welcome Guest!!",
        ).send()
        Parenting_Style = await cl.AskActionMessage(
            content="Pick an action!",
            actions=[
                cl.Action(name="Authoritative", value="Authoritative", label="✅Authoritative"),
                cl.Action(name="Authoritarian", value="Authoritarian", label="✅Authoritarian"),
                cl.Action(name="Permissive", value="Permissive", label="✅Permissive"),
                cl.Action(name="Uninvolved", value="Uninvolved", label="✅Uninvolved"),
            ],
        ).send()
        Major_Challenge = await cl.AskActionMessage(
            content="Pick an action!",
            actions=[
                cl.Action(name="Behaviour problem", value="Behaviour problem", label="❌Behaviour_problem"),
                cl.Action(name="Emotion issues", value="Emotion issues", label="❌Emotion_issues"),
                cl.Action(name="Attention deficit", value="Attention deficit", label="❌Attention_deficit"),
            ],
        ).send()
        Language = await cl.AskActionMessage(
            content="Pick a language!",
            actions=[
                cl.Action(name="Chinese", value="Chinese", label="❌Chinese"),
                cl.Action(name="English", value="English", label="❌English"),
            ],
        ).send()
        df = pd.read_csv('PAIRING_COPING_SKILLS.csv')

        # Filter the DataFrame based on Parenting Style and Major Challenges
        filtered_df = df[(df['Parenting Style'] == Parenting_Style.get("value")) & (df['Major Challenges'] == Major_Challenge.get("value"))]
        # Check if any rows match the filter criteria
        # Retrieve the Parenting Strategies and Coping Skills from the first matched row
        strategies = filtered_df.iloc[0]['Parenting Stratrgies']
        coping_skill = filtered_df.iloc[0]['Coping Skills']
        context = f"Parenting Style: {Parenting_Style}\nMajor Challenges: {Major_Challenge}\nParenting Strategies: {strategies}\nCoping Skills: {coping_skill}"
        context += "Please provide some solutions based on the context given."
@cl.on_message
async def on_message(message: cl.Message, streaming=True):
    questions = f'{message.content}'
    answer = chain.invoke({'question': questions, 'context': context})
    await cl.Message(answer).send()
