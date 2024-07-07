import pandas as pd
import chainlit as cl
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

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

# Input the Parenting Style and Major Challenges
parenting_style = input("Enter the Parenting Style: ")
major_challenges = input("Enter the Major Challenges: ")

df = pd.read_csv('PAIRING_COPING_SKILLS.csv')

# Filter the DataFrame based on Parenting Style and Major Challenges
filtered_df = df[(df['Parenting Style'] == parenting_style) & (df['Major Challenges'] == major_challenges)]

# Check if any rows match the filter criteria
if not filtered_df.empty:
    # Retrieve the Parenting Strategies and Coping Skills from the first matched row
    parenting_strategies = filtered_df.iloc[0]['Parenting Stratrgies']
    coping_skills = filtered_df.iloc[0]['Coping Skills']
    context = f"Parenting Style: {parenting_style}\nMajor Challenges: {major_challenges}\nParenting Strategies: {parenting_strategies}\nCoping Skills: {coping_skills}"
    question = input("Enter the question: ")
    answer = chain.invoke({'question': question, 'context': context})
    print(answer)
else:
    print("No matching data found.")