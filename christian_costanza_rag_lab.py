# -*- coding: utf-8 -*-
"""Christian_Costanza_rag_lab.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1d_VNPDJ-hLTbvJ9-4gAGWuS2oTq-G6-j

# Retrieval-Augmented Generation Lab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/S24-CS143AI/blob/main/rag_lab.ipynb)

In this lab, we'll combine code from our previous two labs on language models and semantic search to perform Retrieval-Augmented Generation - a strategy for providing large language models with information necessary to respond to user queries.

First, we need to install some libraries.
"""

import sys
!{sys.executable} -m pip install transformers sentence_transformers requests

"""## A variation on the chat bot from last week

Last time, we used the *conversational* pipeline, though I recently learned that this is being depreciated. Here's a similar way you can make a chatbot using the `text2test-generation` pipeline.
"""

from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-base")

conversation = "User: What is computer science?\n"
response = chatbot(conversation)
print(response)

conversation += "Assistant: "+response[0]["generated_text"]
conversation += "User: Does it draw from disciplines other than mathematics?"

response = chatbot(conversation)
print(response)

"""## Combining with Semantic Search

Now we are going to combine the chat bot code with the semantic search code from last time.

Here's the strategy:
* Perform semantic search on the user's query
* Collect the most relevant documents from the semantic search
* Prompt the chatbot with the question and the relevant documents

### Getting the data

The data we're using comes in JSON format with information about course names and descriptions.
"""

import requests
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline, Conversation
import numpy as np

def euclidean_distance(vec1, vec2):
    """Calculate the Euclidean distance between two vectors."""
    return np.linalg.norm(vec1 - vec2)

course_descriptions_url = 'https://raw.githubusercontent.com/ericmanley/S24-CS143AI/main/data/course_descriptions.json'
course_descriptions = requests.get(course_descriptions_url).json()

print("Here's a few of the course descriptions from our data:")
course_descriptions[:5]

"""### Preparing the data for the language model

We need to be able to feed text to the language models, so we'll concatenate the important parts of each record in the JSON data. For RAG, we usually refer to each chunk of text like this (each course) as a **document** even though it isn't a file - just a unit of text to consider as a whole.
"""

course_description_concat = []

for course in course_descriptions:
    curr_entry = course["course_code"]+course["course_name"]+course["description"]
    course_description_concat.append(curr_entry)

print("Here's what they will look like when we feed them to a language model:")
course_description_concat[:5]

"""### Creating the embeddings for each entry

Now we will create all of the embeddings for our course descriptions - this is the same as last time.

Typically, you would only do this once, when you're setting up the chatbot to begin with. You should save these to a file that the chatbot has access to - typically called an **index** or **vector database**.
"""

model = SentenceTransformer('all-MiniLM-L6-v2')

course_descriptions_embeddings = model.encode(course_description_concat)

"""### Performing the semantic search

This code just turns what we worked on last time into a function.

We'll pass the query, the course info documents, the embeddings of those documents, and the number of items we want returned.

It return a big string with all the relevant documents concatenated.
"""

def get_relevant_documents(query,documents,document_embeddings,n):

    # get the embedding for the query
    query_embedding = model.encode(query)

    # a list for sorting the document indices by distance to the query embedding
    distance_index_pairs = []

    for i in range(len(documents)):

        curr_dist = euclidean_distance(query_embedding,document_embeddings[i])
        distance_index_pairs.append( (curr_dist,i) )

    distance_index_pairs.sort()

    relevant_documents = ""
    for j in range(n):
        relevant_documents += documents[ distance_index_pairs[j][1]  ]

    return relevant_documents

query = "Which courses cover neural networks?"
relevant_courses = get_relevant_documents(query,course_description_concat,course_descriptions_embeddings,5)
print("Here's context that will be helpful in answering the question:",relevant_courses)

"""### Including the search results in the chat prompt

Now we can take the information we found with semantic search and include it in the prompt for the chat.
"""

instructions = "Answer the user's question using this context: "+relevant_courses
question = "User: "+query
conversation = instructions+question
response = chatbot(conversation,max_new_tokens=200)
display(response)

"""## Discussion

We passed 527 tokens/words to the model, but it has an input limit of 128. What are some things we can do to try and fix this problem? Come up with at least two strategies and enter them on the shared slide for your group.

## Exercise: Try another model

Try this with the following model, which allows for larger input but is still small enough to run on Colab: https://huggingface.co/google/flan-t5-base
"""

query = "Which courses cover calculus?"
relevant_courses = get_relevant_documents(query,course_description_concat,course_descriptions_embeddings,5)
print("Here's context that will be helpful in answering the question: ", relevant_courses)

instructions = "Answer the user's question using this context: "+relevant_courses
question = "User: "+query
conversation = instructions+question
response = chatbot(conversation,max_new_tokens=200)
display(response)

"""## Exercise: Put this in a loop

Build the chat bot into a loop that will allow the user to continually ask follow-up questions. Try testing it with questions like "What other topics does that course cover?"
"""

x = 1
while x != 0:
  query = input("Input a query (enter 0 to end)")
  if query == "0":
    x = 0
  else:
    relevant_courses = get_relevant_documents(query,course_description_concat,course_descriptions_embeddings,5)
    instructions = "Answer the user's question using this context: "+relevant_courses
    question = "User: "+query
    conversation = instructions+question
    response = chatbot(conversation,do_sample = True, temperature = 3.5)
    display(response)

"""## Exercise: Experiment with the temperature

You can set the `do_sample` and `temperature` parameters to affect how random the output is. Setting `do_sample=True` will allow it to use some randomness in generating output. The `temperature` affects how random it allows the output to be. Experiment with different temperature values and determine which value you're happiest with.
"""

conversation = "User: What is computer science?"
response = chatbot(conversation,do_sample=True,temperature= 1)
display(response)

x = 1
while x != 0:
  query = input("Input a query (enter 0 to end)")
  if query == "0":
    x = 0
  else:
    relevant_courses = get_relevant_documents(query,course_description_concat,course_descriptions_embeddings,5)
    instructions = "Answer the user's question using this context: "+relevant_courses
    question = "User: "+query
    conversation = instructions+question
    response = chatbot(conversation,do_sample = True, temperature = 3.5)
    display(response)