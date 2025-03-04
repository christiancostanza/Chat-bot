# Retrieval-Augmented Generation (RAG) Lab  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/S24-CS143AI/blob/main/rag_lab.ipynb)  

## Overview  
This project demonstrates **Retrieval-Augmented Generation (RAG)**—a technique that enhances large language models (LLMs) by incorporating **semantic search** to retrieve relevant documents before generating responses.  

By integrating **semantic search** with **FLAN-T5**, this chatbot can provide accurate, context-aware answers based on a dataset of course descriptions.  

---

##  Features  
 **Conversational AI**: Implements a chatbot using the **FLAN-T5** model.  
 **Semantic Search**: Retrieves the most relevant course descriptions for a given query.  
 **Retrieval-Augmented Generation**: Enhances chatbot responses with factual context.  
 **Interactive Query Loop**: Users can continuously ask follow-up questions.  
 **Adjustable Sampling & Temperature**: Fine-tune response randomness.  

---

##  Installation  

###  Prerequisites  
Ensure you have **Python 3.8+** installed.  

###  Setup  
Clone this repository and install the necessary dependencies:  

```bash
git clone <repository-url>
cd <project-folder>
pip install transformers sentence_transformers requests numpy
```

If using **Google Colab**, install dependencies within the notebook:  
```python
!pip install transformers sentence_transformers requests numpy
```

---

##  Dataset  
- **Source**: Course descriptions dataset from [GitHub](https://raw.githubusercontent.com/ericmanley/S24-CS143AI/main/data/course_descriptions.json).  
- **Format**: JSON file containing `course_code`, `course_name`, and `description`.  
- **Use Case**: Provides structured course details for semantic search and chatbot responses.  

---

##  Methodology  

###  Data Retrieval & Preprocessing  
- Load course descriptions from a JSON file.  
- Concatenate course details (`course_code`, `course_name`, `description`) into single text entries.  

###  Creating Text Embeddings  
- Use **SentenceTransformer ('all-MiniLM-L6-v2')** to generate numerical representations of course descriptions.  
- Store embeddings for efficient **semantic search**.  

###  Semantic Search  
- Compute **Euclidean distance** between **user queries** and stored **embeddings**.  
- Retrieve **top N most relevant** course descriptions.  

###  Chatbot Integration  
- Uses **FLAN-T5** (`text2text-generation` pipeline).  
- Enhances chatbot responses by injecting relevant **course descriptions** into the prompt.  

###  Interactive Chatbot Loop  
- Users can ask multiple questions.  
- Implements **temperature tuning** for response variability.  

---

##  Usage  

###  Running the chatbot  
To start the chatbot in an **interactive mode**, run:  
```python
python rag_chatbot.py
```

###  Example Query  
```
Input a query: Which courses cover neural networks?
```
**Bot Output:**  
```
Here's context that will be helpful in answering the question: 
CS143 - Introduction to AI: Covers neural networks, deep learning, and machine learning techniques.
```

###  Experimenting with Temperature  
Adjust the `temperature` parameter for more diverse responses:  
```python
response = chatbot(conversation, do_sample=True, temperature=2.0)
```
- **Lower values (e.g., `0.5`)** = More deterministic responses.  
- **Higher values (e.g., `3.5`)** = More creative/random responses.  

###  Continuous Chat Mode  
The chatbot runs in a loop until the user exits (`0`).  
```python
while True:
    query = input("Enter a question (or 0 to exit): ")
    if query == "0":
        break
    response = chatbot(query)
    print(response)
```

---

##  Performance & Improvements  

###  **Challenges**  
- The FLAN-T5 model has an **input limit of 128 tokens**, which restricts the amount of context it can process.  
- Current **Euclidean distance ranking** may not always retrieve the most **semantically relevant** courses.  

###  **Future Enhancements**  
 Store embeddings in a **vector database** (e.g., FAISS) for optimized retrieval.  
 Implement **BM25 ranking** for better search accuracy.  
 Compare **FLAN-T5 vs. other LLMs** for improved responses.  

---

##  Contributors  
- **Christian Costanza**  


