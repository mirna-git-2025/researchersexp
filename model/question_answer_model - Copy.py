from transformers import pipeline

# Load the pre-trained question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Define the context (the passage you want the model to read)
context = """
Artificial Intelligence (AI) has undergone significant transformations since its inception in the mid-20th century. 
The journey began with the development of early computing machines capable of performing tasks that required basic intelligence.
In 1956, the Dartmouth Conference marked the official birth of AI as a field of study, bringing together researchers interested in the possibility of creating machines that could "think."
The initial years were characterized by optimism, with researchers developing programs that could solve algebra problems, prove geometric theorems, and engage in rudimentary conversations.
"""

# Define the question
question = "What year did the Dartmouth Conference take place?"

# Get the answer from the model
result = qa_pipeline(question=question, context=context)

# Print the answer
print(f"Answer: {result['answer']}")
