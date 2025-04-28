from transformers import pipeline

def answer_question(question, document):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    answer = qa_pipeline({"question": question, "context": document})
    return answer["answer"]
