from transformers import LongformerTokenizer, LongformerForQuestionAnswering
import torch

def answer_question_with_longformer(question, document):
    # Initialize Longformer tokenizer and model
    model_name = 'allenai/longformer-base-4096'
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    model = LongformerForQuestionAnswering.from_pretrained(model_name)

    # Tokenize question and document
    inputs = tokenizer(question, document, return_tensors='pt', truncation=True, max_length=4096)

    # Get input IDs and attention mask
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Set global attention on question tokens
    attention_mask[:, :len(tokenizer.tokenize(question)) + 2] = 2  # +2 for [CLS] and [SEP] tokens

    # Perform inference
    outputs = model(input_ids, attention_mask=attention_mask)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the most likely beginning and end of answer
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1

    # Decode the answer
    answer = tokenizer.decode(input_ids[0][start_index:end_index])

    return answer

# Example usage
context = "Spectroscopy encompasses a range of techniques that analyze the interaction between light and matter, providing insights into the composition, structure, and dynamics of biological systems. In biomedical engineering, these methods are pivotal for diagnostics, therapeutic monitoring, and research. Key Spectroscopic Techniques in Biomedical Engineering include Near-Infrared (NIR) Spectroscopy, Raman Spectroscopy, Diffuse Optical Spectroscopy (DOS), and Fluorescence Spectroscopy."
question = "What are the key spectroscopic techniques in biomedical engineering?"

answer = answer_question_with_longformer(question, context)
print("Answer:", answer)
