import torch
import pandas as pd
from datasets import load_dataset
from evaluate import load
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')
model.load_state_dict(torch.load('best_model_Multi_News_final.pt'))
model.to(device)

def generate_summaries(texts, tokenizer, model, max_length=200, num_beams=4):
    inputs = tokenizer(texts, max_length=1024, return_tensors='pt', truncation=True, padding=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=num_beams)
    generated_summaries = tokenizer.batch_decode(output, skip_special_tokens=True)
    return generated_summaries

# Example usage
texts = ["The study, led by scientists from renowned institutions including the University of California, Davis, and the National Center for Ecological Analysis and Synthesis, analyzed data from over 100 research expeditions spanning the past three decades. Their results revealed a staggering 30% increase in plastic pollution since the 1990s, with microplastics being the predominant type of debris detected. Dr. Emily Patel, lead author of the study, expressed grave concern over the implications of these findings, emphasizing the detrimental impact of plastic pollution on marine ecosystems and wildlife. Our oceans are inundated with plastic waste, posing a severe threat to marine life and human health, Dr.Patel stated.The study's findings underscore the pressing need for concerted efforts to reduce plastic consumption, improve waste management systems, and implement policies to curb plastic pollution globally. Failure to address this issue urgently could have dire consequences for marine biodiversity and ecosystem functioning, with far-reaching implications for future generations. Environmental advocates are calling for immediate action from governments, industries, and individuals to reverse this alarming trend and safeguard the health of our oceans."]
generated_summaries = generate_summaries(texts, tokenizer, model, max_length=200, num_beams=4)
print(generated_summaries)