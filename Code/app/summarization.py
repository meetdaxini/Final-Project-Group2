import nltk
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# generate chunks of text \ sentences <= 1024 tokens
def nest_sentences(document):
  """
  https://discuss.huggingface.co/t/summarization-on-long-documents/920/7
  """
  nested = []
  sent = []
  length = 0
  for sentence in nltk.sent_tokenize(document):
    length += len(sentence)
    if length < 1024:
      sent.append(sentence)
    else:
      nested.append(sent)
      sent = []
      length = 0

  if sent:
    nested.append(sent)
  return nested

# generate summary on text with <= 1024 tokens
def generate_summary(text, tokenizer, model, max_tokens, min_tokens, length_penalty=3):
  nested_sentences = nest_sentences(text)
  summaries = []
  for nested in nested_sentences:
    input_tokenized = tokenizer.encode(' '.join(nested), truncation=True, return_tensors='pt')
    input_tokenized = input_tokenized.to(device)
    summary_ids = model.generate(input_tokenized,
                                      length_penalty=length_penalty,
                                      min_length=min_tokens,
                                      max_length=max_tokens)
    output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    summaries.append(output)
  summaries = [sentence for sublist in summaries for sentence in sublist]
#   if regenerate_summar:
#     nested_sentences = nest_sentences(" ".join(summaries))
#     summaries = generate_summary(text, tokenizer, model, max_tokens, min_tokens, length_penalty=length_penalty, regenerate_summar=False)
  return " ".join(summaries)


