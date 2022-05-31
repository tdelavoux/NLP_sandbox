from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

# -------------------------------> From
# https://huggingface.co/cmarkea/distilcamembert-base-sentiment


# -------------------------------> Save local from HUB 
# model = TFAutoModelForSequenceClassification.from_pretrained("cmarkea/distilcamembert-base-sentiment")
# model.save_pretrained("./models/cmarkea/distilcamembert-base-sentiment")

# tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base-sentiment")
# tokenizer.save_pretrained('./tokenizer/cmarkea/distilcamembert-base-sentiment/')


tokenizer = AutoTokenizer.from_pretrained('./tokenizer/cmarkea/distilcamembert-base-sentiment/', local_files_only=True)
model = TFAutoModelForSequenceClassification.from_pretrained('./models/cmarkea/distilcamembert-base-sentiment', local_files_only=True)

sequence = "Tu est le meilleur !"

nlp_sentiment = pipeline(task='sentiment-analysis', model=model, tokenizer=tokenizer)
result = nlp_sentiment(sequence,return_all_scores=True) # optional argument to get all scores 
print(result)