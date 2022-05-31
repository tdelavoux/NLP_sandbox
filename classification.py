from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

# -------------------------------> From
# https://huggingface.co/cmarkea/distilcamembert-base-nli


# -------------------------------> Save local from HUB 
# model = TFAutoModelForSequenceClassification.from_pretrained("cmarkea/distilcamembert-base-nli")
# model.save_pretrained("./models/cmarkea/cmarkea_distilcamembert-base-nli")

# tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base-nli")
# tokenizer.save_pretrained('./tokenizer/cmarkea/cmarkea_distilcamembert-base-nli')


tokenizer = AutoTokenizer.from_pretrained('./tokenizer/cmarkea/cmarkea_distilcamembert-base-nli', local_files_only=True, model_max_length=512)
model = TFAutoModelForSequenceClassification.from_pretrained('./models/cmarkea/cmarkea_distilcamembert-base-nli', local_files_only=True)

sequences = "Je souhaites réaliser une demande de réaménagement de prêt"

nlp_zsc = pipeline(task='zero-shot-classification', model=model, tokenizer=tokenizer)
result = nlp_zsc(sequences=sequences, candidate_labels="Derog Habitat, Derog Reamenagement, EasySign")
print(result)