from transformers import AutoTokenizer, TFAutoModelForTokenClassification
from transformers import pipeline

# -------------------------------> From
# https://huggingface.co/gilf/french-camembert-postag-model


# -------------------------------> Save local from HUB 
# model = TFAutoModelForTokenClassification.from_pretrained("gilf/french-camembert-postag-model")
# model.save_pretrained("./models/gilf/french-camembert-postag-model/")

# tokenizer = AutoTokenizer.from_pretrained("gilf/french-camembert-postag-model")
# tokenizer.save_pretrained('./tokenizer/gilf/french-camembert-postag-model/')


tokenizer = AutoTokenizer.from_pretrained('./tokenizer/gilf/french-camembert-postag-model/', local_files_only=True, model_max_length=512)
model = TFAutoModelForTokenClassification.from_pretrained('./models/gilf/model/', local_files_only=True)

sequence = "Tu est le meilleur !"

nlp_token_class = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)
result = nlp_token_class(sequence)
print(result)