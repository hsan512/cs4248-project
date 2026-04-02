from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load original
model_name = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Add your custom "Signal" tokens
new_tokens = ["<USER>", "<URL>"]
tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

# Resize the embedding matrix to make room for 2 new rows
model.resize_token_embeddings(len(tokenizer))

# Save this to a local folder
model.save_pretrained("./roberta-expanded")
tokenizer.save_pretrained("./roberta-expanded")