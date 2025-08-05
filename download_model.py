from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Choose a small, powerful model for Q&A
model_name = "google/flan-t5-base"

# Save locally to your models folder
save_directory = "./models/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Model saved to: {save_directory}")
