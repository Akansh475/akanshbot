from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Memory: holds the conversation so far
chat_history = []

def generate_reply(user_input):
    # Add user's message to history
    chat_history.append(f"You: {user_input}")

    # Create the conversation string
    conversation = "\n".join(chat_history[-6:])  # Keep last 6 lines (adjustable)

    # Tokenize and get model reply
    inputs = tokenizer([conversation], return_tensors="pt", truncation=True, max_length=512)
    reply_ids = model.generate(**inputs)
    reply = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

    # Add bot's reply to history
    chat_history.append(f"AkanshBot: {reply}")

    return reply
