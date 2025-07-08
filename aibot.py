from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import os
from datetime import datetime

# Load model offline
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name, local_files_only=True)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name, local_files_only=True)

print("Welcome to AkanshBot (Emotional Edition)!")
print("Type 'exit' to quit.\n")

chat_history = []
log_file = "akanshbot_chatlog.txt"

def save_to_log(user_input, bot_reply):
    with open(log_file, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"You: {user_input}\n")
        f.write(f"AkanshBot: {bot_reply}\n\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("AkanshBot: Goodbye! Take care of your heart. 💙")
        break

    # Add to memory
    chat_history.append(f"Human: {user_input}")
    conversation = " ".join(chat_history[-6:])  # 3 turns of memory

    # Generate reply
    inputs = tokenizer([conversation], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    reply = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

    # Add emotional touch
    emotional_reply = reply + " ❤️"

    # Print and save
    print("AkanshBot:", emotional_reply)
    save_to_log(user_input, emotional_reply)
    chat_history.append(f"Bot: {reply}")
