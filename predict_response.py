import nltk
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
import json
import pickle
import numpy as np
import random
import tensorflow
from data_preprocessing import get_stem_words
ignore_words=['?','!',',','.',"'s","'m"]
intents=json.loads(open("./intents.json").read())
words=pickle.load(open("./words.pkl","rb"))
classes=pickle.load(open("./classes.pkl","rb"))
model=tensorflow.keras.models.load_model("./chatbot_model.h5")
def preprocessing_user_input(user_input):
    input_word_token_1=nltk.word_tokenize(user_input)
    input_word_token_2=get_stem_words(input_word_token_1,ignore_words)
    input_word_token_2=sorted(list(set(input_word_token_2)))
    bag=[]
    bag_of_words=[]
    for word in words:
        if word in input_word_token_2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
        bag.append(bag_of_words)
        return np.array(bag)
def bot_class_predication(user_input):
    imp=preprocessing_user_input(user_input)
    predication=model.predict(imp)
    predicated_class_label=np.argmax(predication[0])
    return predicated_class_label
def bot_response(user_input):
    predicated_class_label=bot_class_predication(user_input)
    predicated_class=classes[predicated_class_label]
    for intent in intents["intents"]:
        if intent["tags"]==predicated_class:
            bot_response=random.choice(intent["responses"])
            return bot_response
print("Hi! I am Jarvis, how can I help you ?")
while True:
    user_input=input("Type your query")
    print("Use input:",user_input)
    response=bot_response(user_input)
    print("Bot_response:",response)