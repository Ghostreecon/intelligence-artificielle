# -*- coding: utf8 -*-

import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import scrolledtext, font
from PIL import Image, ImageTk

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intent.json', encoding='utf-8').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def handle_unknown_message():
    # Réponse par défaut si le message n'est pas compris
    return "Je suis désolé, je ne comprends pas votre message. Pouvez-vous reformuler ou poser une autre question ?"


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
        else:
            result = handle_unknown_message()
    return result


# Initialisation des messages
messages = []


def send_message():
    # Récupération du message saisi
    chat_history.config(state=tk.NORMAL)
    message = user_input.get()

    chat_history.insert(tk.END, 'Vous: ' + message + '\n\n')
    chat_history.config(state=tk.DISABLED)

    # Prédiction de la réponse du chatbot
    ints = predict_class(message)
    res = get_response(ints, intents)

    # Ajout du message de l'utilisateur 2 à la liste des messages
    messages.append(('bot', res))

    # Affichage de la réponse du chatbot
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, 'Conseiller: '+ res + '\n\n', 'bot_message')
    chat_history.config(state=tk.DISABLED)

    # Effacer le champ de saisie
    user_input.delete(0, tk.END)


# Création de la fenêtre principale
root = tk.Tk()
root.title("Chatbot")
root.configure(bg="#a0a2cb")

# Ajout d'une icône à la fenêtre
app_icon = Image.open('images/chat_ca.png')
app_icon = ImageTk.PhotoImage(app_icon)
root.iconphoto(False, app_icon)

# Création du widget Text pour afficher l'historique des messages
