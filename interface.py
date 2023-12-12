import random
import time
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import scrolledtext
from tkinter import font
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
def typing_indicator():
    chat_history.insert(tk.END, 'Conseiller est en train d\'écrire...')
    chat_history.see(tk.END)
    time.sleep(0.5)
    chat_history.delete("end-1l", tk.END)


def send_message():
    # Récupération du message saisi
    chat_history.config(state=tk.NORMAL)
    message = user_input.get()
    chat_history.insert(tk.END, 'Vous: ' + message + '\n\n')

    # Affichage du typing indicator pendant 0.5s
    chat_history.after(0, typing_indicator)

    # Prédiction de la réponse du chatbot
    ints = predict_class(message)
    res = get_response(ints, intents)

    # Ajout du message de l'utilisateur 2 à la liste des messages
    messages.append(('bot', res))

    # Affichage de la réponse du chatbot
    chat_history.insert(tk.END, 'Conseiller: '+ res + '\n\n', 'bot_message')
    chat_history.config(state=tk.DISABLED)
    # Effacer le champ de saisie
    user_input.delete(0, tk.END)


# Création de la fenêtre principale
root = tk.Tk()
root.title("Chatbot")
font = font.Font(family="Helvetica", size=12)
bg_color = "#a0a2cb"
app_icon = Image.open('images/chat_ca.png')
app_icon = ImageTk.PhotoImage(app_icon)

root.iconphoto(False, app_icon)

# Création du widget Text pour afficher l'historique des messages
chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20, font=("Helvetica", 12))
chat_history.tag_config('bot_message', justify='right')
chat_history.tag_config('bot_message', justify='right')
chat_history.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5)

# Créer un cadre pour la zone de saisie du chat de l'utilisateur 1
user1_input_frame = tk.Frame(root, bg=bg_color)
user1_input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

# Créer une entrée pour la saisie de texte de l'utilisateur 1
user_input = tk.Entry(user1_input_frame, font=font, bg="white", fg="black", relief="flat",borderwidth=2, highlightthickness=0)
user_input.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)

# Créer un bouton pour envoyer le message de l'utilisateur 1
user1_send_button = tk.Button(user1_input_frame, text="Envoyer", font="helvetica 10 bold", bg="#00af9c", fg="white", activebackground="#00af9c", activeforeground="white", relief="flat", borderwidth=0, highlightthickness=0, command=send_message)
user1_send_button.pack(side=tk.RIGHT, padx=5, pady=5)


root.mainloop()
