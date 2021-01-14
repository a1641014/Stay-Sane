import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from pathlib import Path 
from PIL import Image
import streamlit as st
import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, Dense, Activation,RepeatVector, Bidirectional, Dropout, merge, concatenate, TimeDistributed, GRU
from keras.optimizers import Adam, SGD
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
import re

#paths
img_path = Path.joinpath(Path.cwd(),'images')
projfiles_path=Path.joinpath(Path.cwd(),'project-files')
checkpoint_path=Path.joinpath(projfiles_path,'training_checkpoints_100')

#MODEL

#reading preprocessed files
with open(Path.joinpath(projfiles_path,'AI_project_inpdata.txt'), 'r') as f:
  input_vect = f.read()
with open(Path.joinpath(projfiles_path,'AI_project_outdata.txt'), 'r') as f1:
  output_vect = f1.read()    
input_vect=input_vect.split("\n")
output_vect=output_vect.split("\n")

#Tokenizer
from keras.preprocessing.text import Tokenizer
def create_vocab(source,target):
    word2index={}
    index2word={}
    tokenizer=Tokenizer(oov_token='oov',filters='"#$%&()*+,/:;<=>@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(source+target)
    dictionary=tokenizer.word_index
    for key,val in dictionary.items():
        word2index[key]=val
        index2word[val]=key
    return tokenizer,word2index,index2word
tokenizer,word2index,index2word=create_vocab(input_vect,output_vect)

#encoder decoder inputs
encoder_seq=np.load(Path.joinpath(projfiles_path,'encoder-input-after-padding.npy'))
decoder_seq=np.load(Path.joinpath(projfiles_path,'decoder-input-after-padding.npy'))

#hyperparamters
maxlen_encoder=30           
maxlen_decoder=30         
embed_dim=100              
num_units=256              
epochs_no=30                
batch_no=64  

#function for padding the encoder and decoder inputs  
from keras.preprocessing.sequence import pad_sequences
def padding(source,target):
    encoder_seq=pad_sequences(source, dtype='int16',maxlen=30,padding='post',truncating='post')
    decoder_seq=pad_sequences(target, dtype='int16', maxlen=30,padding='post',truncating='post')
    return encoder_seq, decoder_seq

#creating embedding index using glove.6B.100
def create_embedding_index():
    embedding_index={}
    with open(Path.joinpath(projfiles_path,'glove.6B.100d.txt'), encoding='utf=8') as f:
        for line in f:
            values=line.split()
            word=values[0]
            embed_vector=np.array(values[1:],dtype='float32')
            embedding_index[word]=embed_vector
        return embedding_index

embedding_index=create_embedding_index()
embed_size=len(list(embedding_index.items())[0][1])

#creating embedding matrix
def create_embedding_mat(vocabulary):
    embedding_dim=100
    not_found=[]
    embedding_matrix=np.zeros((len(vocabulary)+1,embedding_dim),dtype="int16")
    for key,val in vocabulary.items():
        value=embedding_index.get(key)
        if value is not None:
            embedding_matrix[val]=value
        else:
            not_found.append(val)
    return embedding_matrix, not_found

embedding_matrix, not_found=create_embedding_mat(word2index)

#vocab size
vocab_size=len(word2index)+1 

#creating training and validation sets
from sklearn.model_selection import train_test_split
encoder_seq_train, encoder_seq_val, decoder_seq_train, decoder_seq_val = train_test_split(encoder_seq, decoder_seq, test_size=0.1, random_state=500)

#creating tf.data dataset
train_size=len(encoder_seq_train)
steps_per_epoch=len(encoder_seq_train)//batch_no
dataset=tf.data.Dataset.from_tensor_slices((encoder_seq_train, decoder_seq_train)).shuffle(train_size)

#creating batch dataset
dataset=dataset.batch(batch_no,drop_remainder=True)

#Encoder 
class Encoder(Model):
  def __init__(self,vocab_size,embed_dim,enc_units,batch_size):
    super(Encoder,self).__init__()
    self.batch_size=batch_size
    self.enc_units=enc_units
    self.embedding=Embedding(vocab_size,embed_dim)
    self.gru=GRU(self.enc_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')

  def call(self,x,hidden):
    x=self.embedding(x)
    output,state=self.gru(x,initial_state=hidden)
    return output,state
  
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_size,self.enc_units))

encoder = Encoder(vocab_size, embed_dim, num_units, batch_no)

#Attention
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self,units):
    super(BahdanauAttention,self).__init__()
    self.W1=Dense(units)
    self.W2=Dense(units)
    self.V=Dense(1)
  
  def call(self,query,values):
    query_curr_time_step=tf.expand_dims(query,1)
    score=self.V(tf.nn.tanh(self.W1(query_curr_time_step)+self.W2(values)))
    attention_weights=tf.nn.softmax(score,axis=1)
    context_vector=attention_weights*values
    context_vector=tf.reduce_sum(context_vector,axis=1)
    return context_vector,attention_weights

#Decoder
class Decoder(Model):
  def __init__(self, vocab_size, embed_dim, dec_units, batch_size):
    super(Decoder,self).__init__()
    self.batch_size=batch_size
    self.dec_units=dec_units
    self.embedding=Embedding(vocab_size,embed_dim)
    self.gru=GRU(self.dec_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
    self.fc_dense=Dense(vocab_size)
    self.attention=BahdanauAttention(self.dec_units)

  def call(self,x,hidden,enc_output):
    context_vector,attention_weights=self.attention(hidden,enc_output)
    x=self.embedding(x)
    x=tf.concat([tf.expand_dims(context_vector,1),x], axis=-1)
    output,state=self.gru(x)      
    output=tf.reshape(output,(-1,output.shape[2]))
    x=self.fc_dense(output)
    return x,state,attention_weights

decoder = Decoder(vocab_size, embed_dim, num_units, batch_no)

#optimizr and loss fn
optimizer=Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

#iterator and checkpoint
iterator=iter(dataset)
ckpt=tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, encoder=encoder, decoder=decoder,iterator=iterator)
manager=tf.train.CheckpointManager(ckpt,"./tf_ckpts",max_to_keep=3)

#restoring model
path_checkpoint="C:\\Users\\ashwi\\anaconda3\\envs\\staysane_demo\\project-files\\training_checkpoints_100\\ckpt-58.index"
ckpt.restore(path_checkpoint).expect_partial()

#APP

#background image
page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1468872961186-1d26f74f3355?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

#sidebar content
st.sidebar.title("""STAY SANE""")
about='''STAY SANE is an Automated platform for helping people suffering
 from mental issues. You can really open up to us to get rid of your depression. We are always with you :
   You don’t have to be positive all the time. It’s perfectly okay to feel sad, angry, annoyed, frustrated, 
   scared and anxious. Having feelings doesn’t make you a negative person. It makes you human.'''

if st.sidebar.button('About Us'):    
    st.sidebar.write(about)

import webbrowser

url = 'https://www.helpguide.org/articles/depression/depression-treatment.htm'

if st.sidebar.button('HelpGuide'):
    webbrowser.open_new_tab(url)

#load images 
center = Image.open(Path.joinpath(img_path,'mind3.jpg'))
side1 = Image.open(Path.joinpath(img_path,'life.jpg'))
side2=Image.open(Path.joinpath(img_path,'mind1.jpg'))
st.sidebar.image(center,width=310)
st.sidebar.image(side1,width=310)
st.sidebar.image(side2,width=310)

#main content
st.markdown("<h1 style='text-align: center; color: white;'><b>STAY SANE</b></h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: left; color: white;'><b>How do you feel now?  Share with me</b></h3>", unsafe_allow_html=True)

def check(input):
  if input!="":
    return True
  else:
    return False

#defining contraction words' dctionary for transformation
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", 
                    "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                    "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", 
                    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would",
                    "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                    "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", 
                    "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", 
                    "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                    "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", 
                    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
                    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                    "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", 
                    "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
                    "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                    "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", 
                    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", 
                    "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", 
                    "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is",
                    "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", 
                    "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
                    "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                    "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have",
                    "y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", 
                    "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

#preprocessing user text
def preprocess(data):
  temp=re.sub(' ’ |’| \' ','\'',data) #changing ’ to '
  temp=temp.strip() #removing beginning and tailing whitespaces
  temp=re.sub('[^A-Za-z0-9\s.,?]+','', temp) #Removing chracters except alphanumeric and ? ! . , '
  temp=temp.lower() #coverting to lower case
  temp=re.sub('[\?]+(?=[\?])|[\!]+(?=[\!])|[\']+(?=[\'])|[\.]+(?=[\.])','', temp) #removing occurences of consecutive special characters
  temp=re.sub('\?',' ? ',temp) 
  temp=re.sub('\!',' ! ',temp)
  temp=re.sub('\.',' . ',temp)
  temp=re.sub('[\s]+(?=[\s])','',temp)#these 4 lines ensures a single whitespace at the beginning and the end of 
                                                      #special characters and words.
  for key,val in contraction_dict.items():
    temp=re.sub(key.lower(),val.lower(),temp)
  if len(temp)>0:
    return temp

def generate(user_input):
  print(len(user_input))
  user_input1=preprocess(user_input)
  print(len(user_input1))
  user_input1=tokenizer.texts_to_sequences(user_input1)
  print(user_input1)
  user_input2=[int(i[0]) for i in user_input1 if len(i)>0]
  user_input2=np.expand_dims(user_input2, axis=0)
  print(user_input2)
  user_input2=pad_sequences(user_input2, dtype='int16',maxlen=30,padding='post',truncating='post')
  print(user_input2)
  inputs=tf.convert_to_tensor(user_input2)
  print(inputs.shape)
  result=''
  hidden=[tf.zeros((1,num_units))]
  enc_out,enc_hidden_state=encoder(inputs,hidden)
  dec_hidden=enc_hidden_state
  dec_input=tf.expand_dims([word2index['bos']],0)
  for t in range(maxlen_decoder):
    predictions,dec_hidden,attention_weights=decoder(dec_input,dec_hidden,enc_out)
    predicted_id=tf.argmax(predictions[0]).numpy()
    result+=index2word[predicted_id]+' '
    if index2word[predicted_id]=='eos':
      break
    dec_input=tf.expand_dims([predicted_id],0)
  return result

st.text_area("Your Friend:", value="Hello! I am a chatbot trained to converse with you. Hope we have a great time!", max_chars=None, key=None)
st.text_area("Your Friend:", value="So let's begin! Can you tell me where you are from?", max_chars=None, key=None)
user_input = st.text_input("You:")
if(check(user_input)):
  st.text_area("Bot__:", value=generate(user_input), height=30, max_chars=None, key=None)

'''
score=[]
def resbot(user_input,start=True):
    score.append(user_input)
    if start:
        response = "Hi, I'm happy to have you here \nI have a lot to discuss with you"
        start=False
    else:  
        response="type something"      
    return response    

def resp():
    if user_input=='hello':
        t=resbot(user_input)
    else:
        t=resbot(user_input,start=False)
    return t

user_input = st.text_input("user")  
st.text_area("Bot:", value=resp(), height=100, max_chars=None, key=None)
user_input = st.text_input("user_")
st.text_area("Bot_:", value=resp(),  height=100, max_chars=None, key=None)
user_input = st.text_input("User_")
st.text_area("Bot__:", value=resp(), height=100, max_chars=None, key=None)
"""

def fun():
    st.sidebar.write(score)
    return

if st.sidebar.button('Get user responses'):
    fun()
'''





#user_input = get_text()
#if str(user_input) =="type here":
#   response = botResponse(user_input)
#else:
#    response = botResponse(user_input,is_startup=False)