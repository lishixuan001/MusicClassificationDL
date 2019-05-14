import keras
from keras_self_attention import SeqSelfAttention

class AudioLSTMModel():
    def __init__(self, num_lstm_layers, LSTM_hidden_state_dim, num_time_frames, num_freq_bins, num_genres, lyrics_embedding_dimension, use_attention):
        
        self.num_time_frames = num_time_frames

        self.chroma_input = Input(shape=(num_frames, num_freq_bins))
        self.mfcc_input = Input(shape=(num_frames, num_freq_bins))
        self.lyrics_embedding_input = Input(shape=(embedding_dimension,))
        self.use_attention = use_attention
        
        
        # chroma LSTM and attention
        chroma = keras.layers.LSTM(units=LSTM_hidden_state_dim, return_state=True, input_shape=(self.num_frames, self.num_freq_bins))(self.chroma_input)
        for _ in range(num_lstm_layers - 1):
            chroma = keras.layers.LSTM(units=LSTM_hidden_state_dim, return_state=True)(chroma)
        
        if self.use_attention:
            chroma = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL, name='AttentionChroma')(chroma)
        
        # MFCC LSTM and attetion
        mfcc = keras.layers.LSTM(units=LSTM_hidden_state_dim, return_state=True, input_shape=(self.num_frames, self.num_freq_bins))(self.mfcc_input)
        for _ in range(num_lstm_layers - 1):
            mfcc = keras.layers.LSTM(units=LSTM_hidden_state_dim, return_state=True)(mfcc)
        
        if self.use_attention:
            mfcc = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL, name='AttentionMFCC')(mfcc)
                
        
        # concatinating the mfcc chroma and embedding features
        x = concatenate([mfcc, chroma, self.embedding_input])

        # putting the features through a couple fcnn. 
        x = Dense(2048, activation='relu')(x)
        x = Dense(2048, activation='relu')(x)
        
        # the embedding vector for each song. 
        latent_embedding = Dense(50, activation='relu', name='embedding')(x)
        genre = Dense(num_genres, activation='softmax')(latent_embedding)
        self.net = Model(inputs=[self.chroma_input, self.mfcc_input, self.embedding_input], outputs=genre)
        self.embedding = Model(self.net.input, outputs=self.net.get_layer('embedding').output)