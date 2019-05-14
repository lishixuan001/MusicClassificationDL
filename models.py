import tensorflow as tf
import numpy as np
import keras
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Lambda, concatenate, Input, GlobalMaxPooling1D, GlobalAveragePooling1D
import keras.backend as K
from keras.regularizers import l1_l2
from keras_self_attention import SeqSelfAttention


def l2_norm(x, axis):
    x = x ** 2
    x = K.sum(x, axis=axis)
    x = K.sqrt(x)
    return x

class AudioCNNModel():
    def __init__(self, num_frames, num_freq_bins, num_conv_filters1, pool_size_1, kernel_size, num_genres, embedding_dimension, lambda1=0.0, lambda2=0.0):

        self.num_frames = num_frames
        self.num_freq_bins = num_freq_bins
        self.num_conv_filters1 = num_conv_filters1
        self.pool_size1 = pool_size_1
        self.kernel_size = kernel_size
        self.chroma_input = Input(shape=(num_frames, num_freq_bins))
        self.mfcc_input = Input(shape=(num_frames, num_freq_bins))
        self.embedding_input = Input(shape=(embedding_dimension,))

        chroma = Conv1D(filters=self.num_conv_filters1, kernel_size=self.kernel_size, input_shape=(self.num_frames, self.num_freq_bins))(self.chroma_input)
        chroma = MaxPooling1D(pool_size=self.pool_size1)(chroma)
        chroma = Conv1D(filters=self.num_conv_filters1, kernel_size=self.kernel_size)(chroma)
        chroma = MaxPooling1D(pool_size=2)(chroma)
        chroma = Conv1D(filters=2*self.num_conv_filters1, kernel_size=self.kernel_size)(chroma)
        chroma = MaxPooling1D(pool_size=2)(chroma)

#       temporal pooling, L2, mean
        # max_layer = GlobalMaxPooling1D(data_format='channels_last')(chroma)
        # mean_layer = GlobalAveragePooling1D(data_format='channels_last')(chroma)
        # L2_layer = Lambda(lambda x: self.l2_norm(x, 1))(chroma)
        # #TODO:concatenate
        # chroma = concatenate([max_layer, mean_layer, L2_layer])
        chroma = GlobalAveragePooling1D(data_format='channels_last')(chroma)

        mfcc = Conv1D(filters=self.num_conv_filters1, kernel_size=self.kernel_size, input_shape=(self.num_frames, self.num_freq_bins))(self.mfcc_input)
        mfcc = MaxPooling1D(pool_size=self.pool_size1)(mfcc)
        mfcc = Conv1D(filters=self.num_conv_filters1, kernel_size=self.kernel_size)(mfcc)
        mfcc = MaxPooling1D(pool_size=2)(mfcc)
        mfcc = Conv1D(filters=2*self.num_conv_filters1, kernel_size=self.kernel_size)(mfcc)
        mfcc = MaxPooling1D(pool_size=2)(mfcc)

#       temporal pooling, L2, mean
        # max_layer = GlobalMaxPooling1D(data_format='channels_last')(mfcc)
        # mean_layer = GlobalAveragePooling1D(data_format='channels_last')(mfcc)
        # L2_layer = Lambda(lambda x: self.l2_norm(x, 1))(mfcc)
        # #TODO:concatenate
        # mfcc = concatenate([max_layer, mean_layer, L2_layer])
        mfcc = GlobalAveragePooling1D(data_format='channels_last')(mfcc)

        x = concatenate([mfcc, chroma, self.embedding_input])
#         x = self.embedding_input
        #End
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(x)
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(x)
        latent_embedding = Dense(50, activation='relu', name='embedding', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(x)
        genre = Dense(num_genres, activation='softmax', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(latent_embedding)
        self.net = Model(inputs=[self.chroma_input, self.mfcc_input, self.embedding_input], outputs=genre)
        self.embedding = Model(self.net.input, outputs=self.net.get_layer('embedding').output)

    def l2_norm(self, x, axis):
        x = x ** 2
        x = K.sum(x, axis=axis)
        x = K.sqrt(x)
        return x

class AudioLSTMModel():
    def __init__(self, num_lstm_layers, LSTM_hidden_state_dim, num_frames, num_freq_bins, num_genres, lyrics_embedding_dimension, use_attention, lambda1=0.0, lambda2=0.0):

        self.num_frames = num_frames
        self.num_freq_bins = num_freq_bins

        self.chroma_input = Input(shape=(num_frames, num_freq_bins))
        self.mfcc_input = Input(shape=(num_frames, num_freq_bins))
        self.lyrics_embedding_input = Input(shape=(lyrics_embedding_dimension,))
        self.use_attention = use_attention


        # chroma LSTM and attention
        chroma = keras.layers.LSTM(units=LSTM_hidden_state_dim, return_sequences=True, input_shape=(self.num_frames, self.num_freq_bins))(self.chroma_input)
        for i in range(num_lstm_layers-2):
            chroma = keras.layers.LSTM(units=LSTM_hidden_state_dim, return_sequences=True)(chroma)
        chroma = keras.layers.LSTM(units=LSTM_hidden_state_dim, return_sequences=self.use_attention)(chroma)


        if self.use_attention:
            print(chroma)
            chroma = SeqSelfAttention(attention_activation='sigmoid', attention_width=15, name='AttentionChroma')(chroma)
            chroma = Dense(10)(chroma)

        print(chroma)

        # MFCC LSTM and attetion
        mfcc = keras.layers.LSTM(units=LSTM_hidden_state_dim, return_sequences=True, input_shape=(self.num_frames, self.num_freq_bins))(self.mfcc_input)
        for _ in range(num_lstm_layers - 2):
            mfcc = keras.layers.LSTM(units=LSTM_hidden_state_dim, return_sequences=True)(mfcc)
        mfcc = keras.layers.LSTM(units=LSTM_hidden_state_dim, return_sequences=self.use_attention)(mfcc)

        if self.use_attention:
            mfcc = SeqSelfAttention(attention_activation='sigmoid', attention_width=15, name='AttentionMFCC')(mfcc)
            mfcc=Dense(10)(mfcc)
            print(mfcc)


        # concatinating the mfcc chroma and embedding features
        x = concatenate([mfcc, chroma, self.lyrics_embedding_input])

        # putting the features through a couple fcnn.
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(x)
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(x)
        latent_embedding = Dense(50, activation='relu', name='embedding', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(x)
        genre = Dense(num_genres, activation='softmax', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(latent_embedding)

        self.net = Model(inputs=[self.chroma_input, self.mfcc_input, self.lyrics_embedding_input], outputs=genre)
        self.embedding = Model(self.net.input, outputs=self.net.get_layer('embedding').output)


class AudioCNNPopularityModel():
    def __init__(self, num_frames, num_freq_bins, num_conv_filters1, pool_size_1, kernel_size, num_genres, embedding_dimension, lambda1=0.0, lambda2=0.0):

        self.num_frames = num_frames
        self.num_freq_bins = num_freq_bins
        self.num_conv_filters1 = num_conv_filters1
        self.pool_size1 = pool_size_1
        self.kernel_size = kernel_size
        self.chroma_input = Input(shape=(num_frames, num_freq_bins))
        self.mfcc_input = Input(shape=(num_frames, num_freq_bins))
        self.embedding_input = Input(shape=(embedding_dimension,))

        chroma = Conv1D(filters=self.num_conv_filters1, kernel_size=self.kernel_size, input_shape=(self.num_frames, self.num_freq_bins))(self.chroma_input)
        chroma = MaxPooling1D(pool_size=self.pool_size1)(chroma)
        chroma = Conv1D(filters=self.num_conv_filters1, kernel_size=self.kernel_size)(chroma)
        chroma = MaxPooling1D(pool_size=2)(chroma)
        chroma = Conv1D(filters=2*self.num_conv_filters1, kernel_size=self.kernel_size)(chroma)
        chroma = MaxPooling1D(pool_size=2)(chroma)

#       temporal pooling, L2, mean
        # max_layer = GlobalMaxPooling1D(data_format='channels_last')(chroma)
        # mean_layer = GlobalAveragePooling1D(data_format='channels_last')(chroma)
        # L2_layer = Lambda(lambda x: self.l2_norm(x, 1))(chroma)
        # #TODO:concatenate
        # chroma = concatenate([max_layer, mean_layer, L2_layer])
        chroma = GlobalAveragePooling1D(data_format='channels_last')(chroma)

        mfcc = Conv1D(filters=self.num_conv_filters1, kernel_size=self.kernel_size, input_shape=(self.num_frames, self.num_freq_bins))(self.mfcc_input)
        mfcc = MaxPooling1D(pool_size=self.pool_size1)(mfcc)
        mfcc = Conv1D(filters=self.num_conv_filters1, kernel_size=self.kernel_size)(mfcc)
        mfcc = MaxPooling1D(pool_size=2)(mfcc)
        mfcc = Conv1D(filters=2*self.num_conv_filters1, kernel_size=self.kernel_size)(mfcc)
        mfcc = MaxPooling1D(pool_size=2)(mfcc)

#       temporal pooling, L2, mean
        # max_layer = GlobalMaxPooling1D(data_format='channels_last')(mfcc)
        # mean_layer = GlobalAveragePooling1D(data_format='channels_last')(mfcc)
        # L2_layer = Lambda(lambda x: self.l2_norm(x, 1))(mfcc)
        # #TODO:concatenate
        # mfcc = concatenate([max_layer, mean_layer, L2_layer])
        mfcc = GlobalAveragePooling1D(data_format='channels_last')(mfcc)

        x = concatenate([mfcc, chroma, self.embedding_input])
#         x = self.embedding_input
        #End
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(x)
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(x)
        latent_embedding = Dense(50, activation='relu', name='embedding', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(x)
        popularity = Dense(10, activation='softmax', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(latent_embedding)
        self.net = Model(inputs=[self.chroma_input, self.mfcc_input, self.embedding_input], outputs=popularity)
        self.embedding = Model(self.net.input, outputs=self.net.get_layer('embedding').output)

    def l2_norm(self, x, axis):
        x = x ** 2
        x = K.sum(x, axis=axis)
        x = K.sqrt(x)
        return x

class BaselineFullyConnectedNet():
    def __init__(self, num_frames, num_freq_bins, num_conv_filters1, pool_size_1, kernel_size, num_genres, embedding_dimension, lambda1=0.0, lambda2=0.0):

        self.num_frames = num_frames
        self.num_freq_bins = num_freq_bins
        self.num_conv_filters1 = num_conv_filters1
        self.pool_size1 = pool_size_1
        self.kernel_size = kernel_size
        self.chroma_input = Input(shape=(num_frames, num_freq_bins))
        self.mfcc_input = Input(shape=(num_frames, num_freq_bins))
        self.embedding_input = Input(shape=(embedding_dimension,))

        chroma = GlobalAveragePooling1D(data_format='channels_last')(self.chroma_input)

        mfcc = GlobalAveragePooling1D(data_format='channels_last')(self.mfcc_input)

        x = concatenate([mfcc, chroma, self.embedding_input])
#         x = self.embedding_input
        #End
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(x)
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(x)
        latent_embedding = Dense(50, activation='relu', name='embedding', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(x)
        genre = Dense(num_genres, activation='softmax', kernel_regularizer=l1_l2(l1=lambda1, l2=lambda2), bias_regularizer=l1_l2(l1=lambda1, l2=lambda2))(latent_embedding)
        self.net = Model(inputs=[self.chroma_input, self.mfcc_input, self.embedding_input], outputs=genre)
        self.embedding = Model(self.net.input, outputs=self.net.get_layer('embedding').output)
