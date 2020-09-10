"""
The CSModel uses class weights from derived from the training set distribution and puts class weights on the
predicted samples.

All_TLSTM_Model is based replaces LSTM cells in [2] by T-LSTM cells [1].

CS_TLSTM_Model is a combination of CWModel and ALL_TLSTM_Model.

[1] Baytas, Inci M., et al.
"Patient subtyping via time-aware LSTM networks."
Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining. 2017.

[2] Tax, Niek, Ilya Verenich, Marcello La Rosa, and Marlon Dumas.
“Predictive Business Process Monitoring with LSTM Neural Networks.”
In Advanced Information Systems Engineering. 2017
"""

from tensorflow import keras
import tensorflow as tf
import os
from src.Models.TLSTM_layer import TLSTM_layer


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)



class CSModel:
    """
    CSModel Class
    ....
    
    This Model uses class_weights to improve business process related predictions. Architecture is based on [1].
    [1] Tax et al. “Predictive Business Process Monitoring with LSTM Neural Networks.” In Advanced Information Systems
    Engineering, 2017.


    Attributes
    ----------
    maxlen : int
        Maximum observed length of the case.
    num_feature : int
        Number of features per time step.
    max_task : int
        No of Unique Activities in the event log.
    target_chars  : list
        ascii coded characters of the target activities.
       (target includes one excess activity '!' case end)
       used to define output dimension of last dense layer
    name : str
        name of the file
            
            
     Methods
    -------------
    train(self, X, y_a, y_t, class_weights, hparams, HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_LEARNING_RATE):
        builds and trains the model on the data
    
    
    """

    def __init__(self, maxlen=None,
                 max_task=None,
                 target_chars=None,
                 name='CS_Model',
                 num_features=None):
        """
        Parameter
        -------------------
        maxlen : int
            Maximum observed length of the case.
        num_feature : int
            Number of features per time step.
        max_task : int
            No of Unique Activities in the event log.
        target_chars  : list
            ascii coded characters of the target activities.
           (target includes one excess activity '!' case end)
           used to define output dimension of last dense layer
        name : str
            name of the file

        """

        self.maxlen = maxlen
        self.num_features = num_features
        self.max_task = max_task
        self.target_chars = target_chars
        self.name = name

    def train(self,
              X,
              y_a,
              y_t,
              class_weights,
              hparams,
              HP_NUM_UNITS,
              HP_DROPOUT,
              HP_OPTIMIZER,
              HP_LEARNING_RATE,
              epochs=150,
              batch_size=64,
              val_split=0.2):
        """
        Parameter
        ----------------
        X: numpy.ndarray
           training_set data left padded with zeros (samples, time_steps, features)
        y_a: numpy.ndarray
            training_set labels for next activity one-hot encoded (samples, unique_activities)
        y_t: numpy.ndarray
            training_set labels for time until next event scaled by avg time between events (samples, )
        class_weights: dict
            class weights of the unique activities
        hparams: dict
            Hyperparamter set for each run
        HP_NUM_UNITS:tensorboard.plugins.hparams.summary_v2.HParam
            Number of LSTM units
        HP_DROPOUT:tensorboard.plugins.hparams.summary_v2.HParam
            Dropout rate
        HP_OPTIMIZER:tensorboard.plugins.hparams.summary_v2.HParam
            Optimizer
        HP_LEARNING_RATE:tensorboard.plugins.hparams.summary_v2.HParam
            Learning Rate
        epochs: int
            number of epochs for training
        batch_size: int
            Batchsize for training
        val_split : float
            ratio of data used for validation set (e.g. last val_split % of X)
        
        Returns
        --------------
        hist.history: dict
            history of the training
            
        hist.history['val_loss'][-1]: list
            val_loss of the last epoch 
        
        """

        # build the model: 
        print('Build model...')
        main_input = keras.Input(shape=(self.maxlen, self.num_features), name='main_input')

        # train a 2-layer LSTM with one shared layer
        l1 = keras.layers.LSTM(hparams[HP_NUM_UNITS],
                               implementation=2,
                               kernel_initializer='glorot_uniform',
                               return_sequences=True,
                               dropout=0.2)(main_input)  # the shared layer
        b1 = keras.layers.BatchNormalization()(l1)

        l2_1 = keras.layers.LSTM(hparams[HP_NUM_UNITS],
                                 implementation=2,
                                 kernel_initializer='glorot_uniform',
                                 return_sequences=False,
                                 dropout=0.2)(b1)  # the layer specialized in activity prediction

        b2_1 = keras.layers.BatchNormalization()(l2_1)

        l2_2 = keras.layers.LSTM(hparams[HP_NUM_UNITS],
                                 implementation=2,
                                 kernel_initializer='glorot_uniform',
                                 return_sequences=False,
                                 dropout=0.2)(b1)  # the layer specialized in time prediction

        b2_2 = keras.layers.BatchNormalization()(l2_2)

        act_output = keras.layers.Dense(len(self.target_chars),
                                        activation='softmax',
                                        kernel_initializer='glorot_uniform',
                                        name='act_output')(b2_1)

        time_output = keras.layers.Dense(1,
                                         kernel_initializer='glorot_uniform',
                                         name='time_output')(b2_2)

        model = keras.Model(inputs=[main_input],
                            outputs=[act_output, time_output])

        model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'},
                      optimizer=hparams[HP_OPTIMIZER])

        model.optimizer.learning_rate.assign(hparams[HP_LEARNING_RATE])

        # define callbacks

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=25)
        path = os.path.abspath(os.curdir)
        model_checkpoint = keras.callbacks.ModelCheckpoint(path + '/Model/1.CS/CS_' +
                                                           'lr_' + str(hparams[HP_LEARNING_RATE]) +
                                                           'units_' + str(hparams[HP_NUM_UNITS]) +
                                                           'DO_' + str(hparams[HP_DROPOUT]) + '_' +
                                                           self.name + '{epoch:02d}-{val_loss:.2f}.h5',
                                                           monitor='val_loss',
                                                           verbose=0,
                                                           save_best_only=True,
                                                           save_weights_only=False,
                                                           mode='auto')
        lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       factor=0.5,
                                                       patience=10,
                                                       verbose=0,
                                                       mode='auto',
                                                       min_delta=0.0001,
                                                       cooldown=0,
                                                       min_lr=0)
        model.summary()
        hist = model.fit(X,
                         {'act_output': y_a, 'time_output': y_t},
                         validation_split=val_split,
                         verbose=2,
                         callbacks=[early_stopping, model_checkpoint, lr_reducer],
                         class_weight={'act_output': class_weights, 'time_output': 1},
                         batch_size=batch_size,
                         epochs=epochs)
        return hist.history['val_loss'][-1], hist.history





class ALL_TLSTM_Model:
    """
    ALL_TLSTM_Model Class

    ....

    In this model,the Shared layer and the Time specialised layer are TLSTM layer. Architecture is based on [1].
    [1] Tax et al. “Predictive Business Process Monitoring with LSTM Neural Networks.” In Advanced Information Systems
    Engineering, 2017

    Attributes
    ----------
    maxlen : int
        Maximum observed length of the case.
    num_feature : int
        Number of features per time step.
    max_task : int
        No of Unique Activities in the event log.
    target_chars  : list
        ascii coded characters of the target activities.
       (target includes one excess activity '!' case end)
       used to define output dimension of last dense layer
    name : str
        name of the file


    Methods
    -------------
    train(self, X, dts, y_a, y_t, class_weights, hparams, HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_LEARNING_RATE):
        builds and trains the model on the data


    """

    def __init__(self, maxlen=None,
                 max_task=None,
                 target_chars=None,
                 name='ALL_TLSTM_Model',
                 num_features=None):
        """
        Parameter
        -------------------
        maxlen : int
            Maximum observed length of the case.
        num_feature : int
            Number of features per time step.
        max_task : int
            No of Unique Activities in the event log.
        target_chars  : list
            ascii coded characters of the target activities.
           (target includes one excess activity '!' case end)
           used to define output dimension of last dense layer
        name : str
            name of the file

        """

        self.maxlen = maxlen
        self.num_features = num_features
        self.max_task = max_task
        self.target_chars = target_chars
        self.name = name

    def train(self,
              X,
              dts,
              y_a,
              y_t,
              hparams,
              HP_NUM_UNITS,
              HP_DROPOUT,
              HP_OPTIMIZER,
              HP_LEARNING_RATE,
              epochs=150,
              batch_size=64,
              val_split=0.2):
        """
        Parameter
        ----------------
        X: numpy.ndarray
           training_set data left padded with zeros (samples, time_steps, features)
        dts: numpy.ndarray
            time deltas for each sample left padded not scaled in seconds (samples, time_steps)
        y_a: numpy.ndarray
            training_set labels for next activity one-hot encoded (samples, unique_activities)
        y_t: numpy.ndarray
            training_set labels for time until next event scaled by avg time between events (samples, )
        hparams: dict
            Hyperparamter set for each run
        HP_NUM_UNITS:tensorboard.plugins.hparams.summary_v2.HParam
            Number of LSTM units
        HP_DROPOUT:tensorboard.plugins.hparams.summary_v2.HParam
            Dropout rate
        HP_OPTIMIZER:tensorboard.plugins.hparams.summary_v2.HParam
            Optimizer
        HP_LEARNING_RATE:tensorboard.plugins.hparams.summary_v2.HParam
            Learning Rate
        epochs: int
            number of epochs for training
        batch_size: int
            Batchsize for training
        val_split : float
            ratio of data used for validation set (e.g. last val_split % of X)



        Returns
        --------------
        hist.history: dict
            history of the training

        hist.history['val_loss'][-1]: list
            val_loss of the last epoch

        """

        # build the model:
        print('Build model...')
        # main_input: (batch_size, seq_len, num_feat)
        main_input = keras.Input(shape=(self.maxlen,
                                        self.num_features),
                                 name='main_input')

        # delta_ts: (batch_size, 1)
        delta_ts = keras.Input(shape=(self.maxlen),
                               name='delta_ts')

        l1 = TLSTM_layer(hparams[HP_NUM_UNITS],
                         return_sequence=True)(main_input, delta_ts)

        # batchnorm shared TLSTM layer output
        b1 = keras.layers.BatchNormalization()(l1)

        # the layer specialized in time prediction
        l2_1 = TLSTM_layer(hparams[HP_NUM_UNITS],
                           return_sequence=False)(b1, delta_ts)

        # batchnorm activity prediction specialized LSTM
        b2_1 = keras.layers.BatchNormalization()(l2_1)

        # the layer specialized in time prediction --> only output last hidden state
        l2_2 = TLSTM_layer(hparams[HP_NUM_UNITS],
                           return_sequence=False)(b1, delta_ts)

        # batchnorm time prediction specialized TLSTM output
        b2_2 = keras.layers.BatchNormalization()(l2_2)

        d2_1 = keras.layers.Dropout(rate=hparams[HP_DROPOUT])(b2_1)
        d2_2 = keras.layers.Dropout(rate=hparams[HP_DROPOUT])(b2_2)

        act_output = keras.layers.Dense(len(self.target_chars),
                                        activation='softmax',
                                        kernel_initializer='glorot_uniform',
                                        name='act_output')(d2_1)

        time_output = keras.layers.Dense(1, kernel_initializer='glorot_uniform',
                                         name='time_output')(d2_2)
        # define model
        model = keras.Model(inputs=[main_input, delta_ts],
                            outputs=[act_output, time_output])


        # compile model
        model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'},
                      optimizer=hparams[HP_OPTIMIZER])

        model.optimizer.learning_rate.assign(hparams[HP_LEARNING_RATE])

        # define callbacks


        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=25)

        path = os.path.abspath(os.curdir)
        model_checkpoint = keras.callbacks.ModelCheckpoint(path + '/Model/2.TLSTM/TLSTM_' +
                                                           'lr_' + str(hparams[HP_LEARNING_RATE]) +
                                                           'units_' + str(hparams[HP_NUM_UNITS]) +
                                                           'DO_' + str(hparams[HP_DROPOUT]) + '_' +
                                                           self.name + '{epoch:02d}-{val_loss:.2f}.h5',
                                                           monitor='val_loss',
                                                           verbose=0,
                                                           save_best_only=True,
                                                           save_weights_only=False,
                                                           mode='auto')

        lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       factor=0.5,
                                                       patience=10,
                                                       verbose=0,
                                                       mode='auto',
                                                       min_delta=0.0001,
                                                       cooldown=0,
                                                       min_lr=0)
        model.summary()
        hist = model.fit({'main_input': X, 'delta_ts': dts},
                         {'act_output': y_a, 'time_output': y_t},
                         validation_split=val_split,
                         verbose=2,
                         callbacks=[early_stopping, model_checkpoint, lr_reducer],
                         batch_size=batch_size,
                         epochs=epochs)
        return hist.history['val_loss'][-1], hist.history




class CS_TLSTM_Model:
    """
    CS_TLSTM_Model Class

    ....

    In this model,the Shared layer and the Time specialised layer are TLSTM layer. Architecture is based on [1].
    [1] Tax et al. “Predictive Business Process Monitoring with LSTM Neural Networks.” In Advanced Information Systems
    Engineering, 2017. The model also uses class_weights.


    Attributes
    ----------
    maxlen : int
        Maximum observed length of the case.
    num_feature : int
        Number of features per time step.
    max_task : int
        No of Unique Activities in the event log.
    target_chars  : list
        ascii coded characters of the target activities.
       (target includes one excess activity '!' case end)
       used to define output dimension of last dense layer
    name : str
        name of the file


    Methods
    -------------
    train(self, X, dts, y_a, y_t, class_weights, hparams, HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_LEARNING_RATE):
        builds and trains the model on the data


    """

    def __init__(self, maxlen=None,
                 max_task=None,
                 target_chars=None,
                 name='CS_TLSTM_Model',
                 num_features=None):
        """
        Parameter
        -------------------
        maxlen : int
            Maximum observed length of the case.
        num_feature : int
            Number of features per time step.
        max_task : int
            No of Unique Activities in the event log.
        target_chars  : list
            ascii coded characters of the target activities.
           (target includes one excess activity '!' case end)
           used to define output dimension of last dense layer
        name : str
            name of the file

        """

        self.maxlen = maxlen
        self.num_features = num_features
        self.max_task = max_task
        self.target_chars = target_chars
        self.name = name

    def train(self,
              X,
              dts,
              y_a,
              y_t,
              class_weights,
              hparams,
              HP_NUM_UNITS,
              HP_DROPOUT,
              HP_OPTIMIZER,
              HP_LEARNING_RATE,
              epochs=150,
              batch_size=64,
              val_split=0.2):
        """
        Parameter
        ----------------
        X: numpy.ndarray
           training_set data left padded with zeros (samples, time_steps, features)
        dts: numpy.ndarray
            time deltas for each sample left padded not scaled in seconds (samples, time_steps)
        y_a: numpy.ndarray
            training_set labels for next activity one-hot encoded (samples, unique_activities)
        y_t: numpy.ndarray
            training_set labels for time until next event scaled by avg time between events (samples, )
        class_weights: dict
            class weights of the unique activities
        hparams: dict
            Hyperparamter set for each run
        HP_NUM_UNITS:tensorboard.plugins.hparams.summary_v2.HParam
            Number of LSTM units
        HP_DROPOUT:tensorboard.plugins.hparams.summary_v2.HParam
            Dropout rate
        HP_OPTIMIZER:tensorboard.plugins.hparams.summary_v2.HParam
            Optimizer
        HP_LEARNING_RATE:tensorboard.plugins.hparams.summary_v2.HParam
            Learning Rate
        epochs: int
            number of epochs for training
        batch_size: int
            Batchsize for training
        val_split : float
            ratio of data used for validation set (e.g. last val_split % of X)



        Returns
        --------------
        hist.history: dict
            history of the training

        hist.history['val_loss'][-1]: list
            val_loss of the last epoch

        """

        # build the model:
        print('Build model...')
        # main_input: (batch_size, seq_len, num_feat)
        main_input = keras.Input(shape=(self.maxlen,
                                        self.num_features),
                                 name='main_input')

        # delta_ts: (batch_size, 1)
        delta_ts = keras.Input(shape=(self.maxlen),
                               name='delta_ts')

        l1 = TLSTM_layer(hparams[HP_NUM_UNITS],
                         return_sequence=True)(main_input, delta_ts)

        # batchnorm shared TLSTM layer output
        b1 = keras.layers.BatchNormalization()(l1)

        # the layer specialized in time prediction
        l2_1 = TLSTM_layer(hparams[HP_NUM_UNITS],
                           return_sequence=False)(b1, delta_ts)

        # batchnorm activity prediction specialized LSTM
        b2_1 = keras.layers.BatchNormalization()(l2_1)

        # the layer specialized in time prediction --> only output last hidden state
        l2_2 = TLSTM_layer(hparams[HP_NUM_UNITS],
                           return_sequence=False)(b1, delta_ts)

        # batchnorm time prediction specialized TLSTM output
        b2_2 = keras.layers.BatchNormalization()(l2_2)

        d2_1 = keras.layers.Dropout(rate=hparams[HP_DROPOUT])(b2_1)
        d2_2 = keras.layers.Dropout(rate=hparams[HP_DROPOUT])(b2_2)

        act_output = keras.layers.Dense(len(self.target_chars),
                                        activation='softmax',
                                        kernel_initializer='glorot_uniform',
                                        name='act_output')(d2_1)

        time_output = keras.layers.Dense(1, kernel_initializer='glorot_uniform',
                                         name='time_output')(d2_2)
        # define model
        model = keras.Model(inputs=[main_input, delta_ts],
                            outputs=[act_output, time_output])

        # compile model
        model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'},
                      optimizer=hparams[HP_OPTIMIZER])

        model.optimizer.learning_rate.assign(hparams[HP_LEARNING_RATE])

        # define callbacks

        # model.optimizer.learning_rate.assign(opt)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=25)

        path = os.path.abspath(os.curdir)
        model_checkpoint = keras.callbacks.ModelCheckpoint(path + '/Model/3.CS_TLSTM/CS_TLSTM_' +
                                                           'lr_' + str(hparams[HP_LEARNING_RATE]) +
                                                           'units_' + str(hparams[HP_NUM_UNITS]) +
                                                           'DO_' + str(hparams[HP_DROPOUT]) + '_' +
                                                           self.name + '{epoch:02d}-{val_loss:.2f}.h5',
                                                           monitor='val_loss',
                                                           verbose=0,
                                                           save_best_only=True,
                                                           save_weights_only=False,
                                                           mode='auto')

        lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       factor=0.5,
                                                       patience=10,
                                                       verbose=0,
                                                       mode='auto',
                                                       min_delta=0.0001,
                                                       cooldown=0,
                                                       min_lr=0)
        model.summary()
        hist = model.fit({'main_input': X, 'delta_ts': dts},
                         {'act_output': y_a, 'time_output': y_t},
                         validation_split=val_split,
                         verbose=2,
                         callbacks=[early_stopping, model_checkpoint, lr_reducer],
                         class_weight={'act_output': class_weights, 'time_output': 1},
                         batch_size=batch_size,
                         epochs=epochs)
        return hist.history['val_loss'][-1], hist.history