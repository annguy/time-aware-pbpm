import tensorflow as tf
from tensorflow import keras
import numpy as np


class TLSTM_layer(keras.layers.Layer):
    """
    Time-ware LSTM inherits from tf.keras.layers.Layer. Implmentation is based on [1]
    [1] Baytas, Inci M., Cao Xiao, Xi Zhang, Fei Wang, Anil K. Jain, and Jiayu Zhou. “Patient Subtyping
    via Time-Aware LSTM Networks.”  KDD ’17


    Attributes
    ----------
    input_dim : int
        the dimension of the input layer.
    hidden_dim : int
        the dimension of the hidden layer.
    trainable : bool
        set the variable to be trainable or not.
    Wi, Wf, Wog : tensor
        weight matrices corresponding to the input, forget and output gates respectively,
        modifying the input connections.
    Ui, Uf, Uog : tensor
        weight matrices corresponding to the input, forget and output gates respectively,
        modifying the hidden connections.
    bi, bf, bog : tensor
        the bias vectors corresponding to the input, forget and output gates, respectively.
    Wc, Uc, bc : tensor
        the weight matrices and bias vector to compute the current cell memory.
    W_decomp, b_decomp : tensor
        subspace decompostion weight matrix and bias vector.

    """

    def __init__(self,
                 hidden_dim,
                 return_sequence=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 trainable=True,
                 supports_masking=True,
                 name=None):
        """Constructor of T-LSTM objects.

        Parameters
        ----------
        hidden_dim : int
            dimension of the hidden layer.
        return_sequence : bool
            weather to return all hidden states. Default: True
        kernel_initializer : str
            Initializer for the kernel weights matrix, used for the linear transformation of the inputs.
             Default: glorot_uniform.
         bias_initializer: str
             Initializer for the bias vector. Default: zeros.
        trainable :bool
            set the variables (weight and bias) to be trainable (for training) or not (for testing).
        supports_masking : bool
            weather layer should supporting masking
        name : str
            layers name
        """
        super(TLSTM_layer, self).__init__(name=name)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = supports_masking
        # set initializer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        # set dimensions
        self.hidden_dim = hidden_dim
        self.return_sequence = return_sequence
        self.trainable = trainable

    def build(self, input_shape):
        
        self.input_dim = input_shape[-1]
        # Set trainable weights
        self.Wi = self.add_weight(shape=(self.input_dim, self.hidden_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True,
                                  name='Input_Hidden_weight')
        self.Ui = self.add_weight(shape=(self.hidden_dim, self.hidden_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True,
                                  name='Input_State_weight')
        self.bi = self.add_weight(shape=self.hidden_dim,
                                  initializer=self.bias_initializer,
                                  trainable=True,
                                  name='Input_Hidden_bias')

        self.Wf = self.add_weight(shape=(self.input_dim, self.hidden_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True,
                                  name='Forget_Hidden_weight')
        self.Uf = self.add_weight(shape=(self.hidden_dim, self.hidden_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True,
                                  name='Forget_State_weight')
        self.bf = self.add_weight(shape=self.hidden_dim,
                                  initializer=self.bias_initializer,
                                  trainable=True,
                                  name='Forget_Hidden_bias')

        self.Wog = self.add_weight(shape=(self.input_dim, self.hidden_dim),
                                   initializer=self.kernel_initializer,
                                   trainable=True,
                                   name='Output_Hidden_weight')
        self.Uog = self.add_weight(shape=(self.hidden_dim, self.hidden_dim),
                                   initializer=self.kernel_initializer,
                                   trainable=True,
                                   name='Output_State_weight')
        self.bog = self.add_weight(shape=self.hidden_dim,
                                   initializer=self.bias_initializer,
                                   trainable=True,
                                   name='Output_Hidden_bias')

        self.Wc = self.add_weight(shape=(self.input_dim, self.hidden_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True,
                                  name='Cell_Hidden_weight')
        self.Uc = self.add_weight(shape=(self.hidden_dim, self.hidden_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True,
                                  name='Cell_State_weight')
        self.bc = self.add_weight(shape=self.hidden_dim,
                                  initializer=self.bias_initializer,
                                  trainable=True,
                                  name='Cell_Hidden_bias')

        self.W_decomp = self.add_weight(shape=(self.hidden_dim, self.hidden_dim),
                                        initializer=self.kernel_initializer,
                                        trainable=True,
                                        name='Decomposition_Hidden_weight')
        self.b_decomp = self.add_weight(shape=self.hidden_dim,
                                        initializer=self.bias_initializer,
                                        trainable=True,
                                        name='Decomposition_Hidden_bias_enc')

    def tlstm_unit(self, prev_hidden_memory, concat_input):
        """The update equations for the time-aware LSTM.

        Parameters
        ----------
        prev_hidden_memory : list of tensor
            a list containing the previous hidden states and previous cell memory (1, 2, batch_size, hidden_dim)
        concat_input : tensor
            a tensor containing all the input data and the elapsed times (batch_size, input_dim+1)

        Returns
        -------
        tensor
            Updated prev_hidden_memory and the cell state concatenated (1, 2, batch_size, hidden_dim)
        """
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

        batch_size = tf.shape(concat_input)[0]
        x = tf.slice(concat_input, [0, 1], [batch_size, self.input_dim])  # (batch_size, features)
        t = tf.slice(concat_input, [0, 0], [batch_size, 1])  # (batch_size, 1)

        # Dealing with time irregularity

        # Map elapse time in days or months
        T = self.map_elapse_time(t)

        # Decompose the previous cell if there is an elapse time
        C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        C_ST_dis = tf.multiply(T, C_ST)
        # if T is 0, then the weight is one
        prev_cell = prev_cell - C_ST + C_ST_dis

        # Input gate
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi)

        # Forget Gate
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(prev_hidden_state, self.Uf) + self.bf)

        # Output Gate
        o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(prev_hidden_state, self.Uog) + self.bog)

        # Candidate Memory Cell
        C = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(prev_hidden_state, self.Uc) + self.bc)

        # Current Memory cell
        Ct = f * prev_cell + i * C

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])

    def call(self, x, time):
        """ Compute the hidden states for the input sequence.

        Parameters
        ----------
        x : array
            the input of shape (batch_size, sequence_length, input_dimension).
        time : array
            the elapsed time of shape (batch_size, sequence_length).

        Returns
        -------
        tensor
            The computed hidden states of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size = tf.shape(x)[0]
        scan_input_ = tf.transpose(x, perm=[2, 0, 1])
        scan_input = tf.transpose(scan_input_)  # scan_input: (seq_length, batch_size, input_dim)
        scan_time = tf.transpose(time)

        initial_hidden = tf.zeros([batch_size, self.hidden_dim], tf.float32)
        ini_state_cell = tf.stack([initial_hidden, initial_hidden])

        # Make scan_time have shape of (seq_length, batch_size, 1)
        scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0], tf.shape(scan_time)[1], 1])
        concat_input = tf.concat([scan_time, scan_input], 2)  # concat_input: (seq_length, batch_size, input_dim+1)

        # Call TLSTM_Unit recursively for each time step (seq_length, 2, batch_size, hidden_dim)
        packed_hidden_states = tf.scan(self.tlstm_unit, concat_input, initializer=ini_state_cell, name='states')

        # Extract all the hidden states
        all_states = packed_hidden_states[:, 0, :, :]  # (seq_length, batch_size, hidden_dim)
        all_states = tf.transpose(all_states, perm=[1, 0, 2])  # (batch_size, seq_len, hidden_dim)

        if self.return_sequence:
            return all_states  # return all hidden states
        else:
            return all_states[:, -1, :]  # return last hidden state

    def map_elapse_time(self, t):
        """ Given the time intervals, compute the decay terms.

        Parameters
        ----------
        t : tensor
            input time intervals.

        Returns
        -------
        tensor
            The computed decay terms.
        """
        c1 = tf.constant(1, dtype=tf.float32)
        c2 = tf.constant(2.7183, dtype=tf.float32)

        T = tf.math.divide(c1, tf.math.log(t + c2), name='Log_elapse_time')
        Ones = tf.ones([1, self.hidden_dim], dtype=tf.float32)
        T = tf.matmul(T, Ones)

        return T


    def get_config(self):
        config = ({'hidden_dim': self.hidden_dim,
                   'supports_masking': self.supports_masking,
                   'kernel_initializer': self.kernel_initializer,
                   'bias_initializer': self.bias_initializer,
                   'return_sequence': self.return_sequence,
                   'trainable': self.trainable,
                   'name': self.name})

        return config


if __name__ == "__main__":
    # %% run on dummy tensors
    inp_dim = 2
    hid_dim = 5

    time_steps = 10
    batchSize = 20

    x_ = np.ones([batchSize, time_steps, inp_dim])
    # leftpad sequences with zeros -> simulate variable length inputs in batch
    # first 2 sequences have 2 leading zeros, next two 4 leading zeros, ...
    x_[:2, :2, :] = 0
    x_[2:4, :4, :] = 0
    x_[4:6, :6, :] = 0
    x_ = tf.convert_to_tensor(x_, dtype='float32')
    dts = np.ones([batchSize, time_steps])
    dts[:2, :2] = 0
    dts[2:4, :4] = 0
    dts[4:6, :6] = 0
    dts = tf.convert_to_tensor(dts, dtype='float32')

    # initialize TLSTM layer object
    TLSTM_layer_ = TLSTM_layer(hid_dim, return_sequence=True)
    hidden = TLSTM_layer_(x_, dts)
    print(tf.shape(hidden))

    TLSTM_layer_ = TLSTM_layer(hid_dim, return_sequence=False)
    hidden = TLSTM_layer_(x_, dts)
    print(tf.shape(hidden))
