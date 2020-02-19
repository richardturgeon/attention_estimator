"""Build a Keras classification model."""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, multiply

def build_model(params, tensor=None):
    """Build a Keras classification NN."""

#   --------------------------------------
    def identity_fn(a):
        return a

    def BatchNormalization():
        return identity_fn

    def Dropout(*args):
        return identity_fn
#   --------------------------------------

    nbr_classes = params['nbr_classes']             # number of classes
    dropout = params['dropout']                     # dropout value or 0

    inputs = Input(tensor=tensor)

    x = Dense(1000, activation='relu')(inputs)
    x = BatchNormalization()(x)

    a = Dense(1000, activation='relu')(x)
    a = BatchNormalization()(a)

#   b = Dense(1000, activation='softmax')(x)
#   x = multiply([a,b])
    x = a

    x = Dense(500, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(250, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(125, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(60, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(30, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

#   outputs = Dense(nbr_classes, activation='softmax')(x)       ### Rick's approach
    outputs = Dense(nbr_classes, activation=None)(x)            ### Cerebras 

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model

