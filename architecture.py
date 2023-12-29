from keras.layers import InputLayer, Dense, Dropout, BatchNormalization, Conv1D
from keras import regularizers

# local modules
from library.model_gen import build_model

architectures = {
    # Note:  We create multiple hidden layers by enclosing the layer in a list
    # that is multipled by the number of desired layers.  Placing a * in front
    # of the list expands it out into the current list
    # Example of unpacking star operator:
    # (0,1), *[(2, 3, 4)] * 2, (5,6)] -> [(0, 1), (2, 3, 4), (2, 3, 4), (5, 6)]
    # versus not using * in front of the list
    # [(0,1), [(2, 3, 4)] * 2, (5,6)] -> [(0, 1), [(2, 3, 4), (2, 3, 4)], (5, 6)]
    'base': lambda features_n, hidden_n, width_n, l2_penalty, output_n:
    [
        (InputLayer, [], {'input_shape':(features_n,)}),
        (BatchNormalization, [], {}),
        *[(Dense, [width_n], {'activation': 'relu',
                              'kernel_regularizer': regularizers.l2(l2_penalty)})] * hidden_n,
        (Dense, [output_n], {'activation': 'softmax'})
    ],
    # Wide Network
    'wide': lambda features_n, hidden_n, width_n, l2_penalty, output_n:
    [
        (InputLayer, [], {'input_shape':(features_n,)}),
        (BatchNormalization, [], {}),
        *[(Dense, [100], {'activation': 'relu',
                              'kernel_regularizer': regularizers.l2(l2_penalty)})] * hidden_n,
        (BatchNormalization, [], {}),
        *[(Dense, [100], {'activation': 'relu',
                'kernel_regularizer': regularizers.l2(l2_penalty)})] * hidden_n,
        (Dense, [output_n], {'activation': 'softmax'})
    ],
    # Pairs down nodes throughout the network prior to reaching the softmax layer
    'narrowing': lambda features_n, hidden_n, width_n, l2_penalty, output_n:
    [
        (InputLayer, [], {'input_shape':(features_n,)}),
        (BatchNormalization, [], {}),
        *[(Dense, [100], {'activation': 'relu',
                              'kernel_regularizer': regularizers.l2(l2_penalty)})] * hidden_n,
        (BatchNormalization, [], {}),
        *[(Dense, [50], {'activation': 'relu',
                              'kernel_regularizer': regularizers.l2(l2_penalty)})] * hidden_n,
        (BatchNormalization, [], {}),
        *[(Dense, [25], {'activation': 'relu',
                              'kernel_regularizer': regularizers.l2(l2_penalty)})] * hidden_n,
        (Dense, [output_n], {'activation': 'softmax'})
    ],
    # Adds in dropout for experimentation
    'dropout': lambda features_n, hidden_n, width_n, l2_penalty, output_n:
    [
        (InputLayer, [], {'input_shape':(features_n,)}),
        (BatchNormalization, [], {}),
        *[(Dense, [100], {'activation': 'relu',
                              'kernel_regularizer': regularizers.l2(l2_penalty)})] * hidden_n,
        (Dropout, [0.05], {}),
        (BatchNormalization, [], {}),
        *[(Dense, [100], {'activation': 'relu',
                              'kernel_regularizer': regularizers.l2(l2_penalty)})] * hidden_n,
        (BatchNormalization, [], {}),
        (Dense, [output_n], {'activation': 'softmax'})
    ],
    'conv': lambda features_n, output_n:
    [
        (InputLayer, [], {'input_shape':(features_n,)}),
        (Conv1D, [features_n], {'activation': 'relu', "strides": '1',
                            'padding': 'same'}),            
        (Dense, [25], {'activation': 'relu',
                              'kernel_regularizer': regularizers.l2(0.01)}),
        (Dense, [output_n], {'activation': 'softmax'})
    ],

}

def get_model(name, *args):
    template = architectures[name]
    model_spec = template(*args)
    model = build_model(model_spec)
    return model

if __name__ == "__main__":
    # Demonstration of generating a model architecture
    # Create a model with a 10-dimensional feature input
    # 3 hidden layers with 20 nuerons width going across each and an L2 regularizer with .01
    # and 5 output categories
    model = get_model('l2', 10, 3, 20, .01, 5)
    # Model would still need to be compiled, specifying optimizer, loss, etc.
    # but could then be trained (model.fit)
    model.summary()

