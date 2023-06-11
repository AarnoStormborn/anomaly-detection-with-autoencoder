from keras import Model 
from keras.layers import Input, Dense
from keras.optimizers import Adam

CODE_DIM = 2  #Dimensionality of the compressed represenation
INPUT_SHAPE = 8  #Number of features/variables

X_train_genuine = ''  # Training data consisting of only `genuine` or `normal` class
X_test = ''  # Consisting of both classes

input_layer = Input(shape=(INPUT_SHAPE,))
x = Dense(64, activation='relu')(input_layer)
x = Dense(16, activation='relu')(x)
code = Dense(CODE_DIM, activation='relu')(x)
x = Dense(16, activation='relu')(code)
x = Dense(64, activation='relu')(x)
output_layer = Dense(INPUT_SHAPE, activation='relu')(x)

autoencoder = Model(input_layer, output_layer, name='anomaly')

print(autoencoder.summary())

autoencoder.compile(loss='mae',
                    optimizer=Adam())

history = autoencoder.fit(X_train_genuine, X_train_genuine,
                          epochs=25, batch_size=64,
                          validation_data=(X_test, X_test), shuffle=True)