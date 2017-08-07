from keras.layers import Input
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.merge import Concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model


class UNet(object):
    def __init__(self, layers, input_shape, filters=64):
        self.__layers = layers
        self.__input_shape = input_shape
        self.__filters = filters

    def create_unet_model(self):
        return self.__unet(layers=self.__layers, input_shape=self.__input_shape, filters=self.__filters)

    def __down(self, input_layer, filters, kernel_size=(3, 3), pool=True):
        conv1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(input_layer)
        residual = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(conv1)
        if pool:
            max_pooling = MaxPooling2D(pool_size=(2, 2))(residual)
            return max_pooling, residual
        else:
            return residual

    def __up(self, residual, input_layer, filters, kernel_size=(3, 3)):
        up_sampling = UpSampling2D(size=(2, 2))(input_layer)
        up_conv = Conv2D(filters=filters, kernel_size=(2, 2), padding='same')(up_sampling)
        concat = Concatenate(axis=3)([residual, up_conv])
        conv1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(concat)
        conv2 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(conv1)
        return conv2

    def __unet(self, layers, input_shape, filters):
        inputs = Input(shape=input_shape)
        residuals = []
        input_layer = inputs

        for layer in xrange(layers - 1):
            input_layer, residual = self.__down(input_layer=input_layer, filters=filters)
            residuals.append(residual)
            filters *= 2

        input_layer = self.__down(input_layer=input_layer, filters=filters, pool=False)

        for layer in xrange(layers - 1):
            filters /= 2
            input_layer = self.__up(residual=residuals[-(layer + 1)], input_layer=input_layer, filters=filters)

        outputs = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(input_layer)

        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model
