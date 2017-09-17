from keras.layers import Activation
from keras.layers import Input
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model


# class UNet(object):
#     def __init__(self, layers, input_shape, filters, num_classes, shrink=False):
#         self.__layers = layers
#         self.__input_shape = input_shape
#         self.__filters = filters
#         self.__num_classes = num_classes
#         self.__shrink = shrink
#
#     def __down(self, input_layer, filters, kernel_size=(3, 3), pool=True):
#         conv1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(input_layer)
#         bn1 = BatchNormalization()(conv1)
#         act1 = Activation('relu')(bn1)
#         conv2 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(act1)
#         bn2 = BatchNormalization()(conv2)
#         residual = Activation('relu')(bn2)
#         if pool:
#             max_pooling = MaxPooling2D(pool_size=(2, 2))(residual)
#             return max_pooling, residual
#         else:
#             return residual
#
#     def __up(self, residual, input_layer, filters, kernel_size=(3, 3)):
#         up_sampling = UpSampling2D(size=(2, 2))(input_layer)
#         conv0 = Conv2D(filters=filters, kernel_size=(2, 2), padding='same')(up_sampling)
#         bn0 = BatchNormalization()(conv0)
#         up_conv = Activation('relu')(bn0)
#         concat = Concatenate(axis=3)([residual, up_conv])
#         conv1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(concat)
#         bn1 = BatchNormalization()(conv1)
#         act1 = Activation('relu')(bn1)
#         conv2 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(act1)
#         bn2 = BatchNormalization()(conv2)
#         act2 = Activation('relu')(bn2)
#         return act2
#
#     def __unet(self, layers=8, input_shape=(1280, 1920, 3), filters=8, num_classes=1, shrink=False):
#         inputs = Input(shape=input_shape)
#         residuals = []
#         input_layer = inputs
#
#         for layer in xrange(layers - 1):
#             input_layer, residual = self.__down(input_layer=input_layer, filters=filters)
#             residuals.append(residual)
#             filters *= 2
#
#         input_layer = self.__down(input_layer=input_layer, filters=filters, pool=False)
#
#         for layer in xrange(layers - 1):
#             filters /= 2
#             input_layer = self.__up(residual=residuals[-(layer + 1)], input_layer=input_layer, filters=filters)
#
#         # --------------------------------------------------
#         # resize to 1918X1280
#         if shrink:
#             filters /= 2
#             conv1 = Conv2D(filters=filters, kernel_size=(1, 3), padding='valid')(input_layer)  # 1918X1280
#             bn1 = BatchNormalization()(conv1)
#             act1 = Activation('relu')(bn1)
#             conv2 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(act1)
#             bn2 = BatchNormalization()(conv2)
#             input_layer = Activation('relu')(bn2)
#         # --------------------------------------------------
#
#         outputs = Conv2D(filters=num_classes, kernel_size=(1, 1), activation='sigmoid')(input_layer)
#
#         model = Model(inputs=inputs, outputs=outputs)
#         model.summary()
#         return model
#
#     def create_unet_model(self):
#         return self.__unet(layers=self.__layers, input_shape=self.__input_shape,
#                            filters=self.__filters, num_classes=self.__num_classes,
#                            shrink=self.__shrink)


class UNet(object):
    def __init__(self, layers, input_shape, filters, num_classes, shrink=False, activation='relu'):
        self.__layers = layers
        self.__input_shape = input_shape
        self.__filters = filters
        self.__num_classes = num_classes
        self.__shrink = shrink
        self.__activation = activation

    def __down(self, input_layer, filters, kernel_size=(3, 3), pool=True, activation='relu'):
        conv1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(input_layer)
        bn1 = BatchNormalization()(conv1)
        act1 = Activation(activation)(bn1)
        conv2 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(act1)
        bn2 = BatchNormalization()(conv2)
        residual = Activation(activation)(bn2)
        if pool:
            max_pooling = MaxPooling2D(pool_size=(2, 2))(residual)
            return max_pooling, residual
        else:
            return residual

    def __up(self, residual, input_layer, filters, kernel_size=(3, 3), activation='relu'):
        up_sampling = UpSampling2D(size=(2, 2))(input_layer)
        concat = Concatenate(axis=3)([residual, up_sampling])
        conv0 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(concat)
        bn0 = BatchNormalization()(conv0)
        act0 = Activation(activation)(bn0)
        conv1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(act0)
        bn1 = BatchNormalization()(conv1)
        act1 = Activation(activation)(bn1)
        conv2 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(act1)
        bn2 = BatchNormalization()(conv2)
        act2 = Activation(activation)(bn2)
        return act2

    def __unet(self, layers=8, input_shape=(1280, 1920, 3), filters=8, num_classes=1, shrink=False, activation='relu'):
        inputs = Input(shape=input_shape)
        residuals = []
        input_layer = inputs

        for layer in xrange(layers - 1):
            input_layer, residual = self.__down(input_layer=input_layer, filters=filters, pool=True, activation=activation)
            residuals.append(residual)
            filters *= 2

        input_layer = self.__down(input_layer=input_layer, filters=filters, pool=False, activation=activation)

        for layer in xrange(layers - 1):
            filters /= 2
            input_layer = self.__up(residual=residuals[-(layer + 1)], input_layer=input_layer, filters=filters, activation=activation)

        # ----------------------------------------------------------------------------------------------------
        # resize to 1918X1280
        if shrink:
            filters /= 2
            conv1 = Conv2D(filters=filters, kernel_size=(1, 3), padding='valid')(input_layer)  # 1918X1280
            bn1 = BatchNormalization()(conv1)
            act1 = Activation(activation)(bn1)
            conv2 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(act1)
            bn2 = BatchNormalization()(conv2)
            act2 = Activation(activation)(bn2)
            conv3 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(act2)
            bn3 = BatchNormalization()(conv3)
            input_layer = Activation(activation)(bn3)
        # ----------------------------------------------------------------------------------------------------

        outputs = Conv2D(filters=num_classes, kernel_size=(1, 1), activation='sigmoid')(input_layer)

        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model

    def create_unet_model(self):
        return self.__unet(layers=self.__layers, input_shape=self.__input_shape,
                           filters=self.__filters, num_classes=self.__num_classes,
                           shrink=self.__shrink, activation=self.__activation)
