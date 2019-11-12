#!/usr/bin/env python
import keras
import tensorflowjs as tfjs

model = keras.models.load_model('web_model.h5')
tfjs.converters.save_keras_model(model, 'web/tfjs_model')
