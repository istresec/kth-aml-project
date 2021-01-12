from utils.Hardtanh import Hardtanh
from utils.distributions import *


class BaseModel(tf.keras.Model):

    def __init__(self, params):
        super(BaseModel, self).__init__()

        # add pseudo-inputs for VampPrior
        if params['prior'] == 'vampprior':
            hidden_units = params['hidden-units']
            pseudo_components = params['vamp-components']

            self.means = tf.keras.Sequential([
                tf.keras.Input(shape=(pseudo_components,)),
                tf.keras.layers.Dense(hidden_units, activation=Hardtanh(0., 1.))
            ])

            self.default_input = tf.Variable(tf.eye(pseudo_components, pseudo_components, dtype=tf.float32))
