import tensorflow as tf

def configure_lr(init_lr, decay_policy, decay_steps, decay_rate, global_steps, warm_lr=0.0001, warm_steps=0):
    lr_decay_list = {
        'exponential_decay': tf.train.exponential_decay(learning_rate=init_lr,
                                                        global_step=global_steps,
                                                        decay_steps=decay_steps,
                                                        decay_rate=decay_rate,
                                                        staircase=True
                                                        ),
        'natural_exp_decay': tf.train.natural_exp_decay(
            learning_rate=init_lr,
            global_step=global_steps,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        ),
        'polynomial_decay': tf.train.polynomial_decay(
            learning_rate=init_lr,
            global_step=global_steps,
            decay_steps=decay_steps,
            end_learning_rate=1e-1 * init_lr,
            power=2.0,
            cycle=True,
            name=None
        ),
      'fixed':init_lr
    }

    def false_fn():
        return lr_decay_list[decay_policy]

    def true_fn():
        return tf.train.polynomial_decay(learning_rate=warm_lr,
                                         global_step=global_steps,
                                         decay_steps=warm_steps,
                                         end_learning_rate=init_lr,
                                         power=1.0
                                         )
    pred_result = global_steps < warm_steps

    learning_rate = tf.cond(pred_result, true_fn, false_fn)
    return learning_rate


def configure_optimizer(optimizer, learning_rate):
    opt_gpu_list = {
        'rmsp': tf.train.RMSPropOptimizer(learning_rate,epsilon=1),
        'adam': tf.train.AdamOptimizer(learning_rate),
        'sgd': tf.train.GradientDescentOptimizer(learning_rate),
        'mometum': tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
    }

    return opt_gpu_list[optimizer]
  
def model_optimizer(model_name,learning_rate):
  opt_list = {
    'inception_v4' : tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9),
    'inception_resnet_v2': tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9),
    'resnet_v2_101': tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9),
    'mobilenet_v2_14': tf.train.RMSPropOptimizer(learning_rate,momentum=0.9,decay=0.9),
    'nasnet_large': tf.train.RMSPropOptimizer(learning_rate,epsilon=1,decay=0.9),
    'pnasnet_large': tf.train.RMSPropOptimizer(learning_rate),
    'vgg_16': tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
  }
  return opt_list[model_name]


def model_exclusions(model_name):
    exclusions_list = {
        'inception_v1': ['InceptionV1/Logits'],
        'inception_v2': ['InceptionV2/Logits'],
        'inception_v3': ['InceptionV3/Logits', 'InceptionV3/AuxLogits'],
        'inception_v4': ['InceptionV4/Logits', 'InceptionV4/AuxLogits'],
        'inception_resnet_v2': ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits'],
        'vgg_16': ['vgg_16/fc8'],
        'vgg_19': ['vgg_19/fc8'],
        'resnet_v1_50': ['resnet_v1_50/logits'],
        'resnet_v1_101': ['resnet_v1_101/logits'],
        'resnet_v1_152': ['resnet_v1_152/logits'],
        'resnet_v2_50': ['resnet_v2_50/logits'],
        'resnet_v2_101': ['resnet_v2_101/logits'],
        'resnet_v2_152': ['resnet_v2_152/logits'],
        'mobilenet_v1_025': ['MobilenetV1/Logits'],
        'mobilenet_v1_050' : ['MobilenetV1/Logits'],
        'mobilenet_v1_10':['MobilenetV1/Logits'],
        'mobilenet_v2_10':['MobilenetV2/Logits'],
        'mobilenet_v2_14':['MobilenetV2/Logits'],
        'nasnet_large': ['final_layer','aux'],
        'nasnet_mobile': ['final_layer','aux'],
        'pnasnet_large': ['final_layer','aux'],
        'pnasnet_mobile': ['final_layer','aux']
       }
    return exclusions_list[model_name]


def configure_image_size(model_name):
  image_size_list = {
        'inception_v4' : 299,
    	'inception_resnet_v2': 299,
        'resnet_v2_101': 224,
    	'mobilenet_v2_14': 224,
    	'nasnet_large': 331,
    	'pnasnet_large': 331,
    	'vgg_16': 224
    }
  return image_size_list[model_name]

def configure_weight_decay(model_name):
  weight_decay_list = {
        'inception_v4' : 0.0004,
    	'inception_resnet_v2': 0.0004,
        'resnet_v2_101': 0.0001,
    	'mobilenet_v2_14': 0.0004,
    	'nasnet_large': 0.00004,
    	'pnasnet_large': 0.00004,
    	'vgg_16': 0.0005
  	}
  return weight_decay_list[model_name]

def configure_batch_size(model_name):
  batch_size_list = {
        'inception_v4' : 32,
    	'inception_resnet_v2': 32,
        'resnet_v2_101': 64,
    	'mobilenet_v2_14': 96,
    	'nasnet_large': 8,
    	'pnasnet_large': 8,
    	'vgg_16': 128
  	}
  return batch_size_list[model_name]

def configure_init_learning_rate(model_name):
  init_learning_rate_list = {
        'inception_v4' : 0.001,
    	'inception_resnet_v2': 0.001,
        'resnet_v2_101': 0.001,
    	'mobilenet_v2_14': 0.0001,
    	'nasnet_large': 0.0001,
    	'pnasnet_large': 0.0001,
    	'vgg_16': 0.001
  	}
  return init_learning_rate_list[model_name]