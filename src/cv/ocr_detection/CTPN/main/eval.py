import datetime
import os
import sys
import time
import numpy
import tensorflow as tf

sys.path.append(os.getcwd())
for i in os.listdir(os.getcwd()):
    if not '.' in i:
        sys.path.append(os.getcwd()+'/'+i)

from tensorflow.contrib import slim
from nets import model_train as model
from utils.dataset import data_provider as data_provider

#'input'
tf.app.flags.DEFINE_string('data_folder', "", '')
'output'
tf.app.flags.DEFINE_string('export_checkpoint_path', '', '')
tf.app.flags.DEFINE_string('logs_path', '', '')

tf.app.flags.DEFINE_integer('num_readers', 4, '')

FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    status = os.system('sh setup.sh')
    print(status)
    now = datetime.datetime.now()
    StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(FLAGS.logs_path + StyleTime)
    if not os.path.exists(FLAGS.export_checkpoint_path):
        os.makedirs(FLAGS.export_checkpoint_path)

    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    input_bbox = tf.placeholder(tf.float32, shape=[None, 5], name='input_bbox')
    input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

    with tf.device('/gpu:%d' % 0):
        with tf.name_scope('model_%d' % 0) as scope:
            bbox_pred, cls_pred, cls_prob = model.model(input_image)
            #bbox_pred = tf.Print(bbox_pred,[bbox_pred, cls_pred, cls_prob])
            total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = model.loss(bbox_pred, cls_pred, input_bbox,
                                                                                 input_im_info)
            #total_loss = tf.Print(total_loss,[total_loss, model_loss, rpn_cross_entropy, rpn_loss_box])
    with tf.control_dependencies([total_loss,model_loss, rpn_cross_entropy, rpn_loss_box]):
        eval_op = tf.no_op(name='eval_op')

    saver = tf.train.Saver(tf.global_variables())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        ckpt = tf.train.latest_checkpoint(FLAGS.export_checkpoint_path)
        saver.restore(sess, ckpt)
        print('start')
	
        data_generator = data_provider.get_batch(num_workers=FLAGS.num_readers,data_folder = FLAGS.data_folder, shuffle = False)
        num_example = len(tf.gfile.ListDirectory(os.path.join(FLAGS.data_folder,'image')))
        processed_examples = 0       
        model_loss_list = []
        total_loss_list = [] 
        start = time.time()
        while processed_examples < num_example:
            data = next(data_generator)
            ml, tl, _  = sess.run([model_loss, total_loss, eval_op],
                                       feed_dict={input_image: data[0],
                                       input_bbox: data[1],
                                       input_im_info: data[2]})
            processed_examples += len(data[0])
            if processed_examples % 10 ==0:
                print('processed {} images'.format(processed_examples))            
            model_loss_list.append(ml)
            total_loss_list.append(tl)
        print('processed_examples:{}, validation_loss: model loss {:.4f}, total loss {:.4f}'.format(
                    processed_examples, numpy.mean(model_loss_list), numpy.mean(total_loss_list)))

if __name__ == '__main__':
    tf.app.run()
