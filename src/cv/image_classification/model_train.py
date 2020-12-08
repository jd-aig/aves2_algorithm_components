# from __future__ import print_function
import tensorflow as tf
# the same as neuhub
import os
import numpy
import time
from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import prefetching_ops
from tensorflow.python.framework import function
from tensorflow.python.platform import gfile
import Configure_file
import performence

tf.app.flags.DEFINE_string('pre_trained_model_ckpt_path', '', '')
tf.app.flags.DEFINE_string('checkpoint_dir', '', '')
tf.app.flags.DEFINE_string('summary_dir', '', '')
tf.app.flags.DEFINE_string('result_dir', '', '')
tf.app.flags.DEFINE_string('train_data_dir', '', '')
tf.app.flags.DEFINE_string('validation_data_dir', '', '')
tf.app.flags.DEFINE_integer('num_class', 37, 'the number of training sample categories')
tf.app.flags.DEFINE_string('model_name', 'vgg_16', '')
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_float('epochs', 100, '')
tf.app.flags.DEFINE_float('decay_rate', 0.92, '')
tf.app.flags.DEFINE_float('decay_epochs', 3, '')
tf.app.flags.DEFINE_string('lr_decay', 'exponential_decay',
                           'exponential_decay, natural_exp_decay,polynomial_decay,fixed')
tf.app.flags.DEFINE_integer('display_every_steps', 50, '')
tf.app.flags.DEFINE_integer('eval_every_epochs', 5, '')
tf.app.flags.DEFINE_integer('fine_tune', 1, 'whether the model is trained from a pre-trained model')
tf.app.flags.DEFINE_integer('early_stop', 1, 'whether to stop training model early')
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)
slim = tf.contrib.slim

height = Configure_file.configure_image_size(FLAGS.model_name)
width = height
display_every_steps = FLAGS.display_every_steps
batch_size = FLAGS.batch_size

def preprocess_fn(value, batch_position, is_training):
    from pre import parse_example_proto
    image_buffer, label, filename = parse_example_proto(value)

    from pre import preprocess
    images = preprocess(image_buffer, is_training, FLAGS.model_name, height, width)

    return (images, label, filename)


def train_data_generator(batch_size):
    with tf.name_scope('train_batch_processing'):
        data_dir = FLAGS.train_data_dir
        glob_pattern = os.path.join(data_dir, 'train-*-of-*')
        file_names = gfile.Glob(glob_pattern)
        import random
        random.shuffle(file_names)
        ds = tf.data.TFRecordDataset.list_files(file_names)
        ds = ds.apply(interleave_ops.parallel_interleave(tf.data.TFRecordDataset, cycle_length=10))
        counter = tf.data.Dataset.range(batch_size)
        counter = counter.repeat()
        flags = tf.data.Dataset.from_tensors(tf.constant('train'))
        flags = flags.repeat()
        ds = tf.data.Dataset.zip((ds, counter, flags))
        ds = ds.prefetch(buffer_size=batch_size * 4)
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.repeat()
        ds = ds.apply(batching.map_and_batch(map_func=preprocess_fn, batch_size=batch_size, num_parallel_batches=10))
        ds = ds.prefetch(buffer_size=10)
        from tensorflow.contrib.data.python.ops import threadpool
        ds = threadpool.override_threadpool(ds, threadpool.PrivateThreadPool(10,
                                                                             display_name='input_pipeline_thread_pool'))
        #    ds_iterator = ds.make_initializable_iterator()
        return ds


def validation_data_generator(batch_size):
    with tf.name_scope('validation_batch_processing'):
        data_dir = FLAGS.validation_data_dir
        glob_pattern = os.path.join(data_dir, 'validation-*-of-*')
        file_names = gfile.Glob(glob_pattern)
        ds = tf.data.TFRecordDataset.list_files(file_names)
        ds = ds.apply(interleave_ops.parallel_interleave(tf.data.TFRecordDataset, cycle_length=10))
        counter = tf.data.Dataset.range(batch_size)
        counter = counter.repeat()
        flags = tf.data.Dataset.from_tensors(tf.constant('validation'))
        flags = flags.repeat()
        ds = tf.data.Dataset.zip((ds, counter, flags))
        ds = ds.prefetch(buffer_size=batch_size * 4)
        ds = ds.apply(batching.map_and_batch(map_func=preprocess_fn, batch_size=batch_size, num_parallel_batches=10))
        ds = ds.prefetch(buffer_size=10)
        from tensorflow.contrib.data.python.ops import threadpool
        ds = threadpool.override_threadpool(ds, threadpool.PrivateThreadPool(10,
                                                                             display_name='input_pipeline_thread_pool'))
        #    ds_iterator = ds.make_initializable_iterator()
        return ds

def main(argv=None):
    import json
    train_num_examples_path = FLAGS.train_data_dir + 'num_examples.json'
    validation_num_examples_path = FLAGS.validation_data_dir + 'num_examples.json'
    with open(train_num_examples_path) as load_f:
        load_dict = json.load(load_f)
        train_num_examples = load_dict['the total number of available samples']
    with open(validation_num_examples_path) as load_f:
        load_dict = json.load(load_f)
        validation_num_examples = load_dict['the total number of available samples']

    num_classes = FLAGS.num_class

    train_batch_size = batch_size
    eval_batch_size = batch_size
    init_learning_rate = Configure_file.configure_init_learning_rate(FLAGS.model_name)

    eval_every_epochs = FLAGS.eval_every_epochs
    eval_every_steps = int(eval_every_epochs * train_num_examples / train_batch_size)
    while eval_every_steps == 0:
        eval_every_epochs = eval_every_epochs + eval_every_epochs
        eval_every_steps = int(eval_every_epochs * train_num_examples / train_batch_size)

    if int(FLAGS.epochs * train_num_examples / train_batch_size) == 0:
        train_steps = 1
    else:
        train_steps = int(FLAGS.epochs * train_num_examples / train_batch_size)

    if int(validation_num_examples / eval_batch_size) == 0:
        eval_steps = 1
        eval_batch_size = validation_num_examples
    else:
        eval_steps = int(validation_num_examples / eval_batch_size)
    print('get information')
    fine_tune_flag = 'True' if FLAGS.fine_tune == 1 else 'False'
    early_stop_flag = 'True' if FLAGS.early_stop == 1 else 'False'
    print('---' * 20)
    print('model for classification : %s' % FLAGS.model_name)
    print('input height and width : %d' % height)
    print('whether to fine tune : %s' % fine_tune_flag)
    print('whether to early stop : %s' % early_stop_flag)
    print('number of train samples : %d' % train_num_examples)
    print('number of train classes : %d' % FLAGS.num_class)
    print('eval every epochs training %d' % eval_every_epochs)
    print('train steps : %d' % train_steps)
    print('eval steps : %d' % eval_steps)
    print('train batch size : %d' % train_batch_size)
    print('eval batch size : %d' % eval_batch_size)
    print('init_learning rate :%f' % init_learning_rate)
    print('lr deay policy :%s' % FLAGS.lr_decay)
    print('---' * 20)

    is_training_flag = tf.placeholder(
        tf.bool,
        shape=None,
        name='is_training_flag')
    weight_decay = tf.placeholder_with_default(Configure_file.configure_weight_decay(FLAGS.model_name), [])

    global_steps = tf.train.get_or_create_global_step()
    decay_steps = FLAGS.decay_epochs * train_num_examples / train_batch_size
    lr = Configure_file.configure_lr(init_learning_rate, FLAGS.lr_decay, decay_steps,
                                     FLAGS.decay_rate, global_steps)
    # (init_lr,decay_policy,decay_steps,decay_rate,global_steps,warm_lr=0.0001,warm_steps=0)
    #    tf.summary.scalar('learning_rate', lr)
    # opt = Configure_file.configure_optimizer(FLAGS.optimizer, lr)
    opt = Configure_file.model_optimizer(FLAGS.model_name, lr)
    # (optimizer,learning_rate)

    train_dataset = train_data_generator(batch_size=train_batch_size)
    validation_dataset = validation_data_generator(batch_size=eval_batch_size)
    iterator = tf.data.Iterator.from_structure(output_types=train_dataset.output_types,
                                               output_shapes=train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)
    images, labels, filenames = iterator.get_next()
    images = tf.reshape(images, shape=[-1, height, width, 3])
    labels = tf.reshape(labels, [-1])

    from nets import nets_factory
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=num_classes,
        weight_decay=weight_decay,
        is_training=is_training_flag)

    logits, end_points = network_fn(images)
    pred_soft = tf.nn.softmax(logits)
    values, indices = tf.nn.top_k(pred_soft, 1)
    if 'AuxLogits' in end_points:
        aux_cross_entropy = 0.4 * tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points['AuxLogits'], labels=labels,
                                                           name='aux_cross-entropy'))
    else:
        aux_cross_entropy = 0
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy'))
    loss = cross_entropy + aux_cross_entropy
    with tf.name_scope('accuracy'):
        top_1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(pred_soft, labels, 1), dtype=tf.float32), name='top_1')
        top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(pred_soft, labels, 5), dtype=tf.float32), name='top_5')

    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('aux_cross_entropy', aux_cross_entropy)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('top1', top_1)
    tf.summary.scalar('top5', top_5)

    for i in tf.global_variables():
        tf.summary.histogram(i.name.replace(":", "_"), i)

    #  recommended
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = opt.minimize(loss, global_step=global_steps)
        # ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    with tf.control_dependencies([train_step, loss, top_1, top_5]):  # , variables_averages_op,batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')
        # train_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([loss, top_1, top_5]):
        validation_op = tf.no_op(name='validation_op')

    config = tf.ConfigProto()
    config.allow_soft_placement = True

    checkpoint_dir = FLAGS.checkpoint_dir
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    # print(bn_moving_vars)
    store_restore_var_list = tf.trainable_variables() + bn_moving_vars  # + tf.moving_average_variables()
    # print(store_restore_var_list)
    saver = tf.train.Saver(store_restore_var_list, max_to_keep=1)

    checkpoint_basename = FLAGS.model_name + '.ckpt'

    if FLAGS.fine_tune == 1:
        exclusions = Configure_file.model_exclusions(FLAGS.model_name)
        print('variables to exclude :' + str(exclusions))
        variables_to_restore = []
        for var in store_restore_var_list:
            flag_break = 0
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    flag_break = 1
                    break
            if flag_break == 0:
                variables_to_restore.append(var)
        
		pre_train_saver = tf.train.Saver(variables_to_restore)
		#print(tf.gfile.IsDirectory(FLAGS.pre_trained_model_ckpt_path))
        if tf.gfile.IsDirectory(FLAGS.pre_trained_model_ckpt_path):
            #          print(tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*'))
            if tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*.ckpt'):
                ckpt_path = tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*.ckpt')[0]
                print('there is one ckpt file ' + str(ckpt_path))
            elif tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*.ckpt.*'):
                ckpts = tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*.ckpt.*')[0]
                ckpt_path = ckpts.rsplit('.', 1)[0]
                print('there is one ckpt file ' + str(ckpt_path))
            # imagenet pretrained model
            elif tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*.ckpt-*'):
                ckpts = tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*.ckpt-*')[0]
                ckpt_path = ckpts.rsplit('.', 1)[0]
                print('there is more than one ckpt files ' + str(ckpt_path))
            # pipline pretrained model
        else:
            ckpt_path = FLAGS.pre_trained_model_ckpt_path
            print(ckpt_path)

        #        ckpt_path = tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*')

        def load_pretrain(scaffold, sess):
            pre_train_saver.restore(sess, ckpt_path)

        scaffold = tf.train.Scaffold(init_fn=load_pretrain, saver=pre_train_saver)
    else:
        scaffold = None

    hooks = [
            tf.train.SummarySaverHook(save_steps=100, save_secs=None, output_dir=FLAGS.summary_dir, summary_writer=None,
                                  scaffold=None, summary_op=tf.summary.merge_all())
    ]
    early_stop_param = {}
    early_stop_param['count'] = 0
    early_stop_param['top1_max'] = 0
    with tf.train.MonitoredTrainingSession(checkpoint_dir=None,
                                           config=config,
                                           scaffold=scaffold,
                                           hooks=hooks,
                                           stop_grace_period_secs=1200
                                           ) as mon_sess:
        if FLAGS.early_stop == 1:
            print('start traing with early stop')
            mon_sess._coordinated_creator.tf_sess.run(train_init_op)
            time0 = time.time()
            import early_stop

            global_step = 0
            while global_step < train_steps:
                if (global_step) % eval_every_steps == 0 and global_step > 0:
                    print('start validating')
                    time_va = time.time()
                    mon_sess._coordinated_creator.tf_sess.run(validation_init_op)
                    loss_list = []
                    tt1 = []
                    lla_batch = []
                    ffile_batch = []
                    llo_batch = []
                    for i in range(eval_steps):
                        _, batch_loss, t1, la_batch, file_batch, lo_batch = mon_sess.run([validation_op, loss, top_1, labels, filenames, pred_soft],
                                                         feed_dict={is_training_flag: False, weight_decay: 0.0})
                        loss_list.append(batch_loss)
                        tt1.append(t1)
                        lla_batch.extend(la_batch)
                        ffile_batch.extend(file_batch)
                        llo_batch.extend(lo_batch.tolist())

                    validation_loss = numpy.mean(numpy.asarray(loss_list))
                    validation_top1 = numpy.mean(numpy.asarray(tt1))
                    mon_sess._coordinated_creator.tf_sess.run(train_init_op)
                    th = eval_steps * eval_batch_size / (time.time() - time_va)
                    print('done validating, validation loss is %f , top1 is %f , throughout is %f ' % (
                        validation_loss, validation_top1, th))
                    global_step = global_step + 1
                    # early_stop_param = early_stop.early_stop(validation_loss,early_stop_param)
                    if validation_top1 > early_stop_param['top1_max']:
                    	print('Saving checkpoints')
                        saver.save(
                            mon_sess._coordinated_creator.tf_sess,
                            save_path=os.path.join(checkpoint_dir, checkpoint_basename),
                            global_step=global_step,
                            latest_filename=None,
                            meta_graph_suffix='meta',
                            write_meta_graph=True,
                            write_state=True,
                            strip_default_attrs=False
                        )
                        performence.per(lla_batch,llo_batch,ffile_batch,FLAGS.result_dir)	
                    early_stop_param = early_stop.top1_early_stop(validation_top1, early_stop_param)
                    if early_stop_param['count'] >= 3:
                        print('train process should stop')
                        break
                if (global_step + 1) % display_every_steps == 0 and global_step > 0:
                    global_step, _, batch_loss, top1, top5 = mon_sess.run([global_steps, train_op, loss, top_1, top_5],
                                                                          feed_dict={is_training_flag: True})
                    th = display_every_steps * train_batch_size / (time.time() - time0)
                    time0 = time.time()
                    print('global_step: %d, train_loss: %f, top1: %f, top5: %f , throughout is %f' % (
                        global_step, batch_loss, top1, top5, th))
                else:
                    global_step, _ = mon_sess.run([global_steps, train_op], feed_dict={is_training_flag: True})

        else:
            print('start training without early stop')
            mon_sess._coordinated_creator.tf_sess.run(train_init_op)
            time0 = time.time()
            global_step = 0
            while global_step < train_steps:
                if (global_step + 1) % display_every_steps == 0 and global_step > 0:
                    global_step, _, batch_loss, top1, top5 = mon_sess.run([global_steps, train_op, loss, top_1, top_5],
                                                                          feed_dict={is_training_flag: True})
                    th = display_every_steps * train_batch_size / (time.time() - time0)
                    time0 = time.time()
                    print('global_step: %d, train_loss: %f, top1: %f, top5: %f , throughout is %f' % (
                        global_step, batch_loss, top1, top5, th))
                else:
                    global_step, _ = mon_sess.run([global_steps, train_op], feed_dict={is_training_flag: True})

	print('starting collecting data')
if __name__ == '__main__':
    tf.app.run()

