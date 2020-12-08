import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

import locality_aware_nms as nms_locality
import lanms
from icdar import generate_rbox,load_annoataion,check_and_validate_polys

tf.app.flags.DEFINE_string('test_data_path', '', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '', '')
tf.app.flags.DEFINE_string('output_dir', '', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

import model
from icdar import restore_rectangle

FLAGS = tf.app.flags.FLAGS

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, text_polys, text_tags, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    
    text_polys[:, :, 0] *= ratio_w
    text_polys[:, :, 1] *= ratio_h

    score_map, geo_map, training_mask = generate_rbox((int(resize_h), int(resize_w)), text_polys, text_tags)
    return im, (ratio_h, ratio_w), score_map, geo_map, training_mask


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list


    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
        if FLAGS.geometry == 'RBOX':
            input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
        else:
            input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='input_geo_maps')
        input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)
        model_loss = model.loss(input_score_maps, f_score, input_geo_maps, f_geometry, input_training_masks)
        total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            model_loss_list = []
            total_loss_list = []
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                txt_fn = im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt')
                h,w,_ = im.shape
                text_polys, text_tags = load_annoataion(txt_fn)
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
                start_time = time.time()
                im_resized, (ratio_h, ratio_w), score_map, geo_map, training_mask = resize_image(im, text_polys, text_tags)

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                image_model_loss, image_all_loss, score, geometry = sess.run(
                                                                             [model_loss, total_loss, f_score, f_geometry], 
                                                                             feed_dict={input_images: [im_resized],
                                                                                        input_score_maps:[score_map[::4, ::4, np.newaxis].astype(np.float32)],
                                                                                        input_geo_maps:[geo_map[::4, ::4, :].astype(np.float32)],
                                                                                        input_training_masks:[training_mask[::4, ::4, np.newaxis].astype(np.float32)]})
                model_loss_list.append(image_model_loss)
                total_loss_list.append(image_all_loss)           
                timer['net'] = time.time() - start

                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                '''
                print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))
                '''
                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                duration = time.time() - start_time
                #print('[timing] {}'.format(duration))

                # save to file
                if boxes is not None:
                    res_file = os.path.join(
                        FLAGS.output_dir,
                        '{}.txt'.format(
                            os.path.basename(im_fn).split('.')[0]))

                    with open(res_file, 'w') as f:
                        for box in boxes:
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                            ))
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])
            print(np.mean(model_loss_list),np.mean(total_loss_list))
if __name__ == '__main__':
    tf.app.run()
