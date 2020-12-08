# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import threading
lock = threading.Lock()
import random
import cv2
import codecs
import json
import sys 
reload(sys) 

sys.setdefaultencoding('utf-8')

tf.app.flags.DEFINE_string('data_dir', '/mnt/shared/easydl/train_data_dir', '''''')

tf.app.flags.DEFINE_integer('num_shard', 10, '''''')
tf.app.flags.DEFINE_integer('num_thread', 10, '''''')

tf.app.flags.DEFINE_string('output_dir', '/mnt/shared/easydl/train_data_dir_tfreocrd', '''''')
tf.app.flags.DEFINE_string('config_dir', '/mnt/shared/easydl/train_data_dir_tfreocrd', '''''')
FLAGS = tf.app.flags.FLAGS

new_config = {
	"null_code": 84,
	"name": "MYDATASET",
	"items_to_descriptions": {
		"text": "A unicode string.",
		"image": "A [150 x 150 x 3] colorimage.",
		"num_of_views": "A number of different views stored within the image.",
		"length": "A length of the encodedtext.",
		"label": "Characters codes."
	},
	"image_shape": [150, 150, 3],
	"charset_filename": "charset_size.txt",
	"max_sequence_length": 275,
	"num_of_views": 1,
	"splits": {
		"test": {
			"pattern": "test*",
			"size": 7
		},
		"train": {
			"pattern": "train*",
			"size": 1000
		}
	}
}
def build_char(data_dir):
    config_file = os.path.join(FLAGS.config_dir,'newdataset_config_json.json')
    #f = open(config_file,'r')
    #new_config = json.load(f)
    #f.close()
        
    char_dic = []
    label_dic = {}
    length = 0
    
    files = tf.gfile.Glob(data_dir + '/*.txt')

    for f in files:
        text_in_image = ''
        import codecs
        for line in codecs.open(f,'r',encoding='utf-8').readlines():
            text = ''.join(line.split(',')[8:])
            text = text.replace('\r','')
            text = text.replace('\n','')
            if '#' in text:
                continue
            else:
                text_in_image += text
        label_dic[os.path.abspath(f)] = text_in_image
        length = len(text_in_image) if len(text_in_image) > length else length
        char_dic.extend(list(text_in_image))
        char_dic = list(set(char_dic))

    if 'train' in data_dir:
        key = range(len(char_dic))
        char_set = dict(zip(char_dic,key))
        char_set['length'] = length
        import codecs
        with codecs.open(os.path.join(FLAGS.output_dir,"charset.json"),'w',encoding='utf-8') as json_file:
            json.dump(char_set, json_file, sort_keys=True, indent=4, separators=(',', ': '))
        new_config['null_code'] = len(char_set.keys())-1
        new_config['max_sequence_length'] = length
        new_config['splits']['train']['size'] = len(files)
        #f = open(config_file,'w') 
        #json.dump(new_config,f)
        #f.close() 
    else:
        import codecs
        with codecs.open(os.path.join(FLAGS.output_dir,"charset.json"),'r',encoding='utf-8') as json_file:
            char_set=json.load(json_file)
            length = char_set['length']
        new_config['splits']['test']['size'] = len(files)
        f = open(config_file,'w')
        json.dump(new_config,f,indent=4)
        f.close()
    char_set.pop('length')
    null_char_id = len(char_set.keys())
    if 'train' in data_dir:
        char_path = os.path.join(FLAGS.output_dir,'charset_size.txt')
        import codecs
        fw = codecs.open(char_path,'w+',encoding='utf-8')
        for i in char_set:
            tt = str(char_set[i]) + '\t' + i + '\n'
            fw.write(tt)
        tt = str(null_char_id) + '\t' + '<nul>'
        fw.write(tt)
        fw.close()
    return char_set, length, null_char_id, label_dic

def encode_utf8_string(text, length, charset, null_char_id):
    char_ids_padded = [null_char_id]*length
    char_ids_unpadded = [null_char_id]*len(text)
    for i in range(len(text)):
        hash_id = charset[text[i]]
        char_ids_padded[i] = hash_id
        char_ids_unpadded[i] = hash_id
    return char_ids_padded, char_ids_unpadded

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def process_image(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img,(150,150))
    _,jpgVector = cv2.imencode('.jpg',img)
    image_data = jpgVector.tostring()
    #image_data = tf.gfile.GFile(filename,'r').read()   
    """ 
    sess = tf.Session()
    try:
        image = tf.image.decode_image(image_data, channels = 3).eval(session=sess)
        height = image.shape[0]
        width = image.shape[1]

        assert image.shape[2] == 3
        assert len(image.shape) == 3
    except:
        pass
    """
    width = 150
    height =150
    return image_data, width, height

def convert_to_example(image_file, image_buffer, text, char_ids_padded, char_ids_unpadded, width, height):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/format': _bytes_feature("JPG"),
        'image/encoded': _bytes_feature(image_buffer),
        'image/class':_int64_feature(char_ids_padded),
        'image/unpadded_class': _int64_feature(char_ids_unpadded),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'orig_width': _int64_feature(width),
        'image/text': _bytes_feature(text),
        'image/image_file':_bytes_feature(image_file)
        }
    ))
    return example

def process_image_files_batch(thread_index, ranges, name, filename_label_dic, charset, length, null_char_id, num_shards):

    num_threads = len(ranges)
    num_shards_per_batch = int(num_shards / num_threads)
    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        filenames = filename_label_dic.keys()
        for i in files_in_shard:
            text = filename_label_dic[filenames[i]]
            image_filename = filenames[i].replace('.txt','.jpg')
            try:
                char_ids_padded, char_ids_unpadded = encode_utf8_string(
                    text, length, charset, null_char_id)
                image_buffer, width, height = process_image(image_filename)
            except Exception as e:
                print(e)
                print(image_filename + ' is abandoned')
                lock.acquire()
                try:
                    num_example = num_example - 1
                    num_corrupted = num_corrupted + 1
                finally:
                    lock.release()
                continue
            text = str(text)
            example = convert_to_example(image_filename, image_buffer, text, char_ids_padded, char_ids_unpadded, width, height)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
        writer.close()

def process_image_files(subset, filename_label_dic,charset,length,null_char_id):

    num_shards = FLAGS.num_shard
    num_thread = FLAGS.num_thread
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    num_images = len(filename_label_dic.keys())
    spacing = np.linspace(0, num_images, num_thread + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    coord = tf.train.Coordinator()
    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, subset, filename_label_dic, charset, length, null_char_id, num_shards)
        t = threading.Thread(target=process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)

def main(unused_argv):

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    if not tf.gfile.Exists(FLAGS.config_dir):
        tf.gfile.MakeDirs(FLAGS.config_dir)
    file_list = tf.gfile.ListDirectory(FLAGS.data_dir)
    i = file_list[0] if 'train' in file_list[0] else file_list[1]
    charset,length,null_char_id,train_filename_label_dic = build_char(os.path.join(FLAGS.data_dir,i))
    i = file_list[0] if 'test' in file_list[0] else file_list[1] 
    charset,length,null_char_id,test_filename_label_dic = build_char(os.path.join(FLAGS.data_dir,i))

    process_image_files('train',train_filename_label_dic,charset,length,null_char_id)
    process_image_files('test',test_filename_label_dic,charset,length,null_char_id)

    print('----' * 15)
    print('finished')
    print('----' * 15)

if __name__ == '__main__':
    tf.app.run()
