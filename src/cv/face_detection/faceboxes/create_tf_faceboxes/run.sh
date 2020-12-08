#python create_tfrecords.py --image_dir=/export/luozhuang/data_tf/faceboxes/dataset/FDDB/images/ --annotations_dir=/export/luozhuang/data_tf/faceboxes/dataset/FDDB/annotations/ --output=/export/luozhuang/data_tf/faceboxes/val_tfrecords/ --num_shards=20

#python create_tfrecords.py --image_dir=/export/luozhuang/data_tf/faceboxes/dataset/WIDER_full/images/ --annotations_dir=/export/luozhuang/data_tf/faceboxes/dataset/WIDER_full/annotations/ --output=/export/luozhuang/data_tf/faceboxes/train_tfrecords/ --num_shards=150

/bin/bash cmd.sh --data_dir /export/luozhuang/data_tf/faceboxes/dataset/FDDB/ --output_path ./output/
