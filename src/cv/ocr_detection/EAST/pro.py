txt_fn = im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt')
if not os.path.exists(txt_fn):
    print('text file {} does not exists'.format(txt_fn))
    continue

text_polys, text_tags = load_annoataion(txt_fn)
text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

text_polys[:, :, 0] *= resize_ratio_3_x
text_polys[:, :, 1] *= resize_ratio_3_y
ratio_y_h, ratio_x_w
score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)


