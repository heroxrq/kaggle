import numpy as np
import tensorflow as tf
from PIL import Image
from tf_unet import unet, image_util

from config import *

data_provider = image_util.ImageDataProvider(TRAIN_RESIZED_DIR + "/*", data_suffix='.jpg', mask_suffix='_mask.gif')
net = unet.Unet(channels=3, n_class=2, cost='cross_entropy', layers=4, features_root=16)
trainer = unet.Trainer(net, batch_size=16, optimizer='momentum')

# print "----------start train----------"
# trainer.train(data_provider, CHECKPOINT_DIR, training_iters=10, epochs=500, dropout=0.5, display_step=1, restore=False, write_graph=False)
# print "----------end train----------"

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # print "----------start save----------"
    # net.save(sess, MODEL_FILE)
    # print "----------end save----------"

    pic = np.array(Image.open(TRAIN_RESIZED_DIR + '/00087a6bd4dc_01.jpg'), np.float32)
    Image.open(TRAIN_RESIZED_DIR + '/00087a6bd4dc_01.jpg').show()
    print pic.shape
    print type(pic)
    pic = np.reshape(pic, (1, RESIZED_HEIGHT, RESIZED_WIDTH, 3))
    print pic.shape
    print type(pic)

    print "----------start predict----------"
    prediction = net.predict(MODEL_FILE, pic)
    print "----------end predict----------"

    print prediction.shape
    print type(prediction)

    res = np.reshape(prediction, (164, 164, 2))[:, :, 1]
    print res.shape
    print type(res)
    res = np.where(res > 0.5, 1.0, 0.0)
    print res.shape
    print type(res)

    print res * 255
    img = Image.fromarray(res * 255)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.show()
   # img = util.to_rgb(res)
   # util.save_image(img, test_pic_path)


#    pic_label = np.array(Image.open(base_dir + '/dataset/train/00087a6bd4dc_01_mask.gif'), np.float32)
#    print pic_label.shape
#    print type(pic_label)

#    img = util.combine_img_prediction(np.reshape(pic, (1, 1280, 1918, 3)), np.reshape(pic, (1, 1280, 1918, 3)), prediction)
#    util.save_image(img, "prediction.jpg")

