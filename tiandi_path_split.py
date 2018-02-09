import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import random
import time
from Resnet import Resnet

# python hyperparameters
LEARNING_RATE = 0.0001
EPOCH = 100
CLASS_NUM = 5

# const
CHANEL = 1
res_stru = {'50':[3,4,6,3],
            '101':[3,4,23,3],
            '152':[3,8,36,3]}


def process_data_set():
    TRAIN_N_PATH = './data/train/0'
    TRAIN_D_PATH = './data/train/1'
    TRAIN_W_PATH = './data/train/2'
    TRAIN_Z_PATH = './data/train/3'
    TRAIN_P_PATH = './data/train/4'

    TEST_N_PATH = './data/test/0'
    TEST_D_PATH = './data/test/1'
    TEST_W_PATH = './data/test/2'
    TEST_Z_PATH = './data/test/3'
    TEST_P_PATH = './data/test/4'

    train_n = glob.glob(TRAIN_N_PATH + '/*.*')
    train_d = glob.glob(TRAIN_D_PATH + '/*.*')
    train_w = glob.glob(TRAIN_W_PATH + '/*.*')
    train_z = glob.glob(TRAIN_Z_PATH + '/*.*')
    train_p = glob.glob(TRAIN_P_PATH + '/*.*')

    test_n = glob.glob(TEST_N_PATH + '/*.*')
    test_d = glob.glob(TEST_D_PATH + '/*.*')
    test_w = glob.glob(TEST_W_PATH + '/*.*')
    test_z = glob.glob(TEST_Z_PATH + '/*.*')
    test_p = glob.glob(TEST_P_PATH + '/*.*')
    train_feature = train_n + train_d + train_w + train_z + train_p
    train_label = [0 for i in range(len(train_n))] + \
                  [1 for i in range(len(train_d))] + \
                  [2 for i in range(len(train_w))] + \
                  [3 for i in range(len(train_z))] + \
                  [4 for i in range(len(train_p))]

    test_feature = test_n + test_d + test_w + test_z + test_p
    test_label = [0 for i in range(len(test_n))] + \
                 [1 for i in range(len(test_d))] + \
                 [2 for i in range(len(test_w))] + \
                 [3 for i in range(len(test_z))] + \
                 [4 for i in range(len(test_p))]


    # 对数据进行清洗
    shuffle_index = [i for i in range(len(train_feature))]
    np.random.shuffle(shuffle_index)
    train_x = np.array(train_feature)[shuffle_index]
    train_y = np.array(train_label)[shuffle_index]

    shuffle_index = [i for i in range(len(test_feature))]
    np.random.shuffle(shuffle_index)
    test_x = np.array(test_feature)[shuffle_index]
    test_y = np.array(test_label)[shuffle_index]
    valid_x = np.concatenate((train_x, train_x, train_x), axis=0)[:1000]
    valid_y = np.concatenate((train_y, train_y, train_y), axis=0)[:1000]

    train_y = np.eye(CLASS_NUM)[train_y]
    valid_y = np.eye(CLASS_NUM)[valid_y]
    test_y = np.eye(CLASS_NUM)[test_y]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

assert process_data_set()[-3].shape == (1000,5)


# train data augmentation
def image_augment_from_path(image_path):
    img = cv2.imread(image_path, 0)
    # 这里要根据channel 改变图像

    # resize图片到336*x或者x*336或者336*336
    target_size = 224
    if img.shape[0] / img.shape[1] < 0.9:
        resize_factor = target_size / img.shape[0]
        width = np.int32(img.shape[1] * resize_factor)
        img = cv2.resize(img, (width, target_size), interpolation=cv2.INTER_NEAREST)
    elif img.shape[0] / img.shape[1] > 1.11:
        resize_factor = target_size / img.shape[1]
        height = np.int32(img.shape[0] * resize_factor, interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (target_size, height))
    else:
        # todo 这里变形了 应该用padding
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    # 随机裁剪区域
    v_start = np.int(random.uniform(0, img.shape[0] - 224))
    h_start = np.int(random.uniform(0, img.shape[1] - 224))
    img = img[v_start:v_start + 224, h_start:h_start + 224]

    # 随机水平或者垂直翻转
    is_flip_v = bool(random.getrandbits(1))
    is_flip_h = bool(random.getrandbits(1))
    if is_flip_v and is_flip_h:
        img = cv2.flip(img, -1)
    elif (is_flip_v and (not is_flip_h)):
        img = cv2.flip(img, 0)
    elif ((not is_flip_v) and is_flip_h):
        img = cv2.flip(img, 1)
    a = np.random.randint(80, 120) / 100;
    b = np.random.randint(-20, 20);
    img = np.uint8(np.clip((a * img + b), 0, 255))
    img = np.reshape(img, (224, 224, 1))
    return img


def gen_batch(x, y, batch_size):
    if len(x) % batch_size == 0:
        batch_num = len(x) // batch_size
    else:
        batch_num = len(x) // batch_size + 1
    index = 0
    for i in range(batch_num):
        batch_path = x[index:index + batch_size]
        batch_y = y[index:index + batch_size]
        batch_x = [image_augment_from_path(batch_path[j]) for j in range(len(batch_path))]
        yield batch_x, batch_y
        index = index + batch_size

        # import cv2
        # import numpy as np
        # import matplotlib.pyplot as plt
        # import random
        # %matplotlib inline
        # plt.imshow(np.reshape(image_augment_from_path('D:/teddy/teddy/project/A/data/test/00.jpg'),[224,224]),cmap='gray')
        # print(image_augment_from_path('D:/teddy/teddy/project/A/data/test/00.jpg'))


def show_train(y_list):
    y = y_list
    x = np.linspace(1,len(y_list),len(y_list))
    plt.plot(x,y)


def image_test_stream(image_stream, width, height):
    #     img = tf.read_file(path_holder)
    #     raw_img = tf.image.decode_jpeg(img,channels=CHANEL)
    raw_image = tf.reshape(image_stream, [height, width, -1])

    # 需要设计网络
    # 3行*6列 尽量保持图片不变形
    img = tf.image.resize_images(raw_image, [672, 1344], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    num_width = tf.constant(6)
    num_height = tf.constant(3)
    cond = lambda i, num_width, num_height, img, imgs: tf.less(i, num_height)

    def body(i, num_width, num_height, img, imgs):
        j = tf.constant(0)
        inner_cond = lambda i, j, num_width, num_height, img, imgs: tf.less(j, num_width)

        def inner_body(i, j, num_width, num_height, img, imgs):
            cut_img = img[i * 224: i * 224 + 224, j * 224: j * 224 + 224, :]
            cut_img = tf.reshape(cut_img, [1, 224, 224, CHANEL])
            imgs = tf.concat([imgs, cut_img], 0)
            return i, j + 1, num_width, num_height, img, imgs

        i, j, num_width, num_height, img, imgs = tf.while_loop(inner_cond,
                                                               inner_body,
                                                               (i, j, num_width, num_height, img, imgs),
                                                               (i.get_shape(), j.get_shape(), num_width.get_shape(),
                                                                num_height.get_shape(), img.get_shape(),
                                                                tf.TensorShape([None, 224, 224, CHANEL])))
        return i + 1, num_width, num_height, img, imgs

    i = tf.constant(0)
    imgs = tf.zeros([0, 224, 224, CHANEL], tf.uint8)
    i, num_width, num_height, img, imgs = tf.while_loop(cond,
                                                        body,
                                                        (i, num_width, num_height, img, imgs),
                                                        (i.get_shape(), num_width.get_shape(), num_height.get_shape(),
                                                         img.get_shape(), tf.TensorShape([None, 224, 224, CHANEL])))

    return imgs


def input_holder():
#     x_test_path_ = tf.placeholder(tf.string,name='image_address')
    y_ = tf.placeholder(tf.float32,(None,CLASS_NUM),name='y_input')
    train_flag_ = tf.placeholder(tf.bool,name='train_flag')
    image_ = tf.placeholder(tf.uint8,name='raw_image')
    width_ = tf.placeholder(tf.int32,name='width')
    height_ = tf.placeholder(tf.int32,name='height')
    return y_,train_flag_,image_,width_,height_


# 残差网络
def convolutional_block(x, filters, is_train):
    short_cut = tf.layers.conv2d(x, filters * 4, 1, 2, padding='same')
    x = tf.layers.conv2d(x, filters, 1, 2, activation=tf.nn.relu, padding='same')
    x = tf.layers.batch_normalization(x, training=is_train)
    x = tf.layers.conv2d(x, filters, 3, 1, activation=tf.nn.relu, padding='same')
    x = tf.layers.batch_normalization(x, training=is_train)
    x = tf.layers.conv2d(x, filters * 4, 1, 1, padding='same')
    x = tf.layers.batch_normalization(x, training=is_train)
    x = x + short_cut
    x = tf.nn.relu(x)
    return x


def identity_block(x, filters, is_train):
    short_cut = x
    x = tf.layers.conv2d(x, filters, 1, 1, activation=tf.nn.relu, padding='same')
    x = tf.layers.batch_normalization(x, training=is_train)
    x = tf.layers.conv2d(x, filters, 3, 1, activation=tf.nn.relu, padding='same')
    x = tf.layers.batch_normalization(x, training=is_train)
    x = tf.layers.conv2d(x, filters * 4, 1, 1, padding='same')
    x = tf.layers.batch_normalization(x, training=is_train)
    x = x + short_cut
    x = tf.nn.relu(x)
    return x


def identity_block_depth(x, filters, is_train):
    short_cut = tf.layers.conv2d(x, filters * 4, 1, 1, padding='same')
    x = tf.layers.conv2d(x, filters, 1, 1, activation=tf.nn.relu, padding='same')
    x = tf.layers.batch_normalization(x, training=is_train)
    x = tf.layers.conv2d(x, filters, 3, 1, activation=tf.nn.relu, padding='same')
    x = tf.layers.batch_normalization(x, training=is_train)
    x = tf.layers.conv2d(x, filters * 4, 1, 1, padding='same')
    x = tf.layers.batch_normalization(x, training=is_train)
    x = x + short_cut
    x = tf.nn.relu(x)
    return x


def big_identity_block(x, block_n, is_train):
    x = identity_block_depth(x, 64, is_train)
    for i in range(block_n):
        x = identity_block(x, 64, is_train)
    return x


def big_convolutional_block(x, block_n, filters, is_train):
    x = convolutional_block(x, filters, is_train)
    for i in range(block_n - 1):
        x = identity_block(x, filters, is_train)
    return x


def resnet(x, is_train, depth='50'):
    x = x / 255 - 0.5
    x = tf.layers.conv2d(x, 64, 7, 2, padding='same', name='base_cnn')
    x = tf.layers.batch_normalization(x, name='base_bn', training=is_train)
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, 2, 2, name='base_pooling')
    with tf.variable_scope('cfg0'):
        x = big_identity_block(x, res_stru[depth][0], is_train)
    with tf.variable_scope('cfg1'):
        x = big_convolutional_block(x, res_stru[depth][1], 128, is_train)
    with tf.variable_scope('cfg2'):
        x = big_convolutional_block(x, res_stru[depth][2], 256, is_train)
    with tf.variable_scope('cfg3'):
        x = big_convolutional_block(x, res_stru[depth][3], 512, is_train)
    x = tf.layers.average_pooling2d(x, 7, 1)
    x = tf.reshape(x, (-1, 2048))
    x = tf.layers.dense(x, 1000, activation=tf.nn.relu)
    logits = tf.layers.dense(x, CLASS_NUM, name='logits')

    return logits



# VGG11层模型
def vgg11(x, is_train):
    # VGG 11层模型
    x = x / 255 - 0.5
    x = tf.layers.conv2d(x, 64, 3, 1, padding='same', name='cnnlayer1')
    x = tf.layers.batch_normalization(x, name='bn1', training=is_train)
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name='pooling1')
    # 112*112*64

    x = tf.layers.conv2d(x, 128, 3, 1, padding='same', name='cnnlayer2')
    x = tf.layers.batch_normalization(x, name='bn2', training=is_train)
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name='pooling2')
    # 56*64*128

    x = tf.layers.conv2d(x, 256, 3, 1, padding='same', name='cnnlayer3')
    x = tf.layers.batch_normalization(x, name='bn3', training=is_train)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, 256, 3, 1, padding='same', name='cnnlayer4')
    x = tf.layers.batch_normalization(x, name='bn4', training=is_train)
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name='pooling3')
    # 28*28*256

    x = tf.layers.conv2d(x, 512, 3, 1, padding='same', name='cnnlayer5')
    x = tf.layers.batch_normalization(x, name='bn5', training=is_train)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, 512, 3, 1, padding='same', name='cnnlayer6')
    x = tf.layers.batch_normalization(x, name='bn6', training=is_train)
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name='pooling4')
    # 14*14*512

    x = tf.layers.conv2d(x, 512, 3, 1, padding='same', name='cnnlayer7')
    x = tf.layers.batch_normalization(x, name='bn7', training=is_train)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, 512, 3, 1, padding='same', name='cnnlayer8')
    x = tf.layers.batch_normalization(x, name='bn8', training=is_train)
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name='pooling5')
    # 7*7*512

    x = tf.reshape(x, [-1, 7 * 7 * 512])
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, name='fc1')
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, name='fc2')
    x = tf.layers.dense(x, 1000, activation=tf.nn.relu, name='fc3')
    logits = tf.layers.dense(x, CLASS_NUM, name='logits')

    return logits


def cost_acc_step(logits, y):

    output = tf.nn.softmax(logits)
    output = tf.identity(output, name='output')
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y, name='loss')
    cost = tf.reduce_mean(loss, name='cost')
    step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)), tf.float32), name='acc')
    return cost, acc, step, output


def train(model, epoch=EPOCH, batch_size=16, dev=False, checkpoint_path='tmp'):
    tf.reset_default_graph()

    # graph
    y_, train_flag_, image_, width_, height_ = input_holder()
    image = image_test_stream(image_, width_, height_)
    x_ = tf.placeholder_with_default(image, (None, 224, 224, CHANEL), name='x_train')
    print("开始重建网络resnet")
    start = time.time()
    #logits = resnet(x_, is_train=train_flag_)
    resnet_obj=Resnet(input_x=x_,input_y=y_,is_train=True)
    cost=resnet_obj.cost
    acc = resnet_obj.acc
    step = resnet_obj.step
    output = resnet_obj.output

    end = time.time()
    print("重建网络resnet时间为：%f秒" % (end - start))

    min_pred = tf.reduce_min(output)
    pred = tf.identity(min_pred, name='pred')

    which_box = tf.argmin(output)
    which_box=tf.cast(which_box,dtype=tf.int32)
    row = which_box // 6
    column = which_box[0] % 6
    points = [width_ * column / 6, height_ * row / 3, width_ * (column + 1) / 6, height_ * (row + 1) / 3]
    points = tf.identity(points, name='points')


    train_x, train_y, valid_x, valid_y, test_x, test_y = process_data_set()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=20,)

    counter = 0
    start_time = time.clock()
    current_time = time.clock()
    cost_list = []
    acc_list = []
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    best_valid_acc = 0

    saver.save(sess, './checkpoints/' + checkpoint_path + 'zxq2max')

    with tf.device("/gpu:0"):
        for i in range(epoch):
            for x_batch, y_batch in gen_batch(train_x, train_y, batch_size):
                #print("counter=%d"%counter)
                if counter % 100 == 0:
                    val_cost_list = []
                    val_acc_list = []
                    for test_x_batch, test_y_batch in gen_batch(test_x, test_y, batch_size):
                        step_test_cost, step_test_acc = sess.run([cost, acc],
                                                                 feed_dict={x_: test_x_batch, y_: test_y_batch,
                                                                            train_flag_: False})
                        val_cost_list.append(step_test_cost)
                        val_acc_list.append(step_test_acc)
                    test_cost = np.mean(val_cost_list)
                    test_acc = np.mean(val_acc_list)
                    if best_valid_acc < test_acc:
                        resnet_obj.save_model(sess)
                        best_valid_acc = test_acc
                        saver.save(sess, './checkpoints/' + checkpoint_path + 'zxq2max')


                    time_per_step = time.clock() - current_time
                    seconds = time_per_step * (epoch - i) * len(train_x) / (100 * batch_size)
                    m, s = divmod(seconds, 60)
                    h, m = divmod(m, 60)
                    eta = "%02d:%02d:%02d" % (h, m, s)
                    current_time = time.clock()
                    print('Step:{:>4d}, ETA:{}, Test_cost:{:.5F},best_valid_acc:{:.5F} ,Test_acc:{:.4F}'.format(counter,eta,test_cost,best_valid_acc,test_acc))

                if counter % 500 == 0 and counter > 0:
                    # saver.save(sess,'./checkpoints/'+checkpoint_path,global_step=i)
                    print('checkpoint save as ' + checkpoint_path + str(i))
                _, __, step_cost, step_acc ,_= sess.run([step, extra_update_ops, cost, acc,resnet_obj.maintain_averages_op],feed_dict={x_: x_batch, y_: y_batch, train_flag_: True})

                cost_list.append(step_cost)
                acc_list.append(step_acc)
                counter = counter + 1
                if dev and counter % 10 == 0:
                    update_op=[resnet_obj.mean_list[0], resnet_obj.ema.average(resnet_obj.mean_list[0]),resnet_obj.learning_rate]
                    mean1, sh_mean1 ,learn_rate= sess.run(update_op,feed_dict={x_: x_batch, y_: y_batch, train_flag_: False})
                    cur_acc = np.mean(acc_list[-10:])
                    cur_cost = np.mean(cost_list[-10:])
                    print("Step: ", counter, " Cost: ", cur_cost, " Accuracy: ", cur_acc,"learn_rate=",learn_rate)

                #print("counter=%d" % counter)
        sess.close()
        show_train(acc_list)


# load model + frezee
sess = train(resnet,checkpoint_path='res',dev=True)