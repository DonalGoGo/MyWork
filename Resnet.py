import tensorflow as tf
import numpy as np
import os
import time

res_stru = {'50':[3,4,6,3],
            '101':[3,4,23,3],
            '152':[3,8,36,3]}
MODEL_FILE=""
BN_EPSILON = 0.001
CLASS_NUM=5
INIT_LEARNING_RATE=0.001
DECAY_STEPS=1000
DECAY_RATE=0.9
STAIRCASE=True
SAVE_PATH="./checkpoints/"
class Resnet(object):
    def __init__(self,input_x=None,input_y=None,is_train="True",resnet_type="50",model_file=None):
        self.bn_epsilon=BN_EPSILON
        self.model_save_file=SAVE_PATH+"resnet_50.npy"
        self.model_file = model_file
        self.class_num=CLASS_NUM
        self.is_train=is_train
        self.block_num_list=res_stru[resnet_type]
        self.global_step = tf.Variable(0, name="global_step")
        self.ema = tf.train.ExponentialMovingAverage(0.99,self.global_step)
        if input_x is not None:
            self.input_x=input_x
        else:
            self.input_x = tf.placeholder(tf.uint8, shape=[None, 224, 224, 3])
        if input_y is not None:
            self.input_y=input_y
        else:
            self.input_y=tf.placeholder(tf.float32,(None,CLASS_NUM),name='y_input')
        if self.model_file is not None:
            self.data_dict = np.load(self.model_file, encoding='latin1').item()
        print("start build_net...........")
        start=time.time()
        self.pred = self.build_net(self.input_x, self.is_train)
        end=time.time()
        print("build_net spend time:",(end-start))
        if self.is_train:
            self.cost, self.acc, self.step, self.output=self.cost_acc_step(self.pred,self.input_y)
        self.beta_list = tf.get_collection("betas")
        self.gamma_list = tf.get_collection("gammas")
        self.weight_list = tf.get_collection("weights")
        self.bias_list = tf.get_collection("biases")
        self.mean_list = tf.get_collection("means")
        self.variance_list = tf.get_collection("variances")
        self.maintain_averages_op = self.ema.apply(self.mean_list+self.variance_list)




    def save_model(self,sess):
        self.mean_shadow_list=[]
        self.variance_shadow_list = []
        for i in range(len(self.mean_list)):
            mean_shadow_var=self.ema.average(self.mean_list[i])
            self.mean_shadow_list.append(mean_shadow_var)
            variance_shadow_var=self.ema.average(self.variance_list[i])
            self.variance_shadow_list.append(variance_shadow_var)
        total_variable_list=self.beta_list+self.gamma_list+self.weight_list+self.bias_list+self.mean_shadow_list+self.variance_shadow_list
        total_variable_list_value=sess.run(total_variable_list)
        weight_dict={}
        for i in range(len(total_variable_list)):
            key=total_variable_list[i].name
            value=total_variable_list_value[i]
            weight_dict[key]=value
        np.save(self.model_save_file,weight_dict)





    def cost_acc_step(self,logits, y):
        output = tf.nn.softmax(logits)
        output = tf.identity(output, name='output')
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y, name='loss')
        cost = tf.reduce_mean(loss, name='cost')
        self.init_learning_rate = INIT_LEARNING_RATE
        self.decay_steps=DECAY_STEPS
        self.decay_rate=DECAY_RATE
        self.learning_rate=tf.train.exponential_decay(learning_rate=self.init_learning_rate,global_step=self.global_step,decay_steps=self.decay_steps,decay_rate=self.decay_rate,name="learning_rate",staircase=STAIRCASE)
        step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost,global_step=self.global_step)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)), tf.float32), name='acc')
        return cost, acc, step, output

    def build_net(self,input_x,is_train):
        input_x=input_x/255-0.5
        shape=input_x.get_shape().as_list()
        input_x = self.conv2d_bn(input_x, filter_shape=[7, 7, shape[3], 64], strides=[1, 2, 2, 1], name="conv_7_2")
        input_x = tf.nn.relu(input_x)
        input_x=tf.nn.max_pool(input_x,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

        for i in range(self.block_num_list[0]):
            input_x=self.block(input_x,name="blocks1_block",block_index=i,output_channel=64)

        input_x=self.pool_block(input_x,name="blocks2_block",block_index=0,output_channel=128)
        for i in range(1,self.block_num_list[1]):
            input_x = self.block(input_x, name="blocks2_block", block_index=i, output_channel=128)


        input_x=self.pool_block(input_x,name="blocks3_block",block_index=0,output_channel=256)
        for i in range(1,self.block_num_list[2]):
            input_x = self.block(input_x, name="blocks3_block", block_index=i, output_channel=256)

        input_x=self.pool_block(input_x,name="blocks4_block",block_index=0,output_channel=512)
        for i in range(1,self.block_num_list[3]):
            input_x = self.block(input_x, name="blocks4_block", block_index=i, output_channel=512)

        shape = input_x.get_shape().as_list()
        print("shape=",shape)
        input_x=tf.reshape(input_x,shape=[-1,shape[1]*shape[2]*shape[3]])
        input_x = self.fully_connection(input_x, out_dim=1000, name="fc_1")
        input_x=tf.nn.relu(input_x)
        input_x = self.fully_connection(input_x, out_dim=self.class_num, name="fc_2")
        return input_x

    def fully_connection(self,input_x,out_dim,name):
        shape = input_x.get_shape().as_list()
        if self.model_file is not None:
            weight = self.get_conv_filter(name)
            tf.add_to_collection("weights",weight)
            bias = self.get_conv_bias(name)
            tf.add_to_collection("biases", bias)
        else:
            weight=tf.Variable(tf.truncated_normal([shape[-1],out_dim],0.0,0.1),name=name+"_weight")
            bias = tf.Variable(tf.zeros(shape=(out_dim,), dtype=tf.float32), name=name + "_bias")
            tf.add_to_collection("weights", weight)
            tf.add_to_collection("biases", bias)
        input_x = tf.matmul(input_x, weight)
        input_x = tf.nn.bias_add(input_x, bias)
        return input_x


    def block(self,input_x,name,block_index,output_channel):
        shortcut=input_x
        name=name+"_"+str(block_index)
        shape=input_x.get_shape().as_list()
        input_x = self.conv2d_bn(input_x, filter_shape=[3, 3, shape[3], output_channel], strides=[1, 1, 1, 1], name=name+"_1")
        input_x=tf.nn.relu(input_x)
        input_x = self.conv2d_bn(input_x, filter_shape=[3, 3, output_channel, output_channel], strides=[1, 1, 1, 1], name=name+"_2")
        input_x=input_x+shortcut
        input_x = tf.nn.relu(input_x)
        return input_x
    def pool_block(self,input_x,name,block_index,output_channel):
        shortcut = input_x
        name = name + "_" + str(block_index)
        shape = input_x.get_shape().as_list()
        pooled_input = tf.nn.avg_pool(shortcut, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
        shortcut = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [shape[3] // 2,shape[3] // 2]])
        input_x = self.conv2d_bn(input_x, filter_shape=[3, 3, shape[3], output_channel], strides=[1, 2, 2, 1], name=name+"_1")
        input_x = tf.nn.relu(input_x)
        input_x = self.conv2d_bn(input_x, filter_shape=[3, 3, output_channel, output_channel], strides=[1, 1, 1, 1], name=name+"_2")
        input_x = input_x + shortcut
        input_x = tf.nn.relu(input_x)
        return input_x
    def conv2d_bn(self,input_x,filter_shape,strides,name):
        if self.model_file is not None:
            weight = self.get_conv_filter(name)
            bias = self.get_conv_bias(name)
            input_x = tf.nn.conv2d(input_x, filter=weight, strides=strides, padding="SAME")
            input_x=tf.nn.bias_add(input_x,bias=bias)
            if self.is_train:
                mean, variance = tf.nn.moments(input_x, axes=[0, 1, 2],name=name)
            else:
                mean,variance = self.get_mean_variance(name)
            beta,gamma=self.get_beta_gamma(name)
            tf.add_to_collection("means", mean)
            tf.add_to_collection("variances", variance)
            tf.add_to_collection("betas", beta)
            tf.add_to_collection("gammas", gamma)
            input_x=tf.nn.batch_normalization(input_x, mean, variance, beta, gamma, BN_EPSILON)
        else:
            filter=tf.Variable(tf.truncated_normal(filter_shape,0.0,0.1),name=name+"_weight")
            tf.add_to_collection("weights", filter)
            input_x=tf.nn.conv2d(input_x,filter=filter,strides=strides,padding="SAME")
            bias=tf.Variable(tf.zeros(shape=(filter_shape[-1],),dtype=tf.float32),name=name+"_bias")
            tf.add_to_collection("biases", bias)
            input_x = tf.nn.bias_add(input_x, bias=bias)
            input_x = self.batch_normalization_layer(input_x, name)
        return input_x

    def batch_normalization_layer(self,input_x, name):
        mean, variance = tf.nn.moments(input_x, axes=[0, 1, 2],name=name)
        tf.add_to_collection("means",mean)
        tf.add_to_collection("variances", variance)
        beta = tf.get_variable(name+'_beta', input_x.get_shape().as_list()[-1], tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        tf.add_to_collection("betas", beta)
        gamma = tf.get_variable(name+'_gamma', input_x.get_shape().as_list()[-1], tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        tf.add_to_collection("gammas", gamma)
        bn_layer = tf.nn.batch_normalization(input_x, mean, variance, beta, gamma, BN_EPSILON)
        return bn_layer


    def get_conv_filter(self, name):
        weight_name=name+"_weight"
        weight = self.data_dict[weight_name+":0"]
        weight_var = tf.Variable(np.array(weight),name=weight_name)
        tf.add_to_collection("weights",weight_var)
        if self.is_train:
            weight_decay = tf.multiply(tf.nn.l2_loss(weight_var),0.05,name='weight_loss')
            tf.add_to_collection("weight_loss_list",weight_decay)
        return weight_var
    def get_conv_bias(self,name):
        bias_name=name+"_bias"
        bias=self.data_dict[bias_name+":0"]
        bias_var=tf.Variable(np.array(bias),name=bias_name)
        tf.add_to_collection("biases", bias_var)
        return bias_var
    def get_beta_gamma(self,name):
        beta_name=name+"_beta"
        beta = self.data_dict[beta_name+":0"]
        beta_var = tf.Variable(np.array(beta), name=beta_name)
        tf.add_to_collection("betas", beta_var)
        gamma_name=name+"_gamma"
        gamma=self.data_dict[gamma_name+":0"]
        gamma_var=tf.Variable(np.array(gamma),name=gamma_name)
        tf.add_to_collection("gammas", gamma_var)
        return beta_var,gamma_var
    def get_mean_variance(self,name):
        mean_name=name+"/Squeeze/ExponentialMovingAverage:0"
        mean=self.data_dict[mean_name]
        mean_var=tf.Variable(np.array(mean), name=name+"/Squeeze")
        tf.add_to_collection("means", mean_var)
        variance_name=name+ "/Squeeze_1/ExponentialMovingAverage:0"
        variance=self.data_dict[mean_name]
        variance_var=tf.Variable(np.array(variance),name=name+"/Squeeze_1")
        tf.add_to_collection("variances", variance_var)
        return mean_var,variance_var
#resnet=Resnet()
