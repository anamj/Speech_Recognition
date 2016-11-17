import functools
import numpy as np
import tensorflow as tf
import read_data
from sklearn.model_selection import train_test_split


def out_one_hot(y_data):
    one_hot = np.zeros((1, 127))
    y_data_onehot = np.zeros((len(y_data), 127))
    for i in range(len(y_data)):
        one_hot[0][y_data[i]] = 1
        y_data_onehot[i, :] = one_hot
        one_hot[0][y_data[i]] = 0
    return y_data_onehot

def broadZero(data):
    res=np.zeros((2023,data.shape[1]))
    res[:len(data),:]=data
    return res

# myfunc = deco(myfunc)
def variable_scope(function,scope=None):
    #@functools.wraps(function)
    attribute='_cache_'+function.__name__
    name= scope or function.__name__
    def decorator(self):
        if not hasattr(self,attribute):
            with tf.variable_scope(name):
                setattr(self,attribute,function(self))
        return getattr(self,attribute)
    return decorator



class Model:
    #recommend to always use tf.get_variable
    def __init__(self,data,label,early_stop):
        #input and output shape=[Batch Size, Sequence Length, Input Dimension]
        self.data=data
        self.label=label
        self.early_stop=early_stop
        self.input_dimension=39 #MFCC
        self.output_dimension=127
        self.hidden_layer_size=60
        self.hidden_layer_num=2
        self.batch_size=500
        # not add drop-out part now
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_layer_size, forget_bias=0.0, state_is_tuple=True)
        self.cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*self.hidden_layer_num,state_is_tuple=True)
        #We initialize the hidden states to zero.
        #We then use the final hidden states of the current minibatch as the initial hidden state of the subsequent minibatch
        self.initial_state=self.cell.zero_state(batch_size=self.batch_size,dtype=tf.float32)
        self.final_state=self.initial_state
        self.prediction
        self.optimize
        self.error

    @property
    @variable_scope
    def prediction(self):
        #dynamic rnn cell_output shape is [Batch Size, Sequence Length, size]
        print(np.array(self.final_state).shape)
        cell_output, state = tf.nn.dynamic_rnn(self.cell,inputs=self.data,initial_state=self.final_state,dtype=tf.float32,sequence_length=self.early_stop)
        output = tf.reshape(cell_output, [-1, self.hidden_layer_size])
        with tf.variable_scope("out_layer"):
            weights = tf.get_variable("softmax_w", shape=[self.hidden_layer_size, self.output_dimension],
                                      initializer=tf.random_normal_initializer())
            bias = tf.get_variable("softmax_b", shape=[self.output_dimension],
                                   initializer=tf.constant_initializer())
        logits = tf.matmul(output, weights) + bias
        self.final_state=state
        #rememver to set main loop state to 0
        return logits
    
    @property
    @variable_scope
    def optimize(self):
        #loss = tf.nn.seq2seq.sequence_loss_by_example([self.prediction],[tf.reshape(self.label, [-1])],[tf.ones([self.batch_size * self.sequence_length])])
        #selflabel should be 2 dimension
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.label))
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(loss)

    @property
    @variable_scope
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        #mistakes =tf.reshape(mistakes,-1)
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


def main():
    data = list(read_data.read_joint_feat_alignment(alidir="mono_ali", set="train_20h", type="mfcc", cmvn=True, deltas=True))
    e_stop = [np.array(x[1]).shape[0] for x in data]
    print('e_stop has built.')
    two_dim_x_list=[]
    two_dim_y_list=[]
    for x in data:
        two_dim_x_list.append(broadZero(np.array(x[1])))
        two_dim_y_list.append(broadZero(out_one_hot(x[2])))
    #data = [(0, 1, 1)] X_data's shape should be 7502*2023*39
    '''
    X_data = np.array(two_dim_x_list)
    y_data = np.array(two_dim_y_list)
    print(X_data.shape)
    print(y_data.shape)
    '''
    # Now you can train (and save) your model
    trainData = two_dim_x_list
    trainLabel = two_dim_y_list
    train_stop=e_stop
    '''
    test_ratio=int(len(X_data)*0.7)
    trainData = X_data[:test_ratio,:,:]
    trainLabel = y_data[:test_ratio,:,:]
    train_stop=e_stop[:test_ratio]
    valiData = X_data[test_ratio:,:,:]
    valiLabel = y_data[test_ratio:,:,:]
    valiStop = e_stop[test_ratio:]
    '''
    RANDOM_SEED = 10
    tf.set_random_seed(RANDOM_SEED)

    max_sequence_length = 2023
    del data
    with tf.Graph().as_default():
        input_data =tf.placeholder("float", shape=[None, max_sequence_length, 39])
        output_data = tf.placeholder("float",shape=[None, 127])
        early_stop=tf.placeholder(tf.int32,[None]) #[batch_size]
        model = Model(input_data, output_data, early_stop)
        with tf.Session() as sess:
            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            batch_size = 500     #len(data)=7502 20h 7502*0.67=5026 also should change model.batchsize
            number_of_batch = len(trainData) // batch_size
            number_of_epoch = 5
            for epoch in range(number_of_epoch):
                # no shuffle currently
                print("You are now at epoch %d!" % epoch)
                print('How many batches: %d'%number_of_batch)
                for i in range(number_of_batch-1):
                    inData = trainData[i * batch_size:(i + 1) * batch_size]
                    outData = trainLabel[i * batch_size:(i + 1) * batch_size]
                    outData = np.concatenate(outData)
                    sess.run(model.optimize, feed_dict={input_data: inData, output_data: outData, early_stop: train_stop[i * batch_size:(i + 1) * batch_size]})
                i=i+1
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                model.final_state = model.initial_state
                #pre_result=sess.run(model.prediction,feed_dict={input_data:valiData,early_stop: valiStop})
                #print(np.array(pre_result).shape)
                #validation_accuracy=np.mean(pre_result==np.argmax(valiLabel,1))
                validation_accuracy = sess.run(model.error, feed_dict={input_data: trainData[i * batch_size:(i + 1) * batch_size], output_data: np.concatenate(trainLabel[i * batch_size:(i + 1) * batch_size]), early_stop: train_stop[i * batch_size:(i + 1) * batch_size]})
                print("The accuracy of validation part is: %f" % validation_accuracy)
                model.final_state = model.initial_state
            print("Task over. Model has been built.")
            # Save=tf.train.Saver()
            # save_path=Save.save(sess,"train_20h_model.ckpt")  
      
if __name__ == '__main__': 
    main()
