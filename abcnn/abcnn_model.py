from abcnn.graph import Graph
from abcnn import args
import tensorflow as tf


class AbcnnModel:
    def __init__(self):
        self.model = Graph(True, True)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def load_model(self, model_file):
        self.saver.restore(self.sess, model_file)
        print('load SUCCESS !')

    def train(self, p, h, y, p_eval, h_eval, y_eval):
        p_holder = tf.placeholder(
            dtype=tf.int32, shape=(
                None, args.seq_length), name='p')
        h_holder = tf.placeholder(
            dtype=tf.int32, shape=(
                None, args.seq_length), name='h')
        y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

        dataset = tf.data.Dataset.from_tensor_slices((p_holder, h_holder, y_holder))
        dataset = dataset.batch(args.batch_size).repeat(args.epochs)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 1

        with tf.Session(config=config)as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(
                iterator.initializer,
                feed_dict={
                    p_holder: p,
                    h_holder: h,
                    y_holder: y})
            steps = int(len(y) / args.batch_size)
            last_loss = 1.0
            for epoch in range(args.epochs):
                for step in range(steps):
                    p_batch, h_batch, y_batch = sess.run(next_element)
                    _, loss, acc = sess.run([self.model.train_op, self.model.loss, self.model.acc],
                                            feed_dict={self.model.p: p_batch,
                                                       self.model.h: h_batch,
                                                       self.model.y: y_batch,
                                                       self.model.keep_prob: args.keep_prob})
                    print('epoch:', epoch, ' step:', step, ' loss: ', loss, ' acc:', acc)

                loss_eval, acc_eval = sess.run([self.model.loss, self.model.acc],
                                               feed_dict={self.model.p: p_eval,
                                                          self.model.h: h_eval,
                                                          self.model.y: y_eval,
                                                          self.model.keep_prob: 1})
                print('loss_eval: ', loss_eval, ' acc_eval:', acc_eval)
                print('\n')

                if loss_eval < last_loss:
                    last_loss = loss_eval
                    self.saver.save(sess, f'../output/abcnn/abcnn.ckpt')

    def predict(self, p, h):
        with self.sess:
            prediction = self.sess.run(self.model.prediction,
                                  feed_dict={self.model.p: p,
                                             self.model.h: h,
                                             self.model.keep_prob: 1})

        return prediction


if __name__ == '__main__':
    from utils.load_data import load_char_data

    abcnn = AbcnnModel()

    # predict
    # p, h, y = load_char_data('input/test.csv', data_size=None)
    # abcnn.load_model('../output/abcnn/abcnn.ckpt')
    # prd = abcnn.predict(p, h)
    # print(prd)

    # train
    p, h, y = load_char_data('input/train.csv', data_size=None)
    p_eval, h_eval, y_eval = load_char_data('input/dev.csv', data_size=1000)
    abcnn.train(p, h, y, p_eval, h_eval, y_eval)
