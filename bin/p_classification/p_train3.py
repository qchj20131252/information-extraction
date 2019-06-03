"""
This module to train the relation classification model
"""
from tqdm import tqdm
import os
import sys
import time

import paddle
import paddle.fluid as fluid
import six

import numpy as np

import p_data_reader3
import p_model3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lib")))
import conf_lib3


def train(conf_dict, data_reader, use_cuda=False):
    """
    Training of p classification model
    """
    label_dict_len = data_reader.get_dict_size('label_dict')
    # input layer
    word = fluid.layers.data(name='word_data', shape=[1], dtype='int64', lod_level=1)
    postag = fluid.layers.data(name='token_pos', shape=[1], dtype='int64', lod_level=1)
    arc_relation = fluid.layers.data(name="arc_relation", shape=[1], dtype='int64', lod_level=1)
    arc_head = fluid.layers.data(name="arc_head", shape=[1], dtype="int64", lod_level=1)
    # label
    target = fluid.layers.data(name='target', shape=[label_dict_len], dtype='float32', lod_level=0)
    # NN: embedding + lstm + pooling
    feature_out = p_model3.db_lstm(data_reader, word, postag, arc_relation, arc_head, conf_dict)
    # loss function for multi-label classification
    class_cost = fluid.layers.sigmoid_cross_entropy_with_logits(x=feature_out, label=target)
    avg_cost = fluid.layers.mean(class_cost)
    acc = fluid.layers.accuracy(input=feature_out, label=target)

    test_program =fluid.default_main_program().clone(for_test=True)
    # optimization method
    sgd_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=2e-3, )

    sgd_optimizer.minimize(avg_cost)

    train_batch_reader = paddle.batch(paddle.reader.shuffle(data_reader.get_train_reader(), buf_size=8192),
                                      batch_size=conf_dict['batch_size'])
    test_batch_reader = paddle.batch(paddle.reader.shuffle(data_reader.get_test_reader(), buf_size=8192),
                                     batch_size=conf_dict['batch_size'])

    def train_test(train_test_program, train_test_feed, train_test_reader):
        acc_set = []
        avg_loss_set = []
        for test_data in train_test_reader():
            avg_loss_np, acc_np = exe.run(
                program=train_test_program,
                feed=train_test_feed.feed(test_data),
                fetch_list=[avg_cost, acc])
            acc_set.append(float(acc_np))
            avg_loss_set.append(float(avg_loss_np))
        # get test acc and loss
        acc_val_mean = np.array(acc_set).mean()
        avg_loss_val_mean = np.array(avg_loss_set).mean()
        return avg_loss_val_mean, acc_val_mean

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[word, postag, arc_relation, arc_head, target], place=place)
    exe = fluid.Executor(place)

    save_dirname = conf_dict['p_model_save_dir']

    def train_loop(main_program, trainer_id=0):
        """start train"""
        exe.run(fluid.default_startup_program())

        start_time = time.time()
        batch_id = 0
        lists = []
        best_epoch = 100000
        for pass_id in six.moves.xrange(conf_dict['pass_num']):
            pass_start_time = time.time()
            cost_sum, cost_counter = 0, 0
            for data in tqdm(train_batch_reader()):
                cost = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_cost])
                cost = cost[0]
                cost_sum += cost
                cost_counter += 1
                if batch_id % 10 == 0 and batch_id != 0:
                    print("batch %d finished, second per batch: %02f" % (batch_id, (time.time() - start_time) / batch_id), file=sys.stderr)

                # cost expected, training over
                if float(cost) < 0.01:
                    pass_avg_cost = cost_sum / cost_counter if cost_counter > 0 else 0.0
                    print("%d pass end, cost time: %02f, avg_cost: %f" % (pass_id, time.time() - pass_start_time,
                                                                          pass_avg_cost), file=sys.stderr)
                    save_path = os.path.join(save_dirname, 'final')
                    fluid.io.save_inference_model(save_path, ['word_data', 'token_pos', 'arc_relation', 'arc_head'],
                                                  [feature_out], exe, params_filename='params')
                    return
                batch_id = batch_id + 1

            #test for epoch
            avg_loss_val, acc_val = train_test(train_test_program=test_program,train_test_reader=test_batch_reader,
                                               train_test_feed=feeder)
            if best_epoch > avg_loss_val:
                best_epoch = avg_loss_val
                save_path = os.path.join(save_dirname, 'final')
                fluid.io.save_inference_model(save_path, ['word_data', 'token_pos', 'arc_relation', 'arc_head'],
                                              [feature_out], exe, params_filename='params')

            print("Test with Epoch %d, avg_cost: %s, acc: %s" % (pass_id, avg_loss_val, acc_val))
            lists.append((pass_id, avg_loss_val, acc_val))

            # save the model once each pass ends
            pass_avg_cost = cost_sum / cost_counter if cost_counter > 0 else 0.0
            print("Train with Epoch %d, cost time: %02f, avg_cost: %f" % (pass_id, time.time() - pass_start_time, pass_avg_cost),
                  file=sys.stderr)
            save_path = os.path.join(save_dirname, 'pass_%04d-%f' % (pass_id, pass_avg_cost))
            fluid.io.save_inference_model(save_path, ['word_data', 'token_pos', 'arc_relation', 'arc_head'],
                                          [feature_out], exe, params_filename='params')
        # find the best pass
        best = sorted(lists, key=lambda list: float(list[1]))[0]
        print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
        print('The classification accuracy is %.2f%%' % (float(best[2]) * 100))

        return

    train_loop(fluid.default_main_program())


def main(conf_dict, use_cuda=False):
    """Train main function"""
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        print('No GPU', file=sys.stderr)
        return
    data_generator = p_data_reader3.RcDataReader(
        wordemb_dict_path=conf_dict['word_idx_path'],
        postag_dict_path=conf_dict['postag_dict_path'],
        arc_relation_dict_path=conf_dict['arc_relation_dict_path'],
        label_dict_path=conf_dict['label_dict_path'],
        train_data_list_path=conf_dict['train_data_path'],
        test_data_list_path=conf_dict['test_data_path'])

    train(conf_dict, data_generator, use_cuda=use_cuda)


if __name__ == '__main__':
    # Load the configuration file
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--conf_path", type=str,
    #                     help="conf_file_path_for_model. (default: %(default)s)",
    #                     required=True)
    # args = parser.parse_args()
    conf_file_path = "../../conf/IE_extraction.conf"
    conf_dict = conf_lib3.load_conf(conf_file_path)
    use_gpu = True if conf_dict.get('use_gpu', 'False') == 'True' else False
    main(conf_dict, use_cuda=use_gpu)
