__author__ = 'PC-LiNing'

import logging
import math
import os
import random
import sys
import time
import numpy as np
import tensorflow as tf
import Model
import data_util


tf.app.flags.DEFINE_float("learning_rate", 1., "Learning rate.")
tf.app.flags.DEFINE_integer("size", 200, "Size of hidden layers.")
tf.app.flags.DEFINE_integer("embsize", 200, "Size of embedding.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("test_file", "test.article.txt", "Test filename.")
tf.app.flags.DEFINE_string("test_output", "output.txt", "Test output.")
tf.app.flags.DEFINE_string("train_dir", "model", "Training directory.")
tf.app.flags.DEFINE_string("tfboard", "tfboard", "Tensorboard log directory.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for testing.")
tf.app.flags.DEFINE_boolean("geneos", True, "Do not generate EOS. ")
tf.app.flags.DEFINE_float("max_gradient", 1.0, "Clip gradients l2 norm to this range.")
tf.app.flags.DEFINE_integer("batch_size", 80, "Batch size in training")
tf.app.flags.DEFINE_integer("beam_size", 5, "beam size in testing.")
tf.app.flags.DEFINE_integer("doc_vocab_size", 30000, "Document vocabulary size.")
tf.app.flags.DEFINE_integer("sum_vocab_size", 30000, "Summary vocabulary size.")
tf.app.flags.DEFINE_integer("max_train", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("max_iter", 200000, "Maximum training iterations.")
tf.app.flags.DEFINE_integer("steps_per_validation", 1000, "Training steps between validations.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 10000, "Training steps between checkpoints.")
tf.app.flags.DEFINE_string("checkpoint", "", "Checkpoint to load (use up-to-date if not set)")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets for sampling
_buckets = [(30, 10), (50, 20), (70, 20), (100, 20), (200, 30)]


# add <GO> and <EOS>
def create_bucket(source, target):
    data_set = [[] for _ in _buckets]
    for s, t in zip(source, target):
        t = [data_util.ID_GO] + t + [data_util.ID_EOS]
        for bucket_id, (s_size, t_size) in enumerate(_buckets):
            if len(s) <= s_size and len(t) <= t_size:
                data_set[bucket_id].append([s, t])
                break
    return data_set


def create_model(session, reverse_target_vocab_table, is_training):
    """Create model and initialize or load parameters in session."""
    model = Model.BiGRUModel(
        FLAGS.doc_vocab_size,
        FLAGS.sum_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.embsize,
        FLAGS.max_gradient,
        FLAGS.batch_size,
        FLAGS.beam_size,
        reverse_target_vocab_table,
        is_training,
        FLAGS.learning_rate)
    if FLAGS.checkpoint != "":
        ckpt = FLAGS.checkpoint
    else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt:
            ckpt = ckpt.model_checkpoint_path
    if ckpt and tf.train.checkpoint_exists(ckpt):
        logging.info("Reading model parameters from %s" % ckpt)
        model.saver.restore(session, ckpt)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train():
    logging.info("Preparing summarization data.")
    docid, sumid, doc_dict, sum_dict = data_util.load_data(FLAGS.data_dir + "/train.article.txt",
                                                           FLAGS.data_dir + "/train.title.txt",
                                                           FLAGS.data_dir + "/doc_dict.txt",
                                                           FLAGS.data_dir + "/sum_dict.txt",
                                                           FLAGS.doc_vocab_size, FLAGS.sum_vocab_size)

    val_docid, val_sumid = data_util.load_valid_data(FLAGS.data_dir + "/valid.article.filter.txt",
                                                     FLAGS.data_dir + "/valid.title.filter.txt",
                                                     doc_dict, sum_dict)

    with tf.Session() as sess:
        # Create model.
        logging.info("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        train_writer = tf.summary.FileWriter(FLAGS.tfboard, sess.graph)
        # create reverse table
        reverse_table = tf.contrib.lookup.index_to_string_table_from_file(vocabulary_file="sum_ordered_words.txt",
                                                                          default_value="<UNK>")
        model = create_model(sess, reverse_table, is_training=True)

        # Read data into buckets and compute their sizes.
        logging.info("Create buckets.")
        dev_set = create_bucket(val_docid, val_sumid)
        train_set = create_bucket(docid, sumid)

        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1])/train_total_size for i in range(len(train_bucket_sizes))]

        for (s_size, t_size), nsample in zip(_buckets, train_bucket_sizes):
            logging.info("Train set bucket ({}, {}) has {} samples.".format(s_size, t_size, nsample))

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = sess.run(model.global_step)
        while current_step <= FLAGS.max_iter:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, encoder_len, decoder_len = model.get_batch(train_set, bucket_id)
            current_step, step_loss = model.step(sess, encoder_inputs, decoder_inputs, encoder_len, decoder_len,
                                                 is_training=True, summary_writer=train_writer)
            step_time += (time.time() - start_time) / FLAGS.steps_per_validation
            loss += step_loss * FLAGS.batch_size / np.sum(decoder_len) / FLAGS.steps_per_validation
            # Once in a while, we save checkpoint.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

            # Once in a while, we print statistics and run evals.
            if current_step % FLAGS.steps_per_validation == 0:
                # Print statistics for the previous epoch.
                perplexity = np.exp(float(loss))
                logging.info("global step %d step-time %.2f ppl %.2f" % (model.global_step.eval(), step_time, perplexity))

                # Run evals on development set and print their perplexity.
                step_time, loss = 0.0, 0.0
                for bucket_id in range(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        logging.info("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, encoder_len, decoder_len = model.get_batch(dev_set, bucket_id)
                    eval_step, eval_loss = model.step(sess, encoder_inputs, decoder_inputs, encoder_len, decoder_len,
                                                      is_training=False, summary_writer=None)
                    eval_loss = eval_loss * FLAGS.batch_size / np.sum(decoder_len)
                    eval_ppx = np.exp(float(eval_loss))
                    logging.info("  eval: bucket %d ppl %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


def decode():
    # Load vocabularies.
    doc_dict = data_util.load_dict(FLAGS.data_dir + "/doc_dict.txt")
    sum_dict = data_util.load_dict(FLAGS.data_dir + "/sum_dict.txt")
    if doc_dict is None or sum_dict is None:
        logging.warning("Dict not found.")
    data = data_util.load_test_data(FLAGS.data_dir + "/" + FLAGS.test_file, doc_dict)

    with tf.Session() as sess:
        # Create model and load parameters.
        logging.info("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        # create reverse table
        reverse_table = tf.contrib.lookup.index_to_string_table_from_file(vocabulary_file=FLAGS.data_dir + "/sum_ordered_words.txt",
                                                                          default_value="<UNK>")
        reverse_table.init.run()
        model = create_model(sess, reverse_table, is_training=False)
        result = []
        for idx, token_ids in enumerate(data):
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, encoder_len, decoder_len = model.get_batch(
                    {0: [(token_ids, [data_util.ID_GO, data_util.ID_EOS])]}, 0)
            # repeat
            if encoder_inputs.shape[0] == 1:
                encoder_inputs = np.repeat(encoder_inputs, FLAGS.batch_size, axis=0)
                encoder_len = np.repeat(encoder_len, FLAGS.batch_size, axis=0)
            # outputs = [batch_size,length]
            step, outputs = model.inference(sess, encoder_inputs, encoder_len)
            # If there is an EOS symbol in outputs, cut them at that point.
            target_output = [item[0].decode() for item in outputs]
            if data_util.MARK_EOS in target_output:
                target_output = target_output[:target_output.index(data_util.MARK_EOS)]
            gen_sum = " ".join(target_output)
            result.append(gen_sum)
            logging.info("Finish {} samples. :: {}".format(idx, gen_sum[:75]))
        with open(FLAGS.test_output, "w") as f:
            for item in result:
                print(item, file=f)


def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                        datefmt='%b %d %H:%M')
    try:
        os.makedirs(FLAGS.train_dir)
    except:
        pass
    tf.app.run()