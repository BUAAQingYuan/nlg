__author__ = 'PC-LiNing'

import random
import numpy as np
import tensorflow as tf
import data_util
from tensorflow.python.layers import core as layers_core


class BiGRUModel(object):
    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 state_size,
                 embedding_size,
                 max_gradient,
                 batch_size,
                 beam_size,
                 reverse_target_vocab_table,
                 is_training,
                 learning_rate):

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.state_size = state_size

        self.encoder_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None])
        self.decoder_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None])
        self.decoder_targets = tf.placeholder(tf.int32, shape=[self.batch_size, None])
        self.encoder_len = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.decoder_len = tf.placeholder(tf.int32, shape=[self.batch_size])

        encoder_fw_cell = tf.contrib.rnn.GRUCell(state_size)
        encoder_bw_cell = tf.contrib.rnn.GRUCell(state_size)
        decoder_cell = tf.contrib.rnn.GRUCell(state_size)

        if is_training:
            encoder_fw_cell = tf.contrib.rnn.DropoutWrapper(encoder_fw_cell, output_keep_prob=0.50)
            encoder_bw_cell = tf.contrib.rnn.DropoutWrapper(encoder_bw_cell, output_keep_prob=0.50)
            decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, output_keep_prob=0.50)

        with tf.variable_scope("seq2seq", dtype=tf.float32):
            with tf.variable_scope("encoder"):
                encoder_emb = tf.get_variable("embedding", [source_vocab_size, embedding_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                # encoder_inputs_emb = [batch_size,length,embedding_size]
                encoder_inputs_emb = tf.nn.embedding_lookup(encoder_emb, self.encoder_inputs)
                # encoder_outputs = ([batch_size,num_step,state_size],[batch_size,num_step,state_size])
                # encoder_states = ([batch_size,state_size],[batch_size,state_size])
                encoder_outputs, encoder_states = tf.nn.bidirectional_dynamic_rnn(encoder_fw_cell, encoder_bw_cell,
                                                                                  encoder_inputs_emb,sequence_length=self.encoder_len, dtype=tf.float32)

            with tf.variable_scope("init_state"):
                self.init_state = tf.contrib.layers.fully_connected(tf.concat(encoder_states, 1), state_size)
                self.init_state.set_shape([self.batch_size, state_size])
                # att_states = [batch_size,num_step,state_size * 2]
                self.att_states = tf.concat(encoder_outputs, 2)
                self.att_states.set_shape([self.batch_size, None, state_size*2])

            with tf.variable_scope("decoder"):
                # decode
                decoder_emb = tf.get_variable("embedding", [target_vocab_size, embedding_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                decoder_inputs_emb = tf.nn.embedding_lookup(decoder_emb, self.decoder_inputs)
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_emb, self.decoder_len)

                # num_units = attention layer size or attention depth
                attention = tf.contrib.seq2seq.BahdanauAttention(state_size, self.att_states, self.encoder_len)
                att_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention, state_size)

                decoder = tf.contrib.seq2seq.BasicDecoder(cell=att_decoder_cell,
                                                          helper=helper,
                                                          initial_state=att_decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=self.init_state),
                                                          output_layer=layers_core.Dense(self.target_vocab_size))
                # outputs = (rnn_output,sample_id)
                # rnn_output = [batch_size,length,state_size],sample_id = [batch_size,length]
                # final_state =
                # final_length = [batch_size,]
                outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
                # outputs_logits = [batch_size,length,state_size]
                outputs_logits = outputs[0]
                self.outputs = outputs_logits
                # weights = [batch_size,length]
                weights = tf.sequence_mask(self.decoder_len, dtype=tf.float32)
                # outputs_logits = [batch_size,length,num_decoder_symbols]
                # docoder_targets = [batch_size,length]
                # loss_t = [batch_size,length]
                loss_t = tf.contrib.seq2seq.sequence_loss(outputs_logits, self.decoder_targets, weights,
                                                          average_across_timesteps=False, average_across_batch=False)
                self.loss = tf.reduce_sum(loss_t) / self.batch_size
                params = tf.trainable_variables()
                opt = tf.train.AdadeltaOptimizer(self.learning_rate, epsilon=1e-6)
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient)
                self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
                tf.summary.scalar('loss', self.loss)

            with tf.variable_scope("decoder", reuse=True):
                # Inference
                # tile inputs for beam search decoder
                enc_outputs = tf.contrib.seq2seq.tile_batch(self.att_states, multiplier=self.beam_size)
                # add attention,attention is just a computation,no parameters.
                inference_length = tf.contrib.seq2seq.tile_batch(self.encoder_len, multiplier=self.beam_size)
                inference_initial_state = tf.contrib.seq2seq.tile_batch(self.init_state, multiplier=self.beam_size)
                inference_attention = tf.contrib.seq2seq.BahdanauAttention(state_size, enc_outputs, inference_length)
                inference_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, inference_attention, state_size)
                start_tokens = tf.fill([self.batch_size], data_util.ID_GO)
                end_token = data_util.ID_EOS
                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=inference_decoder_cell,
                    embedding=decoder_emb,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=inference_decoder_cell.zero_state(batch_size*self.beam_size, tf.float32).clone(cell_state=inference_initial_state),
                    beam_width=self.beam_size,
                    output_layer=layers_core.Dense(self.target_vocab_size),
                    length_penalty_weight=0.0
                )
                predict_inference_output, context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder)
                # translations = [length, batch_size, beam_width]
                translations = predict_inference_output.predicted_ids
                # prediction_ids = [length, batch_size]
                prediction_ids = translations[:, :, 0]
                # prediction_ids = [batch_size,length]
                prediction_ids = tf.transpose(prediction_ids, perm=[1, 0])
                # index look up
                # prediction_words = [batch_size,length]
                self.prediction_words = reverse_target_vocab_table.lookup(tf.to_int64(prediction_ids))

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
        self.summary_merge = tf.summary.merge_all()

    def step(self, session, encoder_inputs, decoder_inputs, encoder_len, decoder_len, is_training, summary_writer):
        if encoder_inputs.shape[1] != max(encoder_len):
            raise ValueError("encoder_inputs and encoder_len does not fit")
        if decoder_inputs.shape[1] != max(decoder_len) + 1:
            raise ValueError("decoder_inputs and decoder_len does not fit")
        input_feed = {}
        input_feed[self.encoder_inputs] = encoder_inputs
        input_feed[self.decoder_inputs] = decoder_inputs[:, :-1]
        input_feed[self.decoder_targets] = decoder_inputs[:, 1:]
        input_feed[self.encoder_len] = encoder_len
        input_feed[self.decoder_len] = decoder_len
        # training
        if is_training:
            step, merge, loss, _ = session.run([self.global_step, self.summary_merge, self.loss, self.updates], input_feed)
        else:
            step, merge, loss = session.run([self.global_step, self.summary_merge, self.loss], input_feed)
        if summary_writer:
            summary_writer.add_summary(merge, step)
        return step, loss

    def inference(self, session, encoder_inputs, encoder_len):
        input_feed = {}
        input_feed[self.encoder_inputs] = encoder_inputs
        input_feed[self.encoder_len] = encoder_len
        step, outputs = session.run([self.global_step, self.prediction_words], input_feed)
        return step, outputs

    def add_pad(self, data, fixlen):
        data = map(lambda x: x + [data_util.ID_PAD] * (fixlen - len(x)), data)
        data = list(data)
        return np.asarray(data)

    def get_batch(self, data, bucket_id):
        encoder_inputs, decoder_inputs = [], []
        encoder_len, decoder_len = [], []
        # Get a random batch of encoder and decoder inputs from data,
        # and add GO to decoder.
        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
            encoder_inputs.append(encoder_input)
            encoder_len.append(len(encoder_input))
            decoder_inputs.append(decoder_input)
            decoder_len.append(len(decoder_input))

        batch_enc_len = max(encoder_len)
        batch_dec_len = max(decoder_len)
        encoder_inputs = self.add_pad(encoder_inputs, batch_enc_len)
        decoder_inputs = self.add_pad(decoder_inputs, batch_dec_len)
        encoder_len = np.asarray(encoder_len)
        # decoder_input has both <GO> and <EOS>
        # len(decoder_input)-1 is number of steps in the decoder.
        decoder_len = np.asarray(decoder_len) - 1
        return encoder_inputs, decoder_inputs, encoder_len, decoder_len
