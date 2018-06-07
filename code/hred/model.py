#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 22:52:34 2017

@author: suman
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest
from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear

class Embedder(object):
    """
    Class for Embedding the tokens of the utterances.
    """
    def __init__(self, FLAGS, initializer=None, name="embedder"):
        
        """
        FLAGS: the configuration FLAGS,
        initializer: initializer function for initializing the embedding matrix
        """
        V, d = FLAGS.vocab_size,FLAGS.word_emb_dim
        with tf.variable_scope(name):
            nil_word_slot = tf.constant(np.zeros([1, d]), dtype=tf.float32) #embedding for the <PAD> token
            self.E = tf.get_variable("emb_mat", dtype=tf.float32,
                                         shape=[V, d], initializer=initializer,trainable=True)
            
            self.emb_mat = tf.concat([nil_word_slot,self.E],0)

    def __call__(self, word, name="embedded"):
        """
        Lookup the embedding of "word" from the embedding matrix named "emb_mat"
        """
        out = tf.nn.embedding_lookup(self.emb_mat, word, name=name)
        return out


def rnn_cell(FLAGS, dropout, scope,decoder_cell=False):
    """
    Creates and returns the encoder or decoder cell
    
    args:
        
        FLAGS: the configuration FLAGS,
        dropout: the keep probability of the cell units,
        scope: the scope of the RNN cell parameters,
        decoder_cell: whether this is a decode_cell or not
    """
    with tf.variable_scope(scope):
        if FLAGS.rnn_unit == 'gru':
            rnn_cell_type = tf.contrib.rnn.GRUCell
        elif FLAGS.rnn_unit == 'lstm':
            rnn_cell_type = tf.contrib.rnn.BasicLSTMCell
        else:
            raise Exception("Choose a valid RNN unit type.")

        single_cell = rnn_cell_type(FLAGS.hidden_units)

        # Add dropout wrapper on the RNN cell
        single_cell = tf.contrib.rnn.DropoutWrapper(single_cell,
            output_keep_prob=dropout)

        if decoder_cell:
            return single_cell
        else:
            stacked_cell = tf.contrib.rnn.MultiRNNCell(
                    [single_cell] * FLAGS.num_layers)  # Stack RNN cells to create layers of the RNN
            return stacked_cell


def _extract_argmax_and_embed(W_embedding, output_projection,
    update_embedding=True):
    """
    Function to embed previous decoder step's output and feed it to the next decoder step's input
    
    args:
        W_embedding: embedding matrix from Embedder class to lookup embeddings of generated words.
        output_projection: The parameter matrix and bias vector for projecting the hidden decoder state to vocab size
        update_embedding: To update the parameters while feeding previous decoder output
    
    """
    def loop_function(prev, _):
        prev = tf.matmul(prev, output_projection[0]) + output_projection[1]
        prev_symbol = tf.argmax(prev, axis=1)
        embedded_prev_symbol = tf.nn.embedding_lookup(W_embedding, prev_symbol)
        if not update_embedding:
            embedded_prev_symbol = tf.stop_gradient(embedded_prev_symbol)
        return embedded_prev_symbol
    
    return loop_function

def decoder(decoder_inputs, initial_state,
    cell, output_size, loop_function=None, dtype=None,
    scope=None):
    """
    The Decoder Function which returns the decoder hidden states after decoding the whole output
    args:
        decoder_inputs: The inputs to the decoder, either the targets during training or the previous decoder output during inference.
        initial_state: The tensor used to initialize the first decoder step cell.
        cell: The decoder cell returned by the rnn_cell function.
        output_size: The number of decoder hidden state units.
        loop_function: The function that embeds the previous decoder step's output and provides as input to next decoder step
        dtype: the data type
        scope: the scope of the attention decoder
    
    """
    with tf.variable_scope(scope or 'decoder', dtype=dtype) as scope:

        state = initial_state[0]
        outputs = []
        prev = None

        for i, inp in enumerate(decoder_inputs):

            if i > 0:
                tf.get_variable_scope().reuse_variables()

            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            
            cell_outputs, state = cell(inp, state) 

            if loop_function is not None:
                prev = cell_outputs
            outputs.append(cell_outputs)

        return outputs, state

def embedding_decoder(decoder_inputs, initial_state,
    cell, num_symbols,output_projection,W_embedding,
    feed_previous, update_embedding_for_previous=True,
    scope=None, dtype=None):
    """
    Function to get the proper loop function, set the proper scope and then call the decoder
    args:
        decoder_inputs: The inputs to the decoder, either the targets during training or the previous decoder output during inference.
        initial_state: The tensor used to initialize the first decoder step cell.
        cell: The decoder cell returned by the rnn_cell function.
        num_symbols: The number of vocab symbols
        output_projection: The parameter matrix and bias vector for projecting the hidden decoder state to vocab size
        W_embedding: embedding matrix from Embedder class to lookup embeddings of generated words.
        feed_previous: Feed the previous decoder step's output to the next decoder step's input or not
        update_embedding_for_previous: To update the output_projection parameters while feeding previous decoder output
        dtype: the data type
        scope: the scope of the decoder
        
    """

    output_size = cell.output_size
    if feed_previous==False:
         tf.get_variable_scope().reuse_variables()
    
    with tf.variable_scope(scope or "embedding_decoder",
        dtype=dtype) as scope:
   
        loop_function = _extract_argmax_and_embed(
            W_embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None

        return decoder(
            decoder_inputs,
            initial_state,
            cell,
            output_size=output_size,
            loop_function=loop_function)
        
        
def decoder_logits(FLAGS, outputs, scope):
    """
    Project the decoder hidden states to vocab size logits
    
    args:
        FLAGS: the configuration FLAGS,
        outputs: decoder hidden states after the decoding is done
        scope: the appropriate scope of the decoder
    """
    with tf.variable_scope(scope, reuse=True):
         
         W_softmax = tf.get_variable("W_softmax",

                shape=[FLAGS.hidden_units, FLAGS.vocab_size],
                dtype=tf.float32)       
       
         b_softmax = tf.get_variable("b_softmax",
                shape=[FLAGS.vocab_size],
                dtype=tf.float32)      
            
         l=[]
         for i in outputs:
             
             logits = tf.matmul(i, W_softmax) + b_softmax #unnormalized logits
             l.append(logits)
         return l
     
class HREDModel(object):
    """
    The Hierarchical Recurrent Encoder-Decoder Model
    """
    def __init__(self,sess,FLAGS):

        tf.set_random_seed(1234)
        self.sess=sess
        self.batch_size =FLAGS.batch_size,
        self.max_enc_size = FLAGS.max_enc_size
        self.max_dec_size = FLAGS.max_sent_size
        self.vocab_size = FLAGS.vocab_size
        self.emb_dim = FLAGS.word_emb_dim
        initializer = tf.random_uniform_initializer(-np.sqrt(3), np.sqrt(3))

        # Placeholders        
        self.encoder_inputs = tf.placeholder(tf.int32,shape=[None,self.max_enc_size,None], name='encoder_inputs')
        self.decoder_inputs = tf.placeholder(tf.int32,shape=[None, self.max_dec_size],
                                             name='decoder_inputs')
        self.decoder_targets = tf.placeholder(tf.int32,shape=[None,self.max_dec_size],name='targets')
        self.enc_seq_len = tf.placeholder(tf.int32,shape=[None, ],name="enc_seq_lens")
        self.dec_seq_len = tf.placeholder(tf.int32,shape=[None, ],name="dec_seq_lens")
        self.sent_seq_len = tf.placeholder(tf.int32,[None,self.max_enc_size],name="sent_seq_len")
        self.forward_only = tf.placeholder(tf.bool,name="foward_only")
        self.target_weights =tf.placeholder(tf.int64,[None,self.max_dec_size],name="Target_lengths")
        self.dropout=tf.placeholder(tf.float32,name="dropout")
   
        with tf.variable_scope("embedding") as scope:
            A = Embedder(FLAGS, initializer=initializer, name='Embedder')
            self.enc_emb = A(self.encoder_inputs, name='embedded_encoder')   #[32,1900,100]
            self.dec_ip_emb = A(self.decoder_inputs, name='embedded_decoder_ip')    #[32,30,100]
            
        with tf.variable_scope('sent_encoder_cell') as scope:
            self.sent_encoder_stacked_cell = rnn_cell(FLAGS, self.dropout,
                scope=scope)
        
        with tf.variable_scope('context_encoder_cell') as scope:
            self.encoder_stacked_cell = rnn_cell(FLAGS, self.dropout,
                scope=scope)
        
        with tf.variable_scope('encoder') as scope:
           
            self.c_emb_list=tf.split(self.enc_emb,
                               num_or_size_splits=self.max_enc_size,axis=1)
            self.utterance_reps=[]
            
            for i in range(len(self.c_emb_list)):
                utterance=tf.squeeze(self.c_emb_list[i],axis=1)
                with tf.variable_scope("ut") as scope:
                    if i>0:
                        scope.reuse_variables()
                    _,self.ut_encoder_final_state=tf.nn.dynamic_rnn(
                                self.sent_encoder_stacked_cell,
                                utterance,
                                self.sent_seq_len[:,i],
                                dtype=tf.float32)
                
                self.utterance_reps.append(self.ut_encoder_final_state[0])

            self.context_inputs=tf.stack(self.utterance_reps,axis=1)
            
        
            self.all_encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                cell=self.encoder_stacked_cell,
                inputs=self.context_inputs,
                sequence_length=self.enc_seq_len, time_major=False,
                dtype=tf.float32)


        with tf.variable_scope('decoder') as scope:

            self.decoder_initial_state = self.encoder_state

            self.decoder_cell = rnn_cell(FLAGS,self.dropout,scope=scope,decoder_cell=True)
        

            self.list_decoder_inputs = tf.unstack(
                self.dec_ip_emb, axis=1)    

            W_softmax = tf.get_variable("W_softmax",

                shape=[FLAGS.hidden_units, FLAGS.vocab_size],
                dtype=tf.float32)       
            
            
            b_softmax = tf.get_variable("b_softmax",
                shape=[FLAGS.vocab_size],
                dtype=tf.float32)       
            output_projection = (W_softmax, b_softmax)
            
            W_emb_mat=A.emb_mat    
            self.all_decoder_outputs, self.decoder_state = tf.cond(self.forward_only,
                    lambda: embedding_decoder(
                    decoder_inputs=self.list_decoder_inputs,
                    initial_state=self.decoder_initial_state,
                    cell=self.decoder_cell,
                    num_symbols=FLAGS.vocab_size,
                    output_projection=output_projection,
                    W_embedding=W_emb_mat,
                    feed_previous=True),
                    lambda: embedding_decoder(
                    decoder_inputs=self.list_decoder_inputs,
                    initial_state=self.decoder_initial_state,
                    cell=self.decoder_cell,
                    num_symbols=FLAGS.vocab_size,
                    output_projection=output_projection,
                    W_embedding=W_emb_mat,
                    feed_previous=False))                            

            # Logits
            self.decoder_outputs=decoder_logits(FLAGS,self.all_decoder_outputs,scope)
            self.decoder_outputs_tensor = tf.stack(self.decoder_outputs)
            self.decoder_outputs_logits = tf.transpose(self.decoder_outputs_tensor,[1,0,2])
            
            # Loss with masking
            self.targets_one_hot = tf.one_hot(self.decoder_targets,FLAGS.vocab_size)
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_outputs_logits,
                                                targets=self.decoder_targets,
                                                weights=tf.cast(self.target_weights,tf.float32))
        trainable_vars = tf.trainable_variables()
        #Clip Gradients
        if FLAGS.max_gradient_norm!=0:
            self.grads, _ = tf.clip_by_global_norm(
                    tf.gradients(self.loss, trainable_vars), FLAGS.max_gradient_norm)
        elif FLAGS.max_gradient_norm==0:
            self.grads = tf.gradients(self.loss, trainable_vars)
        
        #Train Op
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self.train_optimizer = optimizer.apply_gradients(
            zip(self.grads, trainable_vars))

        #Predictions
        self.y_pred = tf.argmax(self.decoder_outputs_logits,2)

        #Save Model
        self.saver = tf.train.Saver(tf.global_variables())

    def step(self,sess,FLAGS,batch,fo,dropout):
        """
        Run one pass of the Model using one batch
        
        args:
            sess: The active session,
            FLAGS: the configuration FLAGS,
            batch: One batch of data
            fo: Forward pass only is true or false
            dropout: The keep probability of the encoder-decoder hidden units
        """
        if not fo:
            #Training
            input_feed = {
                self.encoder_inputs: batch[0],
                self.decoder_inputs: batch[1],
                self.decoder_targets: batch[2],
                self.enc_seq_len: batch[3],
                self.dec_seq_len:batch[4],
                self.target_weights: batch[5],
                self.sent_seq_len:batch[6],
                self.forward_only: fo,
                self.dropout: dropout
                }
            output_feed = [self.y_pred, self.loss, self.train_optimizer]
            outputs = sess.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]
        elif fo:
           #Inference
           input_feed = {
                self.encoder_inputs: batch[0],
                self.decoder_inputs: batch[1],
                self.decoder_targets: batch[2],
                self.enc_seq_len: batch[3],
                self.dec_seq_len:batch[4],
                self.target_weights: batch[5],
                self.sent_seq_len:batch[6],
                self.forward_only: fo,
                self.dropout: dropout

                }
           output_feed = [self.y_pred,self.loss,self.y_pred]
           outputs = sess.run(output_feed, input_feed)
           return outputs[0],outputs[1],outputs[2]
