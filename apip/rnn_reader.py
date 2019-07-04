# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
from torch import autograd
import torch.nn as nn
from . import layers
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import gc

# Modification: 
#   - embedding APIP module into DrQA
#   - variatinal inference network optimization
#   - discrete VAE semi-supervised framework
#   - mixed cross-entropy + policy gradietn objective
#   - interpretation diversity objective 
#   - different control vectors to perturb question or document encodings depending on 
#   the value of the interpretation neuron before the final attention layer
#   - different final attention modules
# Origin: https://github.com/taolei87/sru/blob/master/DrQA/drqa/rnn_reader.py

def normalize_emb_(data):
    norms = data.norm(2,1) + 1e-8
    if norms.dim() == 1:
        norms = norms.unsqueeze(1)
    data.div_(norms.expand_as(data))

class RnnDocReader(nn.Module):
    """network for the Document Reader module of DrQA."""
    def __init__(self, opt, padding_idx=0, embedding=None, normalize_emb=False):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings
        if opt['pretrained_words']:
            assert embedding is not None
            self.embedding = nn.Embedding(embedding.size(0),
                                          embedding.size(1),
                                          padding_idx=padding_idx)
            if normalize_emb: normalize_emb_(embedding)
            self.embedding.weight.data = embedding

            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                for p in self.embedding.parameters():
                    p.requires_grad = False
            elif opt['tune_partial'] > 0:
                assert opt['tune_partial'] + 2 < embedding.size(0)
                fixed_embedding = embedding[opt['tune_partial'] + 2:]
                self.register_buffer('fixed_embedding', fixed_embedding)
                self.fixed_embedding = fixed_embedding
        else:  # random initialized
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)
        if opt['pos']:
            self.pos_embedding = nn.Embedding(opt['pos_size'], opt['pos_dim'])
            if normalize_emb: normalize_emb_(self.pos_embedding.weight.data)
        if opt['ner']:
            self.ner_embedding = nn.Embedding(opt['ner_size'], opt['ner_dim'])
            if normalize_emb: normalize_emb_(self.ner_embedding.weight.data)
        # Projection for attention weighted question
        if opt['use_qemb']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'], wn=opt['weight_norm'])

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['use_qemb']:
            doc_input_size += opt['embedding_dim']
        if opt['pos']:
            doc_input_size += opt['pos_dim']
        if opt['ner']:
            doc_input_size += opt['ner_dim']

        # RNN document encoder
        n_actions = opt['n_actions']
        # number of layers to perturb with weights associcated with interpretation neuron value
        # contains indices of start and end layer for adapted parameters 
        # and a name of a method for combining parameters: addition, multiplication, convolution   
        if 'sru' in opt['control_d']:
            func = opt['control_d'][4:]
        else: func = ""

        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,hidden_size=opt['hidden_size'],
            num_layers=opt['doc_layers'],dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],concat_layers=opt['concat_rnn_layers'],
            rnn_stype='sru',n_actions=n_actions,padding=opt['rnn_padding'],func=func 
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=opt['embedding_dim'],hidden_size=opt['hidden_size'],
            num_layers=opt['question_layers'],dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],concat_layers=opt['concat_rnn_layers'],
            rnn_stype=opt['rnn_type'],padding=opt['rnn_padding']
        )
        # if q in pi_q_rnn then create separate SRU for posterior q network
        if opt['vae'] and 'q' in opt['pi_q_rnn']:
            # if not nqd in pi_q_rnn then share SRU in q network for question and document sub-span encoding
            if not 'nqd' in opt['pi_q_rnn'] or 'qd2' in opt['pi_q_rnn']:
                # otherwise create new SRU for document sub-span encoding in q network
                self.q_doc_rnn = layers.StackedBRNN(
                    input_size=doc_input_size,hidden_size=opt['hidden_size'],
                    num_layers=opt['doc_layers'],dropout_rate=opt['dropout_rnn'],
                    dropout_output=opt['dropout_rnn_output'],concat_layers=opt['concat_rnn_layers'],
                    rnn_stype='sru',padding=opt['rnn_padding'])   

            if 'pi_d2' in self.opt['pi_q_rnn']:
                self.pi_doc_rnn = layers.StackedBRNN(
                    input_size=doc_input_size,hidden_size=opt['hidden_size'],
                    num_layers=opt['doc_layers'],dropout_rate=opt['dropout_rnn'],
                    dropout_output=opt['dropout_rnn_output'],concat_layers=opt['concat_rnn_layers'],
                    rnn_stype='sru',padding=opt['rnn_padding'])

            self.q_question_rnn = layers.StackedBRNN(
                input_size=opt['embedding_dim'],hidden_size=opt['hidden_size'],
                num_layers=opt['question_layers'],dropout_rate=opt['dropout_rnn'],
                dropout_output=opt['dropout_rnn_output'],concat_layers=opt['concat_rnn_layers'],
                rnn_stype='sru',padding=opt['rnn_padding']) 

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['question_layers']

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size, wn=opt['weight_norm'])

        d_factor = int(opt['gate'].split('_')[-1])


        # ---------------------------------  Final attention modules ------------------------------------

        # final attention layer for predicting start and end indices of a span
        if  opt['fin_att']=='mix_lin':
            # bilinear final attention 
            # mixes two sequences: doc and q encodings
            self.start_attn = layers.BilinearSeqAttnMix(
                doc_hidden_size,
                question_hidden_size, wn=opt['weight_norm'])
            self.end_attn = layers.BilinearSeqAttnMix(
                doc_hidden_size,
                question_hidden_size, wn=opt['weight_norm'])
        elif  opt['fin_att']=='linear':
            # bilinear final attention
            self.start_attn = layers.BilinearSeqAttn(
                doc_hidden_size,
                question_hidden_size, wn=opt['weight_norm'])
            self.end_attn = layers.BilinearSeqAttn(
                doc_hidden_size,
                question_hidden_size, wn=opt['weight_norm'])
        elif  'caps' in opt['fin_att']:
            # capsule network for final attention layer
            self.start_end_attn = layers.CapsNetFin(
                doc_hidden_size,
                question_hidden_size, opt['n_actions'])
        elif 'param' in opt['fin_att']:
            # parameterized bilinear attention layer for each value of the interpretation neuron
            f = opt['fin_att'].split('_')[-1]; func = 'h'
            if f in ['h', 'eh']:
                func = f                
            self.start_attn = layers.BilinearSeqAttnAction(
                doc_hidden_size,
                question_hidden_size, opt['n_actions'], wn=opt['weight_norm'])
            self.end_attn = layers.BilinearSeqAttnAction(
                doc_hidden_size,
                question_hidden_size, opt['n_actions'], wn=opt['weight_norm'])
        elif 'attn' in opt['fin_att']:
            # attention between a sequence of document encodings and a signle question encoding
            # where parameters are selected according to a value of the interpretation neuron
            self.start_attn = layers.SeqAttentionAction(
                doc_hidden_size,
                question_hidden_size, opt['n_actions'], wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
            self.end_attn = layers.SeqAttentionAction(
                doc_hidden_size,
                question_hidden_size, opt['n_actions'], wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
        elif 'pointer_a' in opt['fin_att']:
            # pointer network with SRU cell, where attention depends on the value of the interpretation neuron
            self.pointer_net = layers.PointerNetworkAction(
                doc_hidden_size,
                doc_hidden_size*d_factor, opt['n_actions'], opt=opt)
        elif 'pointer_s' in opt['fin_att']:
            # pointer network with SRU cell, where attention is independent from the value of the interpretation neuron
            self.pointer_net = layers.PointerNetwork(
                doc_hidden_size,
                doc_hidden_size*d_factor, wn=opt['weight_norm'], opt=opt)

        if opt['critic_loss']:
            # baseline for posterior network learning signal
            if not opt['vae'] or not('pi_d' in self.opt['pi_q_rnn'] and opt['vae']):
                crit_in = doc_hidden_size
            else:
                crit_in = 2*question_hidden_size
            self.critic_layer = layers.CriticLinear(crit_in, crit_in//2, wn=opt['weight_norm'], nl=int(opt['policy_critic'].split('_')[1]))

        
        if opt['vae']:

            # ---------------------------------- prior / posterior policies ----------------------------------------- 

            q_phi_hid = question_hidden_size+doc_hidden_size
            self.q_phi   = layers.PolicyLatent(q_phi_hid, doc_hidden_size, opt['n_actions'], wn=opt['weight_norm'], add=2, nl=int(opt['policy_critic'].split('_')[0]))
            if 'pi_d' in opt['pi_q_rnn']: 
                pi_theta_hid = question_hidden_size+doc_hidden_size; add=2
            else: 
                pi_theta_hid = question_hidden_size; add=1
            self.pi_theta = layers.PolicyLatent(pi_theta_hid, doc_hidden_size, opt['n_actions'], wn=opt['weight_norm'], add=add, nl=int(opt['policy_critic'].split('_')[0]))
            # policy s for selecting interpretation that will score high on SQuAD
            self.select_p = layers.PolicyLatent(pi_theta_hid, doc_hidden_size, opt['n_actions'], wn=opt['weight_norm'], add=add, nl=1+int(opt['policy_critic'].split('_')[0]))
            
            if 'mix' in opt['pi_inp']:
                # mix doc and q features of input to interpretation policy 
                self.q_mix = layers.MixingFeatures(doc_hidden_size, question_hidden_size, wn=opt['weight_norm'], latent=True)
                self.pi_mix = layers.MixingFeatures(doc_hidden_size, question_hidden_size, wn=opt['weight_norm'], latent=True)
           
            if opt['squad'] == 2:
                if 'fin' in opt['ae_archt']:
                    add = 2
                if 'policy' in opt['ae_archt']:
                    self.a_exist = layers.PolicyLatent(pi_theta_hid, doc_hidden_size, 1, wn=opt['weight_norm'], nl=int(opt['ae_archt'].split('_')[-1]), add=add)
                elif 'bili' in opt['ae_archt']:
                    self.a_exist = layers.BilinearSeqAexist(question_hidden_size, wn=opt['weight_norm'])
                elif 'dq_dist' in opt['ae_archt']:
                    self.a_exist = layers.MixingFeatures(doc_hidden_size, question_hidden_size, wn=opt['weight_norm'])

                if 'rnn' in opt['ae_archt']:
                    self.ae_rnn =  layers.StackedBRNN(
                                    input_size=opt['embedding_dim'],hidden_size=opt['hidden_size'],
                                    num_layers=opt['question_layers'],dropout_rate=opt['dropout_rnn'],
                                    dropout_output=opt['dropout_rnn_output'],concat_layers=opt['concat_rnn_layers'],
                                    rnn_stype='sru',padding=opt['rnn_padding']) 

                # use training signal on theta_1
                if 'doc' in opt['ae_archt']:
                    self.a_exist2 = layers.BilinearSeqAexist(doc_hidden_size, wn=opt['weight_norm'])

            # ------------------------------------- parameter adaptation modules ------------------------------------- 

            # control_vectors are used to perturb question or document encodings depending on 
            # the value of the interpretation neuron before the final attention layer
            # e.g. if sru in control_d then parameters of SRU for document encoding get adapted
            if opt['control_d'] == 'q_c':
                self.control_vector = layers.ControlVector(question_hidden_size, opt['gate'], opt['n_actions'], wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
                self.qc_attention = layers.SeqAttention(question_hidden_size, question_hidden_size*d_factor, wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
            elif opt['control_d'] == 'adap_q1':
                self.q_attention = layers.LinearSeqAttnAction_ad1(question_hidden_size, opt['n_actions'], wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
            elif opt['control_d'] == 'q_wa1':
                self.q_attention = layers.LinearSeqAttnAction(question_hidden_size, opt['n_actions'], wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
            elif opt['control_d'] == 'q_wa2':
                self.q_attention = layers.LinearSeqAttnAction2(question_hidden_size, opt['n_actions'], wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
            elif opt['control_d'] == 'd_ca':
                self.d_attention = layers.LinearSeqAttnAction(doc_hidden_size, opt['n_actions'], wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
                self.control_vector = layers.ControlVector(doc_hidden_size, opt['gate'], opt['n_actions'], wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
                self.dc_attention = layers.SeqAttention(doc_hidden_size, doc_hidden_size*d_factor, wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
            elif opt['control_d'] == 'd_qa':
                self.d_attention = layers.LinearSeqAttnAction(doc_hidden_size, opt['n_actions'], wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
                self.control_vector = layers.ControlVector(doc_hidden_size, opt['gate'], opt['n_actions'], wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
                self.qd_attention = layers.SeqAttention(question_hidden_size, doc_hidden_size*d_factor, wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
            elif opt['control_d'] == 'q_da':
                self.q_attention = layers.LinearSeqAttnAction2(question_hidden_size, opt['n_actions'], wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
                self.control_vector = layers.ControlVector(question_hidden_size, opt['gate'], opt['n_actions'], wn=opt['weight_norm'], drop_r=opt['dropout_rate'])
            elif 'd_mrnn' in opt['control_d'] :
                # match-BiSRU for perturbing document encodings, where attention depends on the value of interpretation neuron
                attn = opt['control_d'].split('_')[-1]
                self.d_mrnn = layers.MatchBRNN(input_size=2*doc_hidden_size,
                                    hidden_size=opt['hidden_size'],num_layers=1, n_actions = opt['n_actions'],
                                    attn=attn,dropout_rate=opt['dropout_rnn'],dropout_output=opt['dropout_rnn_output'],
                                    concat_layers=opt['concat_rnn_layers'],rnn_type=self.RNN_TYPES[opt['rnn_type']],
                                    padding=opt['rnn_padding'])
        else:

            if opt['squad'] == 2:
                if 'rnn' in opt['ae_archt']:
                    self.ae_rnn =  layers.StackedBRNN(input_size=opt['embedding_dim'],hidden_size=opt['hidden_size'],
                                    num_layers=opt['question_layers'],dropout_rate=opt['dropout_rnn'],
                                    dropout_output=opt['dropout_rnn_output'],concat_layers=opt['concat_rnn_layers'],
                                    rnn_stype='sru',padding=opt['rnn_padding'])
                if 'policy' in opt['ae_archt']:
                    self.a_exist = layers.PolicyLatent(2*doc_hidden_size, doc_hidden_size, 1, wn=opt['weight_norm'], nl=int(opt['ae_archt'].split('_')[-1]))
                elif 'bili' in opt['ae_archt']:
                    self.a_exist = layers.BilinearSeqAexist(question_hidden_size, wn=opt['weight_norm'])
                elif 'dq_dist' in opt['ae_archt']:
                    self.a_exist = layers.MixingFeatures(doc_hidden_size, question_hidden_size, wn=opt['weight_norm'])

                # use training signal on theta_1
                if 'doc' in opt['ae_archt']:
                    self.a_exist2 = layers.BilinearSeqAexist(doc_hidden_size, wn=opt['weight_norm'])

    
    def forward(self, inpt):
        """ Inputs:
            x1 = document word indices             [batch * len_d]
            x1_f = document word features indices  [batch * len_d * nfeat]
            x1_pos = document POS tags             [batch * len_d]
            x1_ner = document entity tags          [batch * len_d]
            x1_mask = document padding mask        [batch * len_d]
            x2 = question word indices             [batch * len_q]
            x2_mask = question padding mask        [batch * len_q]
            labels = labels for question interpratations       [batch]
            l_mask = mask for labeled question interpretations [batch]
        """
        x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, s_ids, e_ids, scope, labels, l_mask, latent_a = inpt
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        if self.opt['dropout_emb'] > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.opt['dropout_emb'],
                                               training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.opt['dropout_emb'],
                                           training=self.training)

        drnn_input_list = [x1_emb, x1_f]
        # Add attention-weighted question representation
        if self.opt['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input_list.append(x2_weighted_emb)
        if self.opt['pos']:
            x1_pos_emb = self.pos_embedding(x1_pos)
            if self.opt['dropout_emb'] > 0:
                x1_pos_emb = nn.functional.dropout(x1_pos_emb, p=self.opt['dropout_emb'],
                                               training=self.training)
            drnn_input_list.append(x1_pos_emb)
        if self.opt['ner']:
            x1_ner_emb = self.ner_embedding(x1_ner)
            if self.opt['dropout_emb'] > 0:
                x1_ner_emb = nn.functional.dropout(x1_ner_emb, p=self.opt['dropout_emb'],
                                               training=self.training)
            drnn_input_list.append(x1_ner_emb)
        drnn_input = torch.cat(drnn_input_list, 2)

        if (self.opt['vae']) and not self.opt['all_emb_tune']:
            drnn_input = drnn_input.detach()
            x2_emb = x2_emb.detach()

        # Encode document with RNN
        if not ('sru' in self.opt['control_d']):
            doc_hiddens = self.doc_rnn(drnn_input, x1_mask)
            doc_hiddens_init = doc_hiddens.clone()
        else: 
            doc_hiddens = 0

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        # weighted average of question encodings 
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)
        question_hidden_init = question_hidden.clone()

        # ------------------------------------ VAE -------------------------------------------------

        start_scores, end_scores, crit_f1 = [None]*3
        s_sma, e_sma, s_mxa, e_mxa, s_logp, e_logp = [None]*6
        kl_loss, r_kl, q_logp_t, p_logp_t, actions, control, ent_loss = [None]*7
        pi_logits, vf, vf_t, ae_prob2, dq_dist, s_logits, ae_prob = [None]*7

        if self.opt['vae']:
            inpt = [scope, s_ids, e_ids, doc_hiddens, question_hidden, question_hiddens, drnn_input.detach(), x1_emb.detach(), x2_emb.detach(), x1_mask, x2_mask, labels, l_mask, latent_a]
            outpt = self.vae_framework(inpt)
            vae_vars, doc_hiddens, question_hidden, question_hiddens = outpt
            kl_loss, r_kl, q_logp_t, p_logp_t, s_logits, ae_prob, dq_dist, actions, ent_loss, computed_a, doc_h = vae_vars           
        else:
            computed_a = actions = Variable(torch.zeros(1)).cuda()

            if self.opt['squad'] == 2:
                doc_h_temp = torch.cat((doc_hiddens[:,-1,:doc_hiddens.size(2)//2], doc_hiddens[:,0,doc_hiddens.size(2)//2:]), 1)
                input_state = torch.cat((doc_h_temp, question_hidden), 1)

                if 'rnn' in self.opt['ae_archt']:
                    d_hids = self.ae_rnn(x1_emb)
                    q_hids = self.ae_rnn(x2_emb)
                    doc_h_ae = torch.cat((d_hids[:,-1,:d_hids.size(2)//2], d_hids[:,0,d_hids.size(2)//2:]), 1)
                    que_h_ae = torch.cat((q_hids[:,-1,:q_hids.size(2)//2], q_hids[:,0,q_hids.size(2)//2:]), 1)
                    input_state = torch.cat((doc_h_ae, que_h_ae), 1)

                if 'policy' in self.opt['ae_archt']:
                    if 'det' in self.opt['ae_archt']:
                        input_state = input_state.detach()
                    ae_logits, ae_prob = self.a_exist(input_state)
                elif 'bili' in self.opt['ae_archt']:
                    if 'det' in self.opt['ae_archt']:
                        ae_logits, ae_prob = self.a_exist(doc_hiddens.detach(), question_hidden.detach())
                    else:
                        ae_logits, ae_prob = self.a_exist(doc_hiddens, question_hidden)
                elif 'dq_dist' in self.opt['ae_archt']:
                    if 'det' in self.opt['ae_archt']:
                        m_d, m_q, ae_prob = self.a_exist(doc_hiddens.detach(), x1_mask, question_hiddens.detach(), x2_mask)
                    else:
                        m_d, m_q, ae_prob = self.a_exist(doc_hiddens, x1_mask, question_hiddens, x2_mask)
                    dq_dist = torch.norm(m_d-m_q, p=2, dim=1)

        # ------------------ Parameter adaptation depending on the value of interpretation neuron (actions) -----------------

        if 'sru' in self.opt['control_d']:
            # if adapting parameters in the document SRU network in p_theta1
            doc_hiddens = self.doc_rnn(drnn_input, x1_mask, actions=actions)
            doc_hiddens_init = doc_hiddens.clone()
        elif self.opt['control_d'] == 'q_c':
            # get control vector by adapting question encoding and merge the control vector with initial question encodings
            control = self.control_vector(question_hidden, actions)                
            qc_att_weights = self.qc_attention(question_hiddens, control, x2_mask)
            question_hidden = layers.weighted_avg(question_hiddens, qc_att_weights) 
        elif 'q_wa' in self.opt['control_d'] or 'adap_q1' in self.opt['control_d']:
            # adapt parameters by action value for obtaining weighted question encoding sum
            q_att_weights = self.q_attention(question_hiddens, x2_mask, actions)
            question_hidden = layers.weighted_avg(question_hiddens, q_att_weights)
            control = question_hidden.clone()
        elif self.opt['control_d'] == 'd_ca':
            # get single document encoding by weighted sum with action associated weights
            # get control_vector by adapting document encoding
            # get sequence of adapted document encodings by weighting with adapted attention weights 
            d_att_weights = self.d_attention(doc_hiddens, x1_mask, actions)
            doc_hidden = layers.weighted_avg(doc_hiddens, d_att_weights)
            control = self.control_vector(doc_hidden, actions)                
            dc_att_weights = self.dc_attention(doc_hiddens, control, x1_mask)
            doc_hiddens = doc_hiddens* dc_att_weights.unsqueeze(2).expand_as(doc_hiddens)
        elif self.opt['control_d'] == 'd_qa':
            # get single document encoding by weighted sum with action associated weights
            # get control_vector by adapting document encoding 
            # update control vector by merging it with question encodings
            d_att_weights = self.d_attention(doc_hiddens, x1_mask, actions)
            doc_hidden = layers.weighted_avg(doc_hiddens, d_att_weights)
            control = self.control_vector(doc_hidden, actions)                
            qd_att_weights = self.qd_attention(question_hiddens, control, x2_mask)
            question_hidden = layers.weighted_avg(question_hiddens, qd_att_weights)
            control = question_hidden.clone()
        elif self.opt['control_d'] == 'q_da':
            # obtain new attention weights
            # get control_vector by adapting question encoding based on actions
            q_att_weights = self.q_attention(question_hiddens, x2_mask, actions)
            question_hidden = layers.weighted_avg(question_hiddens, q_att_weights)
            control = self.control_vector(question_hidden, actions)                
        elif 'd_mrnn' in self.opt['control_d'] :
            # match-BiSRU for perturbing document encodings, where attention depends on the action
            doc_hiddens = self.d_mrnn(doc_hiddens, x1_mask, actions) 
        else:
            control = question_hidden


        # --------------------------------- Final attention layer ------------------------------------

        # Predict start and end positions
        if 'pointer' in self.opt['fin_att']:
            # pointer network
            start_scores, end_scores = self.pointer_net(doc_hiddens, x1_mask, control, actions)
        elif 'attn' in self.opt['fin_att']:
            # attention between a sequence of document encodings and a signle question encoding
            if self.training:
                nonlin = lambda x: F.log_softmax(x, dim=-1)
            else:
                nonlin = lambda x: F.softmax(x, dim=-1)
            start_scores = nonlin(self.start_attn(doc_hiddens, question_hidden, x1_mask, actions))
            end_scores = nonlin(self.end_attn(doc_hiddens, question_hidden, x1_mask, actions))
        elif 'caps' in self.opt['fin_att']:
            # final layer as a capsule network
            start_scores, end_scores = self.start_end_attn(doc_hiddens, question_hidden, x1_mask, actions)
        elif 'mix_lin' in self.opt['fin_att']:
            start_scores = self.start_attn(doc_hiddens, question_hiddens, question_hidden, x1_mask, x2_mask)
            end_scores = self.end_attn(doc_hiddens, question_hiddens, question_hidden, x1_mask, x2_mask) 
        else:
            start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask, actions)
            end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask, actions)

        if self.opt['squad'] == 2:
            if 'fin' in self.opt['ae_archt']:
                # if ae classifier is at final layer
                doc_h_temp = torch.cat((doc_hiddens[:,-1,:doc_hiddens.size(2)//2], doc_hiddens[:,0,doc_hiddens.size(2)//2:]), 1)
                input_state = torch.cat((doc_h_temp, question_hidden), 1)

                if 'rnn' in self.opt['ae_archt']:
                    d_hids = self.ae_rnn(x1_emb)
                    q_hids = self.ae_rnn(x2_emb)
                    doc_h_ae = torch.cat((d_hids[:,-1,:d_hids.size(2)//2], d_hids[:,0,d_hids.size(2)//2:]), 1)
                    que_h_ae = torch.cat((q_hids[:,-1,:q_hids.size(2)//2], q_hids[:,0,q_hids.size(2)//2:]), 1)
                    input_state = torch.cat((doc_h_ae, que_h_ae), 1)

                if 'policy' in self.opt['ae_archt']:
                    if 'det' in self.opt['ae_archt']:
                        input_state = input_state.detach()
                    ae_logits, ae_prob = self.a_exist(input_state)
                elif 'bili' in self.opt['ae_archt']:
                    if 'det' in self.opt['ae_archt']:
                        ae_logits, ae_prob = self.a_exist(doc_hiddens.detach(), question_hidden.detach())
                    else:
                        ae_logits, ae_prob = self.a_exist(doc_hiddens, question_hidden)
                elif 'dq_dist' in self.opt['ae_archt']:
                    if 'det' in self.opt['ae_archt']:
                        m_d, m_q, ae_prob = self.a_exist(doc_hiddens.detach(), x1_mask, question_hiddens.detach(), x2_mask)
                    else:
                        m_d, m_q, ae_prob = self.a_exist(doc_hiddens, x1_mask, question_hiddens, x2_mask)
                    dq_dist = torch.norm(m_d-m_q, p=2, dim=1)

            elif 'doc' in self.opt['ae_archt']:
                if 'dq_dist' in self.opt['ae_archt']:
                    ae_prob2 = self.a_exist(doc_hiddens, x1_mask, question_hiddens, x2_mask)[1]
                else:
                    ae_prob2 = self.a_exist2(doc_hiddens, question_hidden)[1]

        if self.opt['critic_loss']:
            if not self.opt['vae']:
                s_probs = start_scores.exp() if self.training else start_scores
                e_probs = end_scores.exp() if self.training else end_scores
                doc_hidden = layers.weighted_avg(doc_hiddens, s_probs) + layers.weighted_avg(doc_hiddens, e_probs)
                crit_f1 = self.critic_layer(doc_hidden)
                crit_f1 = F.sigmoid(crit_f1)
            else:
                if 'pi_d' in self.opt['pi_q_rnn']:
                    critic_h = torch.cat((doc_h, question_hidden), 1)
                else:
                    critic_h = question_hidden
                crit_f1 = self.critic_layer(critic_h.detach())


        if self.opt['self_critic']:
            s_probs = start_scores.exp() if self.training else start_scores
            s_sma, s_logp = layers.make_action(s_probs)
            _, s_mxa = s_probs.max(1)
            
            e_probs = end_scores.exp() if self.training else end_scores
            e_sma, e_logp = layers.make_action(e_probs)
            _, e_mxa = e_probs.max(1)

        span_vars = start_scores, end_scores, crit_f1, drnn_input, doc_hiddens_init.data, question_hidden.data
        scrit_vars = s_sma, e_sma, s_mxa, e_mxa, s_logp, e_logp
        vae_vars = kl_loss, r_kl, q_logp_t, p_logp_t, s_logits, [ae_prob, ae_prob2, dq_dist], ent_loss, actions.data, computed_a.data
        return span_vars, scrit_vars, vae_vars



    def get_qphi_ans(self, s_ids, e_ids, drnn_input):
        # get embeddings of a correct document sub-span (answer) for posterior q network
        e_ids, s_ids = e_ids.data.cpu().numpy(), s_ids.data.cpu().numpy()
        a_len = (e_ids - s_ids).max(0) + 1
        ans_hiddens = torch.FloatTensor(drnn_input.size(0), int(a_len), drnn_input.size(2)).fill_(0).cuda() 
        a_mask = []
        for b in range(drnn_input.size()[0]):
            ans_hiddens[b, :e_ids[b]+1-s_ids[b]] = drnn_input[b, s_ids[b]:e_ids[b]+1].data
            a_mask.append([1]*(e_ids[b]+1-s_ids[b]) + [0]*(a_len - (e_ids[b]+1-s_ids[b])))
        ans_hiddens = ans_hiddens.contiguous()
        if self.opt['cuda']:
            a_mask = Variable(torch.ByteTensor(a_mask).cuda())
            ans_hiddens = Variable(ans_hiddens.cuda())
        else:
            a_mask = Variable(torch.ByteTensor(a_mask))
            ans_hiddens = Variable(ans_hiddens) 
        return ans_hiddens, a_mask

            

    def vae_framework(self, inpt):

        scope, s_ids, e_ids, doc_hiddens, question_hidden, question_hiddens, drnn_input, \
                                                    x1_emb, x2_emb, x1_mask, x2_mask, labels, l_mask, latent_a = inpt
        kl_loss, r_kl, q_logp_t, p_logp_t, s_logits, ae_prob, dq_dist = [None]*7; ent_loss=0

        # if current training framework is VAE
        if scope == 'pi_q':            
            ans_hiddens, a_mask = self.get_qphi_ans(s_ids, e_ids, drnn_input)
            if 'q' in self.opt['pi_q_rnn']:
                if 'qd2' in self.opt['pi_q_rnn']:
                    # if exists separate SRU for document sub-span in q network
                    ans_hiddens_rnn = self.q_doc_rnn(ans_hiddens, a_mask)
                    q_doc_hid = self.q_doc_rnn(drnn_input, a_mask)
                elif not 'nqd' in self.opt['pi_q_rnn']:
                    # if exists separate SRU for document sub-span in q network
                    ans_hiddens_rnn = self.q_doc_rnn(ans_hiddens, a_mask)
                else:
                    # otherwise re-use SRU from question encoding of q network for document sub-span
                    ans_hiddens_rnn = self.q_question_rnn(ans_hiddens[:,:,:self.embedding.weight.size(1)], a_mask)
                # use first and last encoding of bi-directional sru as answer (question) final encoding vector
                ans_hiddens = torch.cat((ans_hiddens_rnn[:,-1,:ans_hiddens_rnn.size(2)//2], ans_hiddens_rnn[:,0,ans_hiddens_rnn.size(2)//2:]), 1)
                # encode question with SRU in q network
                question_hiddens_q = self.q_question_rnn(x2_emb, x2_mask)
                question_hidden_q = torch.cat((question_hiddens_q[:,-1,:question_hiddens_q.size(2)//2], question_hiddens_q[:,0,question_hiddens_q.size(2)//2:]), 1)
            else:
                # when q network re-uses SRU network for encoding document sub-span and reuses question encoding from p_theta
                ans_hiddens_rnn = self.doc_rnn(ans_hiddens, x2_mask, use_a=False)
                question_hidden_q = question_hidden.detach()
                ans_hiddens = torch.cat((ans_hiddens_rnn[:,-1,:ans_hiddens_rnn.size(2)//2], ans_hiddens_rnn[:,0,ans_hiddens_rnn.size(2)//2:]), 1).detach()

            if 'qd2' in self.opt['pi_q_rnn']:
                doc_hiddens_q = torch.cat((q_doc_hid[:,-1,:q_doc_hid.size(2)//2], q_doc_hid[:,0,q_doc_hid.size(2)//2:]), 1)
                ans_hiddens = torch.cat((ans_hiddens, doc_hiddens_q), 1)
                print(ans_hiddens.size())

            # -------------------  q_phi posterior distribution  -----------------------
            if not 'mix' in self.opt['pi_inp']:
                q_logits, q_prob = self.q_phi(torch.cat((ans_hiddens, question_hidden_q), 1))
            else:
                q_ans, q_ques = self.q_mix(ans_hiddens_rnn, a_mask, question_hiddens, x2_mask)[:2]
                q_logits, q_prob = self.q_phi(torch.cat((q_ans, q_ques), 1))
            q_ta, q_logp_t = layers.make_action(q_prob) # sampling

        # -------------------  pi_theta prior distribution  -----------------------

        if 'pi_d' in self.opt['pi_q_rnn']:
            # if pi is conditioned on document encoding
            if 'pi_d2' in self.opt['pi_q_rnn']:
                d_hids = self.pi_doc_rnn(drnn_input)
            else:
                d_hids = self.question_rnn(x1_emb)
            doc_h = torch.cat((d_hids[:,-1,:d_hids.size(2)//2], d_hids[:,0,d_hids.size(2)//2:]), 1)
            
            if not 'mix' in self.opt['pi_inp']:
                pi_input_state = torch.cat((doc_h, question_hidden), 1)
            else:
                pi_doc, pi_ques = self.pi_mix(d_hids, x1_mask, question_hiddens, x2_mask)[:2]
                pi_input_state = torch.cat((pi_doc, pi_ques), 1)
        else:
            doc_h = 0
            pi_input_state = question_hidden

        
        if not 'pi' in self.opt['pi_q_rnn'] or (not self.opt['rl_tuning'] and scope=='rl'):
            # if pi network does not include SRU from theta_1 for question encoding
            pi_input_state = pi_input_state.detach()

        if scope in ['pi_q', 'rl']:
            p_logits, p_prob = self.pi_theta(pi_input_state)
            p_ta, p_logp_t = layers.make_action(p_prob) # sampling
        elif scope == 'select_i':
            s_logits, s_prob = self.select_p(pi_input_state.detach())
            #s_ta, s_logp_t = layers.make_action(s_prob)
            _, s_ta = torch.max(s_logits, 1)
        

        if self.opt['squad'] == 2 and not 'fin' in self.opt['ae_archt']:
            que_h_ae = question_hidden; q_hids = question_hiddens
            if 'rnn' in self.opt['ae_archt']:
                d_hids = self.ae_rnn(x1_emb)
                q_hids = self.ae_rnn(x2_emb)
                doc_h_ae = torch.cat((d_hids[:,-1,:d_hids.size(2)//2], d_hids[:,0,d_hids.size(2)//2:]), 1)
                que_h_ae = torch.cat((q_hids[:,-1,:q_hids.size(2)//2], q_hids[:,0,q_hids.size(2)//2:]), 1)
                pi_input_state = torch.cat((doc_h_ae, que_h_ae), 1)

            if 'policy' in self.opt['ae_archt']:
                if 'det' in self.opt['ae_archt']:
                    pi_input_state = pi_input_state.detach()
                ae_logits, ae_prob = self.a_exist(pi_input_state)
            elif 'bili' in self.opt['ae_archt']:
                if 'det' in self.opt['ae_archt']:
                    ae_logits, ae_prob = self.a_exist(d_hids.detach(), que_h_ae.detach())
                else:
                    ae_logits, ae_prob = self.a_exist(d_hids, que_h_ae)
            elif 'dq_dist' in self.opt['ae_archt']:
                if 'det' in self.opt['ae_archt']:
                    m_d, m_q, ae_prob = self.a_exist(d_hids.detach(), x1_mask, q_hids.detach(), x2_mask)
                else:
                    m_d, m_q, ae_prob = self.a_exist(d_hids, x1_mask, q_hids, x2_mask)
                dq_dist = torch.norm(m_d-m_q, p=2, dim=1)


        assert scope in ['pi_q', 'rl', 'select_i'], "vae scope values"

        if scope == 'pi_q':
            # if VAE framework - use actions from posterior q to compute stochastic gradients
            actions = q_ta
        elif scope == 'select_i':
            # if select_i framework - use actions from select_p to compute stochastic gradients
            actions = s_ta
        elif scope == 'rl':
            # during policy gradient tuning the actions are sampled from the learned policy pi
            if self.training:
                actions = Variable(latent_a)
                taken_actions = layers.one_hot(actions, self.opt['n_actions'])
                p_logp_t = (F.log_softmax(p_logits, dim=1)*taken_actions).sum(1)
            else:
                actions = p_prob.max(1)[1]


        if not scope == 'rl' and self.opt['semisup'] and self.training:
            # if semi-supervised VAE framework - use available interpretations
            actions = (1-l_mask)*actions + labels*l_mask
            computed_a = actions
        elif self.opt['interpret']:
            # in the induced interpretation mode use induced actions
            computed_a = actions
            actions = latent_a
        else:
            # otherwise use actions from prior
            computed_a = actions


        if not 'pi' in self.opt['pi_q_rnn'] or (not self.opt['rl_tuning'] and scope=='rl'):
            # if policy pi does not update SRU parameters for question encoding in p_theta
            if 'sru' not in self.opt['control_d']:
                doc_hiddens = doc_hiddens.detach()
            question_hiddens = question_hiddens.detach()
            question_hidden = question_hidden.detach()


        if scope == 'pi_q':
            kl_loss = (q_prob.detach()*(F.log_softmax(p_logits, dim=1)          - F.log_softmax(q_logits.detach(), dim=1))).sum(1)
            # partial learning signal for q_phi network
            r_kl    = (q_prob.detach()*(F.log_softmax(q_logits.detach(), dim=1) - F.log_softmax(p_logits.detach(), dim=1))).sum(1)
            taken_actions = layers.one_hot(actions, self.opt['n_actions'])
            # prior -1*cross-entropy w.r.t actions from posterior
            p_logp_t = (F.log_softmax(p_logits, dim=1)*taken_actions).sum(1)
            # posterior -1*cross-entropy w.r.t actions from posterior
            q_logp_t = (F.log_softmax(q_logits, dim=1)*taken_actions).sum(1)
            if self.opt['entropy_loss']:
                ent_loss = layers.cat_entropy(q_logits).sum()
            else:
                ent_loss = 0

        vae_vars = [kl_loss, r_kl, q_logp_t, p_logp_t, s_logits, ae_prob, dq_dist, actions, ent_loss, computed_a, doc_h]
        return [vae_vars, doc_hiddens, question_hidden, question_hiddens]

