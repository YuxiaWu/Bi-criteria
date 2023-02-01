from lib2to3.pgen2 import token
from operator import mod
import torch
import torch.nn.functional as F
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch.nn.parameter import Parameter
from .utils import PairEnum
import copy

from pytorch_pretrained_bert.tokenization import BertTokenizer

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

def getContextEmbedding(current_value_feats, hiddens, valid_length, indices):
    """
        get the Context embedding by attention

        current_value_feats: (1, feat_dim);
        hiddens: (seq_len, feat_dim);
        valid_length: length of valid part of the sequence;
        indices: indices of slot in the sequence; (slot_len)
    """
    current_value_feats = current_value_feats.detach()
    index = torch.arange(hiddens.shape[0]).cuda()#.to(self.device)
    #valid_length = valid_length.cpu()
    mask = (index<indices[0])| ((index>indices[-1]) & (index< valid_length))
    mask = mask.cuda()
    masked_weights = hiddens.matmul(current_value_feats.squeeze()) * mask
    unnorm_weights = (masked_weights - masked_weights.max()).exp()
    norm_weights = unnorm_weights / unnorm_weights.sum()
    
    attention_weights = torch.mul(hiddens, norm_weights.unsqueeze(-1))
    context_embedding = attention_weights.sum(dim=0)

    return context_embedding

# [only value, mask valus mask]
def get_slot_emb(args, bert_model, input_ids, input_mask, segment_ids, bin_label_ids, lstm_for_value=None):
    """
    1. get_hidden feature for each token
    2. get the features of the candidate values based on binary_labels
    3. get the context feature
    4. concat(value,context) and project to the slot space
    5. predict the prob 
    """
    lengths = torch.sum(input_mask, 1) # length of every example (bsz,)
    # 1. hidden feature for each token
    
    binary_labels = bin_label_ids # if binary_golds is not None else binary_preditions
    feats_concat = []
    bsz = input_ids.size()[0]
    #print('bsz: ', bsz)
    for i in range(bsz):
        #slot_list_based_domain = domain2slot[domain_name]  # a list of slot names  #['genre', 'music_item', 'service', 'year', 'playlist', 'album', 'sort', 'track', 'artist']
        valid_length = lengths[i]
        # we can also add domain embeddings after transformer encoder
        #hidden_i = feats_token[i]    # (seq_len, feat_dim) #20,400
        bin_label = binary_labels[i]
        # get indices of B and I 
        indices = torch.nonzero(bin_label) 
        
        # the input for value only
        input_ids_value = input_ids[i][indices].transpose(0,1)
        segment_ids_value = segment_ids[i][indices].transpose(0,1)
        input_mask_value = input_mask[i][indices].transpose(0,1)
        hidden_values, _ = bert_model(
            input_ids_value, 
            token_type_ids=segment_ids_value, 
            attention_mask=input_mask_value,
            output_all_encoded_layers=False)

        if args.value_enc_type == "lstm":
            value_feats, (_, _) = lstm_for_value(hidden_values)   # (1, subseq_len, hidden_dim)
            value_feats = torch.mean(value_feats, dim=1)  # (1, hidden_dim)
        else:
            value_feats = torch.mean(hidden_values, dim=1) # (1, hidden_dim)

        # replace the indice of value with MASK
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True) 
        
        input_ids_copy = copy.deepcopy(input_ids)
        input_id = input_ids_copy[i].unsqueeze(0) #[1,256]
        
        mask_id = tokenizer.convert_tokens_to_ids(['mask'])

        for ind in indices:
            input_id[0,ind] = torch.LongTensor(mask_id).cuda()
        # get the CLS feat 
        #print(input_id.size())
        feats_token, _ = bert_model(
            input_id, 
            token_type_ids=segment_ids[i].unsqueeze(0), 
            attention_mask=input_mask[i].unsqueeze(0),
            output_all_encoded_layers=False)
 
        hidden_values_mask = feats_token[0][indices].transpose(0,1) #[1,1,768] #[4,1,768]
        if args.value_enc_type == "lstm":
            context_feats, (_, _) = lstm_for_value(hidden_values_mask)   # (1, subseq_len, hidden_dim)
            context_feats = torch.mean(context_feats, dim=1)  # (1, hidden_dim)
        else:
            context_feats = torch.mean(hidden_values_mask, dim=1) # (1, hidden_dim)

        if args.context:
            value_context = torch.cat((value_feats, context_feats),1)
        else:
            value_context = value_feats
        feats_concat.append(value_context)

    feats_concat_o = torch.vstack(feats_concat)
    return feats_concat_o 

def get_slot_emb_only_value(args, bert_model, input_ids, input_mask, segment_ids, bin_label_ids, lstm_for_value=None):
    """
    1. get_hidden feature for each token
    2. get the features of the candidate values based on binary_labels
    3. get the context feature
    4. concat(value,context) and project to the slot space
    5. predict the prob 
    """
    lengths = torch.sum(input_mask, 1) # length of every example (bsz,)
    # 1. hidden feature for each token
    feats_token, _ = bert_model(
        input_ids, 
        token_type_ids=segment_ids, 
        attention_mask=input_mask,
        output_all_encoded_layers=False)

    binary_labels = bin_label_ids # if binary_golds is not None else binary_preditions
    feats_concat = []
    bsz = input_ids.size()[0]
    #print('bsz: ', bsz)
    for i in range(bsz):
        #slot_list_based_domain = domain2slot[domain_name]  # a list of slot names  #['genre', 'music_item', 'service', 'year', 'playlist', 'album', 'sort', 'track', 'artist']
        valid_length = lengths[i]
        # we can also add domain embeddings after transformer encoder
        hidden_i = feats_token[i]    # (seq_len, feat_dim) #20,400
        bin_label = binary_labels[i]
        #tensor([0, 0, 1, 2, 0, 0, 0])
        # get indices of B and I 
        indices = torch.nonzero(bin_label) 
        if len(indices)==0:
            print("debug")  
            the_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
            uttr_list = [the_tokenizer.ids_to_tokens[k] for k in input_ids.cpu().numpy()[i]]
            print(uttr_list)
            print("debug")  

      
        #hidden_values = hidden_i[indices].transpose(0,1) # [1,num_value_token,  feat_dim]
        
        # the input for value only
        input_ids_value = input_ids[i][indices].transpose(0,1)
        segment_ids_value = segment_ids[i][indices].transpose(0,1)
        input_mask_value = input_mask[i][indices].transpose(0,1)
        hidden_values, _ = bert_model(
            input_ids_value, 
            token_type_ids=segment_ids_value, 
            attention_mask=input_mask_value,
            output_all_encoded_layers=False)
        #print('hidden_values.size(): ', hidden_values.size())

        if args.value_enc_type == "lstm":
            value_feats, (_, _) = lstm_for_value(hidden_values)   # (1, subseq_len, hidden_dim)
            value_feats = torch.mean(value_feats, dim=1)  # (1, hidden_dim)
        else:
            value_feats = torch.mean(hidden_values, dim=1) # (1, hidden_dim)
        
        if args.context:
            context_feats = getContextEmbedding(value_feats, hidden_i, valid_length, indices).unsqueeze(0) # (1, feat_dim)
            value_context = torch.cat((value_feats, context_feats),1)
        else:
            value_context = value_feats

        feats_concat.append(value_context)
        #print('151feats_concat.len: ', len(feats_concat))         

    feats_concat_o = torch.vstack(feats_concat)
    #print('154feats_concat.size(): ', feats_concat_o.size())         

    return feats_concat_o  


def get_slot_emb_in_uttr(args, bert_model, input_ids, input_mask, segment_ids, bin_label_ids, lstm_for_value=None):
#def get_slot_emb(args, bert_model, input_ids, input_mask, segment_ids, bin_label_ids, lstm_for_value=None):
    """
    1. get_hidden feature for each token
    2. get the features of the candidate values based on binary_labels
    3. get the context feature
    4. concat(value,context) and project to the slot space
    5. predict the prob 
    """
    lengths = torch.sum(input_mask,1) # length of every example (bsz,)
    # 1. hidden feature for each token
    feats_token, _ = bert_model(
        input_ids, 
        token_type_ids=segment_ids, 
        attention_mask=input_mask,
        output_all_encoded_layers=False)

    binary_labels = bin_label_ids # if binary_golds is not None else binary_preditions
    feats_concat = []
    bsz = input_ids.size()[0]
    for i in range(bsz):
        #slot_list_based_domain = domain2slot[domain_name]  # a list of slot names  #['genre', 'music_item', 'service', 'year', 'playlist', 'album', 'sort', 'track', 'artist']
        valid_length = lengths[i]
        # we can also add domain embeddings after transformer encoder
        hidden_i = feats_token[i]    # (seq_len, feat_dim) #20,400
        bin_label = binary_labels[i]
        #tensor([0, 0, 1, 2, 0, 0, 0])
        # get indices of B and I 
        indices = torch.nonzero(bin_label) 
        #if len(indices）        
        hidden_values = hidden_i[indices].transpose(0,1) # [1,num_value_token,  feat_dim]
        
        if args.value_enc_type == "lstm":
            value_feats, (_, _) = lstm_for_value(hidden_values)   # (1, subseq_len, hidden_dim)
            #value_feats = torch.sum(value_feats, dim=1)  # (1, hidden_dim)
            value_feats = torch.mean(value_feats, dim=1)  # (1, hidden_dim)
        else:
            #value_feats = torch.sum(hidden_values, dim=1) # (1, hidden_dim)
            value_feats = torch.mean(hidden_values, dim=1) # (1, hidden_dim)
        
        #value_feats = torch.mean(hidden_values, dim=1) # (1, feat_dim)
        
        if args.context:
            context_feats = getContextEmbedding(value_feats, hidden_i, valid_length, indices).unsqueeze(0) # (1, feat_dim)
            value_context = torch.cat((value_feats, context_feats),1)
        else:
            value_context = value_feats

        feats_concat.append(value_context)

    feats_concat = torch.vstack(feats_concat)         

    return feats_concat        


class BERT(BertPreTrainedModel):
    
    def __init__(self, config, args):

        super(BERT, self).__init__(config)

        # print(args)

        self.args = args
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.lstm_for_value = nn.LSTM(
            input_size = config.hidden_size, 
            hidden_size = config.hidden_size//2, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True)
        if args.context:
            self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)      
        self.apply(self.init_bert_weights)
        
    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None,
                feature_ext = False, mode = None, loss_fct = None, bin_label_ids = None):
        '''
        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = False)
        pooled_output = self.dense(encoded_layer_12.mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        '''
        pooled_output = get_slot_emb(self.args, self.bert,
            input_ids = input_ids, 
            input_mask = attention_mask, 
            segment_ids = token_type_ids, 
            bin_label_ids = bin_label_ids,
            lstm_for_value = self.lstm_for_value)
        
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train' or mode == "pre-train":
                return logits
            else:
                return pooled_output, logits


class BertForMultiTask(BertPreTrainedModel):

    def __init__(self, config, args):

        super(BertForMultiTask, self).__init__(config)
        self.args = args
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.lstm_for_value = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True)
        if args.context:
            self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_slu = nn.Linear(config.hidden_size, args.num_labels)
        self.classifier_ori = nn.Linear(config.hidden_size, args.num_labels_parser)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,
                feature_ext=False, mode=None, loss_fct=None, bin_label_ids=None):
        '''
        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = False)
        pooled_output = self.dense(encoded_layer_12.mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        '''
        pooled_output = get_slot_emb(self.args, self.bert,
                                     input_ids=input_ids,
                                     input_mask=attention_mask,
                                     segment_ids=token_type_ids,
                                     bin_label_ids=bin_label_ids,
                                     lstm_for_value=self.lstm_for_value)

        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits_slu = self.classifier_slu(pooled_output)
        logits_ori = self.classifier_ori(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                return logits_slu, logits_ori
            elif mode == "pre-train": # in case of pre-training, slu is the same as ori
                return logits_slu
            elif mode == "bothbranch":
                return pooled_output, logits_slu, logits_ori
            else:
                return pooled_output, logits_slu # only return logits_slu
