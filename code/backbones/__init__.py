#from .bert import BERT, BertForConstrainClustering, BertForDTC, BertForKCL_Similarity, \
#    BertForKCL, BertForMCL

from .bert import BERT, BertForConstrainClustering, BertForDTC, BertForKCL_Similarity, \
    BertForKCL, BertForMCL, BertForMultiTask
from .glove import GloVeEmbeddingVectorizer
from .sae import get_stacked_autoencoder

backbones_map = {
                    'bert': BERT,
                    'bert_MultiTask': BertForMultiTask,
                }