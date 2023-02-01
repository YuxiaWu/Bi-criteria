from .slot_loader import SLOT_Loader

backbone_loader_map = {
                            'bert': SLOT_Loader,
                            'bert_MultiTask': SLOT_Loader,
                      }