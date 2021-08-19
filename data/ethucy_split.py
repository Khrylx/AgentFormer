

def get_ethucy_split(dataset):
     seqs = [
          'biwi_eth',
          'biwi_hotel',
          'crowds_zara01',
          'crowds_zara02',
          'crowds_zara03',
          'students001',
          'students003',
          'uni_examples'
     ]

     if dataset == 'eth':
          test = ['biwi_eth']
     elif dataset == 'hotel':
          test = ['biwi_hotel']
     elif dataset == 'zara1':
          test = ['crowds_zara01']
     elif dataset == 'zara2':
          test = ['crowds_zara02']
     elif dataset == 'univ':
          test = ['students001', 'students003']

     train, val = [], []
     for seq in seqs:
          if seq in test:
               continue
          train.append(f'{seq}_train')
          val.append(f'{seq}_val')
     return train, val, test