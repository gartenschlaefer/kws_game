# --
# latex table maker

import re
import numpy as np


class LatexTableFunctions():
  """
  contains useful table creation functions
  """

  def table_header(self, col_spaces_cm, sep=[], caption=''):
    """
    table header
    """

    # begin table
    header_str = '\\begin{table}[ht!]\n\\small\n\\begin{center}\n'

    # add top caption
    if len(caption): header_str += self.table_caption(caption)

    # cells
    cells = [' M{{{}cm}} '.format(c) for c in col_spaces_cm]

    # double separation
    if len(sep): cells = ['|{}'.format(c) if s else c for c, s in zip(cells, sep)] 

    # begin tabular
    header_str += '\\begin{tabular}{' + ''.join(cells)  + '}\n'

    # top ruler
    header_str += '\\toprule\n'
    #header_str += '\\hline\n'

    return header_str


  def table_titles(self, titles, multicol=[], textbf=True, color=True, midrule=False):
    """
    table titles
    """

    # color
    title_str = '\\rowcolor{{{}}}\n'.format(self.thesis_color) if color else ''

    # add textbf
    if textbf: titles = ['\\textbf{{{}}}'.format(t) for t in titles]

    # multicolumn
    if len(multicol): titles = ['\\multicolumn{{{}}}{{c}}{{{}}}'.format(m, t) for t, m in zip(titles, multicol) if m > 1]

    # add titles
    title_str += ''.join(['{} & '.format(t) if i != len(titles)-1 else '{} \\\\\n'.format(t) for i, t in enumerate(titles)])

    # mid rule
    if midrule: title_str += '\\midrule\n'

    return title_str


  def table_row_entry(self, entries):
    """
    talbe row entries
    """
    return ''.join(['{} & '.format(e) if i != len(entries)-1 else '{} \\\\\n'.format(e) for i, e in enumerate(entries)])


  def table_caption(self, caption):
    """
    caption for table
    """
    return '\\caption{{{}}}\n'.format(caption)


  def table_footer(self, label):
    """
    footer for table
    """
    return '\\bottomrule\n\\label{{{}}}\n\\end{{tabular}}\n\\end{{center}}\n\\vspace{{-4mm}}\n\\end{{table}}\n\\FloatBarrier\n\\noindent'.format(label)



class LogExtractor():
  """
  extraction stuff for log entries
  """

  def __init__(self):

    # my evaluation (used for logs)
    #self.my_eval_exists = (self.file_content.find('Eval my on arch') != -1)

    # regex
    #self.log_instance_re = r'[\w+ 0-9 \-,:]+ Training on arch.+\n[\w+ 0-9 \-,:]+ Eval test on arch.+\n[\w+ 0-9 \-,:]+ Eval my on arch.+' if self.my_eval_exists else r'[\w+ 0-9 \-,:]+ Training on arch.+\n[\w+ 0-9 \-,:]+ Eval test on arch.+'
    self.log_instance_re = r'[\w+ 0-9 \-,:]+ Training on arch.+\n[\w+ 0-9 \-,:]+ Eval test on arch.+\n[\w+ 0-9 \-,:]+ Eval my on arch.+'
    self.arch_re = r'Training on arch: \[[\w \-]+\],'
    self.classes_re = r'_c[0-9]+n[01]m[01]_'
    self.n_examples_re = r'_n-[0-9]+_'
    self.randomize_onsets_re = r'_r[10]-[0-9]+_'
    self.mfcc_re = r'_mfcc[0-9]+-[0-9]+_'
    self.norm_re = r'norm[01]'
    self.feature_sel_re = r'_c[01]d[01]d[01]e[01]_'
    self.feature_size_re = r'_f-[1-3]x[0-9]+x[0-9]+'
    self.acc_re = r'acc: \[[0-9]+\.[0-9]+\]'


  def get_train_instance_infos(self, in_file):
    """
    extract train instance infos from the input file
    """

    # read file content
    with open(in_file, 'r') as f: file_content = f.read()

    # collection
    train_instance_dicts = []
    
    # get train instances
    train_instances = re.findall(self.log_instance_re, file_content)

    # go through each training instance
    for ti in train_instances:

      # create dict
      train_instance_dict = {
        'arch': re.sub(r'Training on arch: |([,\[\]])', '', re.findall(self.arch_re, ti)[0]), 
        'classes': re.sub(r'[_]', '', re.findall(self.classes_re, ti)[0]), 
        'n_examples': re.sub(r'[_]', '', re.findall(self.n_examples_re, ti)[0]),
        'randomize_onsets': re.sub(r'[_]', '', re.findall(self.randomize_onsets_re, ti)[0]),
        'mfcc': re.sub(r'[_]', '', re.findall(self.mfcc_re, ti)[0]),
        'feature_sel': [int(i) for i in re.sub(r'[cde_]', '', re.findall(self.feature_sel_re, ti)[0])], 
        'norm': re.sub('norm', '', re.findall(self.norm_re, ti)[0]), 
        'feature_size': re.split('x', re.sub(r'[f\-_]', '', re.findall(self.feature_size_re, ti)[0])),
        'acc': ['{:.2f}'.format(float(re.sub(r'[acc: \[\]]', '', a))) for a in re.findall(self.acc_re, ti)],
        'adv-it': None,
        'adv-model': None
        }

      # check if adversarial pre training is in the training instance
      if ti.find('adv-pre') != -1:

        # get adv params
        adv_params = re.findall(r'adv-pre[\w_\-0-9]+]', ti)[0]

        #print(adv_params)
        train_instance_dict.update({
          'adv-it': re.sub(r'[it\-_]', '', re.findall(r'it-[0-9]+_', adv_params)[0]),
          'adv-model': re.sub(r'model-', '', re.findall(r'model-[gd]', adv_params)[0])
          })

      # append
      train_instance_dicts.append(train_instance_dict)

    return train_instance_dicts


class LatexTableMakerCepstral(LatexTableFunctions, LogExtractor):
  """
  cepstral table maker
  """

  def __init__(self, in_file, out_file, caption='', label=''):

    # parent init
    super().__init__()

    # vars
    titles = ['arch', 'mfcc', 'norm', 'acc test', 'acc my']
    col_spaces_cm = [3, 2, 2, 2.5, 2.5]

    # get training instances
    train_instances_dict = self.get_train_instance_infos(in_file)

    # separators used to get training instances
    sep_combi = {'arch': ['conv-trad', 'conv-fstride', 'conv-jim'], 'mfcc': ['mfcc32-12', 'mfcc32-32'], 'norm': ['0', '1']}

    # all combinations
    separators = [(a, m, n) for a in sep_combi['arch'] for m in sep_combi['mfcc'] for n in sep_combi['norm']]

    # separation
    train_instances_sep = [[ti for ti in train_instances_dict if ti['arch'] == a and ti['mfcc'] == m and ti['norm'] == n] for a, m, n in separators]

    # row entries
    row_entries = [[tis[0]['arch'], tis[0]['mfcc'], tis[0]['norm'], '${:.2f} \\pm {:.2f}$'.format(np.mean(np.array([ti['acc'][0] for ti in tis]).astype(float)), np.sqrt(np.var(np.array([ti['acc'][0] for ti in tis]).astype(float)))), '${:.2f} \\pm {:.2f}$'.format(np.mean(np.array([ti['acc'][1] for ti in tis]).astype(float)), np.sqrt(np.var(np.array([ti['acc'][1] for ti in tis]).astype(float))))] for tis in train_instances_sep if len(tis)]
    
    # header
    table_str = self.table_header(col_spaces_cm=col_spaces_cm, sep=[], caption=caption)

    # title string
    table_str += self.table_titles(titles=titles, textbf=True, midrule=True, color=False)

    # row entries
    table_str += ''.join([self.table_row_entry(row_entry) for row_entry in row_entries])

    # footer
    table_str += self.table_footer(label=label)

    # message
    print("table_str: ", table_str)

    # save to file
    with open(out_file, 'w') as f: print(table_str, file=f, end='')



class LatexTableMakerMFCC(LatexTableFunctions, LogExtractor):
  """
  latex table maker class
  """

  def __init__(self, in_file, out_file, caption='', label=''):


    # parent init
    super().__init__()

    # vars
    titles = ['c', 'd', 'dd', 'e', 'acc test', 'acc my']
    col_spaces_cm = [1, 1, 1, 1, 2.5, 2.5]

    # get training instances
    train_instances_dict = self.get_train_instance_infos(in_file)

    # separators used to get training instances
    sep_combi = {'arch': ['conv-jim'], 'c': [0, 1], 'd': [0, 1], 'dd': [0, 1], 'e': [0, 1]}

    # all combinations
    separators = [(a, c, d, dd, e) for a in sep_combi['arch'] for c in sep_combi['c'] for d in sep_combi['d'] for dd in sep_combi['dd'] for e in sep_combi['e']]

    # separation
    train_instances_sep = [[ti for ti in train_instances_dict if ti['arch'] == a and ti['feature_sel'][0] == c and ti['feature_sel'][1] == d and ti['feature_sel'][2] == dd and ti['feature_sel'][3] == e] for a, c, d, dd, e in separators]

    # row entries
    row_entries = [[tis[0]['feature_sel'][0], tis[0]['feature_sel'][1], tis[0]['feature_sel'][2], tis[0]['feature_sel'][3], '${:.2f} \\pm {:.2f}$'.format(np.mean(np.array([ti['acc'][0] for ti in tis]).astype(float)), np.sqrt(np.var(np.array([ti['acc'][0] for ti in tis]).astype(float)))), '${:.2f} \\pm {:.2f}$'.format(np.mean(np.array([ti['acc'][1] for ti in tis]).astype(float)), np.sqrt(np.var(np.array([ti['acc'][1] for ti in tis]).astype(float))))] for tis in train_instances_sep if len(tis)]
    
    # header
    table_str = self.table_header(col_spaces_cm=col_spaces_cm, sep=[], caption=caption)

    # title string
    table_str += self.table_titles(titles=titles, textbf=True, midrule=True, color=False)

    # row entries
    table_str += ''.join([self.table_row_entry(row_entry) for row_entry in row_entries])

    # footer
    table_str += self.table_footer(label=label)

    # message
    print("table_str: ", table_str)

    # save to file
    with open(out_file, 'w') as f: print(table_str, file=f, end='')


class LatexTableMakerAdv(LatexTableFunctions, LogExtractor):
  """
  latex table maker class
  """

  def __init__(self, in_file, out_file, caption='', label=''):


    # parent init
    super().__init__()

    # vars
    titles = ['adv iterations', 'adv model', 'acc test', 'acc my']
    col_spaces_cm = [2, 2, 2.5, 2.5]

    # get training instances
    train_instances_dict = self.get_train_instance_infos(in_file)

    # separators used to get training instances
    sep_combi = {'arch': ['conv-jim'], 'adv-it': ['100', '1000'], 'adv-model': ['g', 'd']}

    # all combinations
    separators = [(a, it, m) for a in sep_combi['arch'] for it in sep_combi['adv-it'] for m in sep_combi['adv-model']]

    # separation
    train_instances_sep = [[ti for ti in train_instances_dict if ti['arch'] == a and ti['adv-it'] == it and ti['adv-model'] == m] for a, it, m in separators]

    # row entries
    row_entries = [[tis[0]['adv-it'], tis[0]['adv-model'], '${:.2f} \\pm {:.2f}$'.format(np.mean(np.array([ti['acc'][0] for ti in tis]).astype(float)), np.sqrt(np.var(np.array([ti['acc'][0] for ti in tis]).astype(float)))), '${:.2f} \\pm {:.2f}$'.format(np.mean(np.array([ti['acc'][1] for ti in tis]).astype(float)), np.sqrt(np.var(np.array([ti['acc'][1] for ti in tis]).astype(float))))] for tis in train_instances_sep if len(tis)]
    
    # header
    table_str = self.table_header(col_spaces_cm=col_spaces_cm, sep=[], caption=caption)

    # title string
    table_str += self.table_titles(titles=titles, textbf=True, midrule=True, color=False)

    # row entries
    table_str += ''.join([self.table_row_entry(row_entry) for row_entry in row_entries])

    # footer
    table_str += self.table_footer(label=label)

    # message
    print("table_str: ", table_str)

    # save to file
    with open(out_file, 'w') as f: print(table_str, file=f, end='')



class LatexTableMakerFinal(LatexTableFunctions, LogExtractor):
  """
  latex table maker class
  """

  def __init__(self, in_file, out_file, caption='', label=''):


    # parent init
    super().__init__()

    # vars
    titles = ['arch', 'norm', 'pre-train', 'acc test', 'acc my']
    col_spaces_cm = [3, 1, 2, 2.5, 2.5]

    # get training instances
    train_instances_dict = self.get_train_instance_infos(in_file)

    # separators used to get training instances
    sep_combi = {'norm': ['0', '1'], 'arch': ['conv-trad', 'conv-fstride', 'conv-jim'], 'adv-it': [None, '100']}

    # all combinations
    separators = [(n, a, it) for n in sep_combi['norm'] for a in sep_combi['arch'] for it in sep_combi['adv-it']]

    # separation
    train_instances_sep = [[ti for ti in train_instances_dict if ti['norm'] == n and ti['arch'] == a and ti['adv-it'] == it] for n, a, it in separators]

    # row entries
    row_entries = [[tis[0]['arch'], tis[0]['norm'], '-' if tis[0]['adv-it'] is None else tis[0]['adv-it'], '${:.2f} \\pm {:.2f}$'.format(np.mean(np.array([ti['acc'][0] for ti in tis]).astype(float)), np.sqrt(np.var(np.array([ti['acc'][0] for ti in tis]).astype(float)))), '${:.2f} \\pm {:.2f}$'.format(np.mean(np.array([ti['acc'][1] for ti in tis]).astype(float)), np.sqrt(np.var(np.array([ti['acc'][1] for ti in tis]).astype(float))))] for tis in train_instances_sep if len(tis)]
    
    # header
    table_str = self.table_header(col_spaces_cm=col_spaces_cm, sep=[], caption=caption)

    # title string
    table_str += self.table_titles(titles=titles, textbf=True, midrule=True, color=False)

    # row entries
    table_str += ''.join([self.table_row_entry(row_entry) for row_entry in row_entries])

    # footer
    table_str += self.table_footer(label=label)

    # message
    print("table_str: ", table_str)

    # save to file
    with open(out_file, 'w') as f: print(table_str, file=f, end='')



class LatexTableMakerAudiosetLabels(LatexTableFunctions):
  """
  latex table maker class
  """

  def __init__(self, all_label_file_dict, caption, label, out_file):

    # vars
    titles = ['label', 'train', 'test', 'validation', 'total']
    col_spaces_cm = [3, 3, 3, 3, 3]

    # row entries
    row_entries = [[label, len(set_dict['train']), len(set_dict['test']), len(set_dict['validation']), len(set_dict['train']) + len(set_dict['test']) + len(set_dict['validation'])] for label, set_dict in all_label_file_dict.items() if len(set_dict.keys())]

    # header
    table_str = self.table_header(col_spaces_cm=col_spaces_cm, sep=[], caption=caption)

    # title string
    table_str += self.table_titles(titles=titles, textbf=True, midrule=True, color=False)

    # row entries
    table_str += ''.join([self.table_row_entry(row_entry) for row_entry in row_entries])

    # footer
    table_str += self.table_footer(label=label)

    # message
    print("table_str: ", table_str)

    # save to file
    with open(out_file, 'w') as f: print(table_str, file=f)



if __name__ == '__main__':
  """
  main
  """
  
  # c5
  in_file, out_file, extraction_type = '../ignore/logs/ml_it500_c5_features_fc1.log', '../docu/thesis/4_practice/tables/b1_feature_selection/ml_it500_c5_features_fc1.tex', 'feature_selection'
  
  # c30
  in_file, out_file, extraction_type = '../ignore/logs/ml_it1000_c30_features_fc1.log', '../docu/thesis/4_practice/tables/b1_feature_selection/ml_it1000_c30_features_fc1.tex', 'feature_selection'

  # instances
  lt_maker = LatexTableMaker(in_file=in_file, extraction_type=extraction_type)

  # extract table
  tables = lt_maker.extract_table(out_file=out_file)

  print("table:\n{}".format(''.join(tables)))
