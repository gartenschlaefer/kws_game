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
    header_str = '\\begin{table}[ht!]\n\\begin{center}\n'

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
    return '\\bottomrule\n\\label{{{}}}\n\\end{{tabular}}\n\\end{{center}}\n\\end{{table}}\n\\FloatBarrier\n\\noindent'.format(label)



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
        'acc': ['{:.2f}'.format(float(re.sub(r'[acc: \[\]]', '', a))) for a in re.findall(self.acc_re, ti)]
        }

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



class LatexTableMaker(LatexTableFunctions):
  """
  latex table maker class
  """

  def __init__(self, in_file, extraction_type='feature_selection'):

    # arguments
    self.in_file = in_file
    self.extraction_type = extraction_type

    # vars
    self.in_file_name = in_file.split('/')[-1].split('.')[0]
    self.out_file_name = 'none'

    # my eval exists
    self.f = open(in_file, "r")
    self.file_content = self.f.read()

    # my evaluation (used for logs)
    self.my_eval_exists = (self.file_content.find('Eval my on arch') != -1)

    # regex
    self.log_instance_re = r'[\w+ 0-9 \-,:]+ Training on arch.+\n[\w+ 0-9 \-,:]+ Eval test on arch.+\n[\w+ 0-9 \-,:]+ Eval my on arch.+' if self.my_eval_exists else r'[\w+ 0-9 \-,:]+ Training on arch.+\n[\w+ 0-9 \-,:]+ Eval test on arch.+'
    self.norm_re = r'norm[01]'
    self.feature_sel_re = r'_c[01]d[01]d[01]e[01]_'
    self.feature_size_re = r'_f-[1-3]x[0-9]+x[0-9]+_'
    self.acc_re = r'acc: \[[0-9]+\.[0-9]+\]'

    # input file
    self.in_file = None

    # colors
    self.thesis_color = 'ThesisColor'


  def extract_table(self, out_file, caption=''):
    """
    extract table
    """

    # out file name setting
    self.out_file_name = out_file.split('/')[-1].split('.')[0]

    # extract desired table
    if self.extraction_type == 'feature_selection': table_string = self.create_feature_selection_tables(caption=caption)

    # save to file
    with open(out_file, 'w') as f: print(table_string, file=f)

    return table_string


  def create_feature_selection_tables(self, caption='', label=None):
    """
    table for feature selection, data form log file
    """

    # get infos
    norm_list, feature_sel_list, data_size_list, acc_list = self.get_train_instance_infos()

    # format row entries
    row_n1 = [[f[0], f[1], f[2], f[3], a[0], a[1]] for n, f, d, a in zip(norm_list, feature_sel_list, data_size_list, acc_list) if int(n) and int(d[0]) == 1] if self.my_eval_exists else [[f[0], f[1], f[2], f[3], a[0]] for n, f, d, a in zip(norm_list, feature_sel_list, data_size_list, acc_list) if int(n) and int(d[0]) == 1]
    row_n0 = [[f[0], f[1], f[2], f[3], a[0], a[1]] for n, f, d, a in zip(norm_list, feature_sel_list, data_size_list, acc_list) if not int(n) and int(d[0]) == 1] if self.my_eval_exists else [[f[0], f[1], f[2], f[3], a[0]] for n, f, d, a in zip(norm_list, feature_sel_list, data_size_list, acc_list) if not int(n) and int(d[0]) == 1]
    
    # table params
    col_spaces_cm = [1] * 4 + [1.5] * 4 if self.my_eval_exists else [1] * 4 + [1.5] * 2
    sep = [False] * 4 + [True] + [False] * 3 if self.my_eval_exists else [False] * 4 + [True] + [False] * 1
    #titles = ['cepstral', 'delta', 'double delta', 'energy', 'acc test', 'acc my', 'acc test norm', 'acc my norm'] if self.my_eval_exists else ['cepstral', 'delta', 'double delta', 'energy', 'acc test', 'acc test norm']
    
    # titles
    titles = ['c', 'd', 'dd', 'e', 'acc test', 'acc my', 'acc test norm', 'acc my norm'] if self.my_eval_exists else ['c', 'd', 'dd', 'e', 'acc test', 'acc test norm']
    
    # caption
    if not len(caption): caption = 'Feature Selection ' + self.in_file_name.replace('_', ' ')
    
    # label
    if label is None: label = 'tab:' + self.out_file_name.replace('tab_', '')

    # create row entries
    row_entries_norm = [rn0 + rn1[4:6] for rn1, rn0 in zip(row_n1, row_n0)] if self.my_eval_exists else [rn0 + [rn1[4]] for rn1, rn0 in zip(row_n1, row_n0)]

    # header
    table_str = self.table_header(col_spaces_cm=col_spaces_cm, sep=[], caption=caption)

    # title string
    table_str += self.table_titles(titles=['Feature Groups', 'Accuracy'], multicol=[4, 2], textbf=True, color=False)
    table_str += self.table_titles(titles=titles, textbf=True, midrule=True, color=False)

    # row entries
    table_str += ''.join([self.table_row_entry(row_entry) for row_entry in row_entries_norm])

    # footer
    table_str += self.table_footer(label=label)

    return table_str


  def get_train_instance_infos(self):
    """
    extract train instance infos from the input file
    """

    # init lists
    norm_list, feature_sel_list, data_size_list, acc_list = [], [], [], []

    # read info from log file
    #with open(self.in_file, 'r') as f:

    # whole string
    #whole_string = ''.join(f.readlines())
    
    # get train instances
    train_instances = re.findall(self.log_instance_re,  self.file_content)

    # go through each training instance
    for ti in train_instances:

      # norm
      norm = re.sub('norm', '', re.findall(self.norm_re, ti)[0])

      # feature
      feature_sel = [int(i) for i in re.sub(r'[cde_]', '', re.findall(self.feature_sel_re, ti)[0])]

      # datasize
      data_size = re.split('x', re.sub(r'[f\-_]', '', re.findall(self.feature_size_re, ti)[0]))
      
      # accuracy
      acc = ['{:.2f}'.format(float(re.sub(r'[acc: \[\]]', '', a))) for a in re.findall(self.acc_re, ti)]

      # append to lists
      norm_list.append(norm), feature_sel_list.append(feature_sel), data_size_list.append(data_size), acc_list.append(acc)

    return norm_list, feature_sel_list, data_size_list, acc_list



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
