"""
pdf handling
"""

import pdftotext
import filecmp

# open files to compare
with open('./ignore/pdf/thesis_v6.pdf', 'rb') as f: thesis_v6 = pdftotext.PDF(f)
with open('./ignore/pdf/thesis.pdf', 'rb') as f: thesis = pdftotext.PDF(f)

# compare length
print('len: ', len(thesis_v6))
print('len: ', len(thesis))

# compare content
for i, (p, q) in enumerate(zip(thesis_v6, thesis)):

  # page num
  print('page num: ', i)

  if p == q: print('okay')
  else: 
    print('----------------------nope')
    
    for l_p, l_q in zip(p, q):

      if l_p == l_q: print(l_p)
      else:
        print('diff:\n{}\n{}'.format(l_p, l_q))
        stop


print('\nBoth files are identical.')