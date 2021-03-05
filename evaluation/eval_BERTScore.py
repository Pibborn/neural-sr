__author__='thiagocastroferreira'

"""
Author: Organizers of SR'20
Date: 23/04/2020
Description:
    This script aims to evaluate the output of NLG models using BERT-Score.
    
    ARGS:
        usage: eval.py [-h] -R REFERENCE -H HYPOTHESIS [-lng LANGUAGE] [-nr NUM_REFS]

        optional arguments:
          -h, --help            show this help message and exit
          -R REFERENCE, --reference REFERENCE
                                reference translation
          -H HYPOTHESIS, --hypothesis HYPOTHESIS
                                hypothesis translation
          -lng LANGUAGE, --language LANGUAGE
                                evaluated language
          -nr NUM_REFS, --num_refs NUM_REFS
                                number of references

    EXAMPLE:
        ENGLISH: 
            python3 eval.py -R data/en/references/reference -H data/en/hypothesis -nr 4
        RUSSIAN:
            python3 eval.py -R data/ru/reference -H data/ru/hypothesis -lng ru -nr 1
"""

import argparse
import codecs
import copy
import os
import logging

from bert_score import score
from tabulate import tabulate


def parse(refs_path, hyps_path, num_refs, lng='en'):
    logging.info('STARTING TO PARSE INPUTS...')
    print('STARTING TO PARSE INPUTS...')
    # references
    references = []
    for i in range(num_refs):
        fname = refs_path + str(i) if num_refs > 1 else refs_path
        with codecs.open(fname, 'r', 'utf-8') as f:
            texts = f.read().split('\n')
            for j, text in enumerate(texts):
                if text.startswith('# text = '):
                    text = text.replace('# text = ', '')
                elif text.startswith('#text = '):
                    text = text.replace('#text = ', '')
                else:
                    continue
                
                #if len(references) <= j:
                references.append([text])
                #else:
                #    references[j].append(text)

    # hypothesis
    hypothesis = []
    with codecs.open(hyps_path, 'r', 'utf-8') as f:
        for text in f:
        
          if text.startswith('# text = '):
             text = text.replace('# text = ', '')
          elif text.startswith('#text = '):
             text = text.replace('#text = ', '')
          else:
                    continue
          hypothesis.append(text) 
  
    print('len(hypothesis), len(references)', len(hypothesis), len(references))
    logging.info('FINISHING TO PARSE INPUTS...')
    print('FINISHING TO PARSE INPUTS...')
    return references, hypothesis


def bert_score_(references, hypothesis, lng='en'):
    logging.info('STARTING TO COMPUTE BERT SCORE...')
    print('STARTING TO COMPUTE BERT SCORE...')
    for i, refs in enumerate(references):
        references[i] = [ref for ref in refs if ref.strip() != '']

    scores = []
    P, R, F1 = score(hypothesis, references,   lang=lng) ###
    print('P, R, F1', P, R, F1)
#    for hyp, refs in zip(hypothesis, references):
#        print('.')
#        P, R, F1 = score([hyp], [refs],   lang=lng) ### 
#        scores.append(F1)
    scores = list(F1)
    logging.info('FINISHING TO COMPUTE BERT SCORE...')
    print('FINISHING TO COMPUTE BERT SCORE...')
    bert = float(sum(scores) / len(scores))
    return bert


if __name__ == '__main__':
    FORMAT = '%(levelname)s: %(asctime)-15s - %(message)s'
    logging.basicConfig(filename='eval.log', level=logging.INFO, format=FORMAT)

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-R", "--reference", help="reference translation", required=True)
    argParser.add_argument("-H", "--hypothesis", help="hypothesis translation", required=True)
    argParser.add_argument("-lng", "--language", help="evaluated language", default='en')
    argParser.add_argument("-nr", "--num_refs", help="number of references", type=int, default=1)

    args = argParser.parse_args()

    logging.info('READING INPUTS...')
    print('READING INPUTS...')
    refs_path = args.reference
    hyps_path = args.hypothesis
    lng = args.language
    num_refs = args.num_refs
    logging.info('FINISHING TO READ INPUTS...')
    print('FINISHING TO READ INPUTS...')

    references, hypothesis = parse(refs_path, hyps_path, num_refs, lng)

    logging.info('STARTING EVALUATION...')
    print('STARTING EVALUATION...')
    headers, values = [], []
    headers.append('BERT-SCORE')
    bert = bert_score_(references, hypothesis, lng=lng)
    values.append(round(bert, 2))
    logging.info('FINISHING EVALUATION...')
    print('FINISHING EVALUATION...')

    logging.info('PRINTING RESULTS...')
    print('PRINTING RESULTS...')
    print(tabulate([values], headers=headers))
