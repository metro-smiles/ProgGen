#!/usr/bin/env python
# coding=utf-8

import argparse

from proggen.data import Dataset
from proggen.llm2prog import LLM2Prog, template_OriginalScreenshot_AnnotatedContour_WithoutSemanticLabels

def llm2prog_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='collision')
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    print(args)
    dataset = Dataset(args.dataset, args.split, seed=0,)
    data = dataset[0]
    trajs, screenshots = data['trajs'], data['frames']
    llm2prog = LLM2Prog(template_OriginalScreenshot_AnnotatedContour_WithoutSemanticLabels)
    code = llm2prog(trajs, screenshots, screenshots.shape[-2]/10.)
    print(code)

if __name__ == '__main__':
    llm2prog_main()


