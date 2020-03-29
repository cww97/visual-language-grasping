import os
import re

import yaml
from torchtext import data


class Data(object):

    class DataSet(data.TabularDataset):

        @staticmethod
        def sort_key(ex):
            return len(ex.text)

        def __init__(self, text_field: data.Field, filename):
            def clean_str(string):
                """
                Tokenization/string cleaning for all datasets except for SST.
                Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
                """
                string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
                string = re.sub(r"\'s", " \'s", string)
                string = re.sub(r"\'ve", " \'ve", string)
                string = re.sub(r"n\'t", " n\'t", string)
                string = re.sub(r"\'re", " \'re", string)
                string = re.sub(r"\'d", " \'d", string)
                string = re.sub(r"\'ll", " \'ll", string)
                string = re.sub(r",", " , ", string)
                string = re.sub(r"!", " ! ", string)
                string = re.sub(r"\(", " \\( ", string)
                string = re.sub(r"\)", " \\) ", string)
                string = re.sub(r"\?", " \\? ", string)
                string = re.sub(r"\s{2,}", " ", string)
                return string.strip().split()

            text_field.tokenize = clean_str
            fields = [('text', text_field)]
            super().__init__(filename, format='tsv', fields=fields)

    def __init__(self, filename=os.path.join(os.path.dirname(__file__), 'sample.tsv')):
        self.text_field = data.Field(lower=True)
        self.dataset = Data.DataSet(self.text_field, filename)
        self.text_field.build_vocab(self.dataset)

    def get_tensor(self, x: str):
        x = self.text_field.preprocess(x)
        return self.text_field.numericalize([x]).t()


def generate(inf, ouf):
    template = 'pick up the {color} {name}.\n'
    colors = {'blue', 'green', 'brown', 'orange', 'yellow', 'gray', 'red', 'purple', 'cyan', 'pink'}
    names = set()
    with open(inf, 'r') as f:
        blocks = yaml.safe_load(f)
        for name_list in blocks['names'].values():
            for name in name_list:
                names.add(name)

    results = [template.format(color=color, name=name) for color in colors for name in names]
    import random
    random.shuffle(results)
    with open(ouf, 'w') as f:
        f.writelines(results)


if __name__ == '__main__':
    import os
    # run this file to generate the sample
    inf = os.path.join(os.path.dirname(__file__), 'objects/blocks/blocks.yml')
    ouf = os.path.join(os.path.dirname(__file__), 'sample.tsv')
    generate(inf, ouf)
    test_data = Data()
    print(len(test_data.text_field.vocab))
    print(test_data.get_tenser('pick up the red cube'))
