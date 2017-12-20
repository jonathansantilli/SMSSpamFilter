from os import path

from expects import expect, equal, be_false, be_true

from spamdetector.file_helper import FileHelper

with description('FileHelper'):
    with before.all:
        self.spam_ham_examples = path.dirname(path.abspath(__file__)) + '/data/invalid_spam_ham_examples.txt'

    with it('should verify if a file exist'):
        do_not_exists = FileHelper().exist_path('invalid_path')
        exists =  FileHelper().exist_path(self.spam_ham_examples)

        expect(do_not_exists).to(be_false)
        expect(exists).to(be_true)

    with it('should read a file using a pattern'):
        pattern_separated_lines = FileHelper().read_pattern_separated_file(self.spam_ham_examples, '\t')

        expect(len(pattern_separated_lines)).to(equal(11))
