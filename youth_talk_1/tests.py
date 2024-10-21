import sys
from unittest import TestCase

from simple_text_model import SimpleTextModel


lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
shakespear = "Let me not to the marriage of true minds. Admit impediments. Love is not love. Which alters when it alteration finds. Or bends with the remover to remove. O no! it is an ever-fixed mark"

class ISCRITests(TestCase):


    def test_version(self):
        print(sys.version)


    def test_normalize(self):
        m = SimpleTextModel()
        s = "Cyril Vincent éàè"
        s = m.normalize(s)
        self.assertEqual("cyril vincent eae", s)

    def test_split_phrases(self):
        m = SimpleTextModel()
        s = "ab\ncd.e"
        l = m.split_phrases(s)
        self.assertEqual(["ab", "cd", "e"], l)

    def test_tokenize(self):
        m = SimpleTextModel()
        res = m.tokenize(shakespear)
        print(res)
        res = m.tokenize(lorem)
        print(res)

    def test_split_2_words(self):
        m = SimpleTextModel()
        s = m.split_phrases(lorem)[0]
        res = m.split_2_words(s)
        print(res)

    def test_count(self):
        m = SimpleTextModel()
        res = m.tokenize(shakespear)
        dico = m.count(res)
        print(dico)

    def test_group(self):
        m = SimpleTextModel()
        res = m.tokenize(shakespear)
        dico = m.count(res)
        res = m.grouping(dico)
        print(res)