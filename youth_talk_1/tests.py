import sys
from unittest import TestCase
from NLTK_model import NLTKModel
from sqlentities import Topic, Lema

lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
shakespear = "Let me not to the marriage of true minds. Admit impediments. Love is not love. Which alters when it alteration finds. Or bends with the remover to remove. O no! it is an ever-fixed mark wedding wedd marriage mariage war"
tanzania = "am in Tanzania, but I am a Congomani scout of fesco, and association de scouts du sud kivu, district scout du lac Tanganyika, for my part, I wanted to talk about the south of Congomani scouts in Tanzania who fled, I mean refugees, from the Congo who are in Tanzania, on the Tanzanian side I grew up I request the World Scout Association to try to contact the TSA Tanzania Scout Association for permission to be given to Congomani scouts present in the Nyarugusu camp to continue scout movements and to grow. The scouts grew up from the beginning of the camp but in recent years they have been tamed without any reason and they have tried asking for permission reached other young people to travel to the scout headquarters in Tanzania TSA to ask for help with the matter now they have received answers every time they try to remind you they give a promise we will work on it today the camp leaders wrote a letter saying that in order for a scout to be in the camp a letter is required from for the minister of interior to give permission to the existing group in the camp now what is going on in the nyarugusu camp, there are more than 900000 young people and children and there are more than 545 scouts, the young people are suffering from not having a group they like."

class ISCRITests(TestCase):

    def test_version(self):
        print(sys.version)

    def test_normalize(self):
        m = NLTKModel()
        s = "Cyril Vincent éàè"
        s = m.normalize(s)
        self.assertEqual("cyril vincent eae", s)

    def test_split_phrases(self):
        m = NLTKModel()
        s = "ab\ncd.e"
        l = m.split_phrases(s)
        self.assertEqual(["ab", "cd", "e"], l)

    def test_tokenize(self):
        m = NLTKModel()
        res = m.tokenize(shakespear)
        print(res)
        res = m.tokenize(lorem)
        print(res)

    # def test_split_2_words(self):
    #     m = SimpleTextModel()
    #     s = m.split_phrases(lorem)[0]
    #     res = m.split_2_words(s)
    #     print(res)

    # def test_is_synonym(self):
    #     syno = Synonyms(search_string='mother')
    #     res = syno.find_synonyms()
    #     print(res)
    #     syno = Synonyms(search_string='father')
    #     res = syno.find_synonyms()
    #     print(res)

    def test_wordnet(self):
        import os
        os.environ["NLTK_DATA"] = r"C:\Users\conta\git-CVC\Skema\git-youth_talk\youth_talk_1\nltk"
        import nltk
        nltk.download('wordnet')

    def test_synonym_antonym_extractor(self):
        import os
        os.environ["NLTK_DATA"] = r"C:\Users\conta\git-CVC\Skema\git-youth_talk\youth_talk_1\nltk"
        phrase = "mother"
        from nltk.corpus import wordnet
        synonyms = []
        for syn in wordnet.synsets(phrase):
            for l in syn.lemmas():
                synonyms.append(l.name())
        print(set(synonyms))

    def test_get_synonyms(self):
        m = NLTKModel()
        res = m.get_synonyms("running")
        print(res)

    def test_count(self):
        m = NLTKModel()
        res = m.tokenize(shakespear)
        dico = m.count(res)
        print(dico)

    def test_group_synonyms(self):
        m = NLTKModel()
        t1 = Topic()
        t1.label = "marriage"
        t1.count = 1
        l1 = Lema()
        l1.label = "marriage"
        t1.lemas.append(l1)
        t2 = Topic()
        t2.label = "wedding"
        t2.count = 2
        l2 = Lema()
        l2.label = "wedding"
        t2.lemas.append(l2)
        l3 = Lema()
        l3.label = "wedd"
        t2.lemas.append(l3)
        m.topics = {"marriage": t1, "wedding": t2}
        m.group_synonyms()
        print(m.topics)

    def test_group(self):
        m = NLTKModel()
        res = m.tokenize(shakespear)
        # res = m.tokenize(tanzania)
        dico = m.count(res)
        res = m.grouping(dico)
        print(m.topics)

    def test_textrank(self):
        m = NLTKModel()
        text = m.normalize(tanzania)
        phrases = m.textrank(text)
        print(phrases)
