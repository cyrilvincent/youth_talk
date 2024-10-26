import spacy
import pytextrank

# example text
text = "Compatibility of system of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."
text = "Let me not to the marriage of true minds. Admit impediments. Love is not love. Which alters when it alteration finds. Or bends with the remover to remove. O no! it is an ever-fixed mark wedding wedd marriage mariage war"
text = "am in Tanzania, but I am a Congomani scout of fesco, and association de scouts du sud kivu, district scout du lac Tanganyika, for my part, I wanted to talk about the south of Congomani scouts in Tanzania who fled, I mean refugees, from the Congo who are in Tanzania, on the Tanzanian side I grew up I request the World Scout Association to try to contact the TSA Tanzania Scout Association for permission to be given to Congomani scouts present in the Nyarugusu camp to continue scout movements and to grow. The scouts grew up from the beginning of the camp but in recent years they have been tamed without any reason and they have tried asking for permission reached other young people to travel to the scout headquarters in Tanzania TSA to ask for help with the matter now they have received answers every time they try to remind you they give a promise we will work on it today the camp leaders wrote a letter saying that in order for a scout to be in the camp a letter is required from for the minister of interior to give permission to the existing group in the camp now what is going on in the nyarugusu camp, there are more than 900000 young people and children and there are more than 545 scouts, the young people are suffering from not having a group they like."
text = "a short text"

# load a spaCy model, depending on language, scale, etc.
# python3 -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# add PyTextRank to the spaCy pipeline
nlp.add_pipe("textrank")
doc = nlp(text)

# examine the top-ranked phrases in the document
for phrase in doc._.phrases:
    print(phrase.text)
    print(phrase.rank, phrase.count)
    print(phrase.chunks)