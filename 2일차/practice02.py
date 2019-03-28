from collections import defaultdict
import math

def sent_processing(lines):

    if isinstance(lines, list):
        lines = [line.strip().split(" ") for line in lines]

        corpus = []
        for line in lines:
            sent = []
            for word in line:
                word = tuple(word.rsplit("/", 1))
                sent.append(word)
            corpus.append(sent)

        return corpus

    elif isinstance(lines, str):
        line = []
        for word in lines.strip().split(" "):
            word = tuple(word.rsplit("/", 1))
            line.append(word)
        return line

    else:
        print("wrong type of input sentence")
        exit(1)

def train(corpus):

    def bigram_count(sent):
        poslist = [pos for _, pos in sent]
        return [(pos0, pos1) for pos0, pos1 in zip(poslist, poslist[1:])]

    pos2words = defaultdict(lambda: defaultdict(int)) # number of (word, tag)
    trans = defaultdict(int) # bigram count --> (tag-1, tag)
    bos = defaultdict(int) # count for bos bigram --> number of (BOS, tag)

    # sent format: [(word, tag), (word, tag), ....)]
    for sent in corpus: # counting
        for word, pos in sent:
            pos2words[pos][word] +=1

        for bigram in bigram_count(sent):
            trans[bigram] +=1

        bos[sent[0][1]] +=1 # number of (BOS, tag) bigram
        trans[(sent[-1][1], 'EOS')] +=1 # number of (tag, EOS) bigram

    # obervation (x|y) - emission prob.
    base = # P(y) for every y (count for each tag)
    pos2words_ =  # log(p(x, y)/p(y)) for every (x, y)

    # p(y_k|p_(k-1)) - transition prob.
    base = defaultdict(int)
    for (pos0, pos1), count in trans.items():
		# p(y_(k-1))

    trans_ = # log P(y_k, y_(k-1))/p(y_(k-1))

    # BOS
    base = sum(bos.values()) # p(BOS)
    bos_ = # log P(tag|BOS) 

    return pos2words_, trans_, bos_

class HMM_tagger(object):
    def __init__(self, pos2words, trans, bos):
        self.pos2words = pos2words
        self.trans = trans
        self.bos = bos
        self.unk = -15
        self.eos ='EOS'

    def sent_log_prob(self, sent):
        # emission prob.
        log_prob = # get emission prob. for each (w, t), otherwise unk value

        # bos
        log_prob += bos.get(sent[0][1], self.unk) # get BOS prob for the first (w, t)

        # transition prob.
        bigrams = [(t0, t1) for (_, t0), (_, t1) in zip(sent, sent[1:])] # every bigram in sentence
        log_prob+= # get transition prob. 

        # eos
        log_prob += self.trans.get(
            (sent[-1][1], self.eos), self.unk)

        # length norm.
        log_prob /= len(sent)

        return log_prob


with open("corpus.ko.tm.full-written.km-tok-pos.dev", "r", encoding='utf-8') as f:
    lines = f.readlines()

corpus = sent_processing(lines)
pos2words, trans, bos = train(corpus)
tagger = HMM_tagger(pos2words, trans, bos)
test_sent1= "감기/CMC 는/fjb 줄이/YBD 다/fmof ./g"
test_sent2= "감기/fmotg 는/fjb 줄/CMC 이다/fjj ./g"
print("%s: %f" % (test_sent1, tagger.sent_log_prob(sent_processing(test_sent1))))
print("%s: %f" % (test_sent2, tagger.sent_log_prob(sent_processing(test_sent2))))

