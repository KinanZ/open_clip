import re
import numpy as np
from nltk.tokenize import word_tokenize

groups = [
    ["ventral", "dorsal"],  # parietal
    ["medial", "temporal"],  # mittel vs ausen
    ["frontal", "okzipital"],  # vorne hinten
    ["vordere", "hintere"],
    [["frische", "neue", "akute", "subakute"], ["bekannte", "alte", "ältere", "chronische"]],  # Zeit

    # Blutung
    ["Parenchymblutung", "Massenblutung", "Kontusionsblutung", "Kontusionsblutung", "Stammganglienblutung",
     "Ventrikeleinblutung", "Ventrikelblutung", "Aneurysmablutung", "Hirnblutung", "Stammgangleinblutung",
     "Flaxhämatom", "Epiduralhämatom", "Galeahämatom",
     ["Subduralhämatom", "SDH", "Subduralblutung"], ["Subarachnoidalblutung", "SAB"]],

    # Infarkt
    ["Posteriorinfarkt", "Kleinhirninfarkt", "Mediainfarkt", "Kleinhirninfarkte", "Mediastrominfarkt",
     "Territorialinfarkt"],
    ["Mediateilinfarkt", "Posteriorteilinfarkt", "Anteriorteilinfarkt", "MCA-Teilinfarkt"],
    ["Kleinhirnischämie", "Mediaischämie", "Territorialischämie"],
]

def flatten(foo):
    # don't flatten strings.
    for x in foo:
        if hasattr(x, '__iter__') and not isinstance(x, str):
            for y in flatten(x):
                yield y
        else:
            yield x


class SetAugmenter:
    def __init__(self, options):
        self.options = options

    def aug_set(self, sent):
        return np.random.choice(self.options)


class ReplaceAugmenter:
    def __init__(self, groups):
        self.ant_probability = .5
        self.syn_probability = .5
        self.flip_probability = .5

        self.antonym_changes = 0
        self.synonym_changes = 0

        self.rl_dict = {'rechts':'links', ' re ':' li ',
                        'links':'rechts', ' li ':' re ',
                        'rechten':'linken', 'linken':'rechten'}
        self.ant_dict = {}
        self.syn_dict = {}

        for group in groups:
            whole_set = set(flatten(group))
            for synonyms in group:
                if type(synonyms) == str:
                    synonyms = [synonyms]

                reduced_set = list(whole_set - set(synonyms))
                for word in synonyms:
                    self.ant_dict[word] = reduced_set

                    if len(synonyms) > 1:
                        syn_set = set(synonyms)
                        syn_set.remove(word)
                        self.syn_dict[word] = list(syn_set)

        self.rl_pattern = re.compile("|".join(self.rl_dict.keys()))
        self.ant_pattern = re.compile("|".join(self.ant_dict.keys()))
        self.syn_pattern = re.compile("|".join(self.syn_dict.keys()))

    def __getitem__(self, word):
        ant_set = self.ant_dict[word]
        if np.random.random() < self.ant_probability:
            self.antonym_changes += 1
            return np.random.choice(ant_set)
        else:
            return word

    def aug_negative(self, sent):
        # genrate a wrong caption by substituting antonyms
        # make sure that there is at least one change
        options = re.findall(self.ant_pattern, sent)
        count = len(options)
        if count == 0:
            # no chance to generate a negative caption
            return sent
        scores = np.random.random(count)
        replace = scores < self.ant_probability
        if not np.any(replace):
            replace[np.argmin(scores)] = True
        replace_dict = dict(zip(options, replace))
        def repl(matchobj):
            word = matchobj.group(0)
            if replace_dict[word]:
                return np.random.choice(self.ant_dict[word])
            else:
                return word

        sent_out = re.sub(self.ant_pattern, repl, sent)
        return sent_out

    def aug_positive(self, sent):
        # generate a currect caption by substituting synonyms
        def repl(matchobj):
            word = matchobj.group(0)
            if np.random.random() < self.syn_probability:
                return np.random.choice(self.syn_dict[word])
            else:
                return word
        sent_out = re.sub(self.syn_pattern, repl, sent)
        return sent_out

    def aug_flip_horizontal(self, sent):
        # flip horizontal must always replace all
        def repl(matchobj):
            word = matchobj.group(0)
            if np.random.random() < self.flip_probability:
                return self.rl_dict[word]
            else:
                return word
        return re.sub(self.rl_pattern, repl, sent)


def skip_some_words(sent):
    words = word_tokenize(sent)
    if np.floor(len(words)/4) > 0:
        num_skip = np.random.randint(0, np.floor(len(words)/4))
    else:
        num_skip = 0
    idx_skip = [np.random.randint(0, len(words)) for i in range(num_skip)]
    for idx in idx_skip:
        words[idx] = '_'

    new_sent = ' '.join(words)
    return new_sent