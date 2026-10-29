"""Build the paper's additive CSRAG bias for vLLM SamplingParams."""
import string

from nltk.corpus import stopwords


def collect_stopword_ids(tokenizer):
    words = set(stopwords.words('english')) | set(string.punctuation)
    return {i for word in words for text in (word, ' ' + word)
            for i in tokenizer.encode(text, add_special_tokens=False)}


def build_logit_bias(tokenizer, facts, context, stop_ids, alpha=-1.0, beta=3.0):
    parametric = {i for fact in facts
                  for i in tokenizer.encode(fact, add_special_tokens=False)} - stop_ids
    contextual = set(tokenizer.encode(context, add_special_tokens=False)) - stop_ids
    bias = {i: alpha * (i in parametric) + beta * (i in contextual)
            for i in parametric | contextual}
    return {i: float(value) for i, value in bias.items() if value and 0 <= i < len(tokenizer)}
