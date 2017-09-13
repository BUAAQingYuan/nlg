__author__ = 'PC-LiNing'

from tensorflow.python.platform import gfile
import tensorflow as tf
import sys
import os
import re


buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


# source_path, target_path is int
def read_data(source_path, target_path, max_size=None):
  data_set = [[] for _ in buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]


def basic_tokenizer(sentence):
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,tokenizer=None, normalize_digits=True):
    sentence = tf.compat.as_bytes(sentence)
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                            tokenizer, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, from_vocabulary_size,
                 to_vocabulary_size, tokenizer=None):
  """
    Args:
      data_dir: directory in which the data sets will be stored.
      from_train_path: path to the file that includes "from" training samples.
      to_train_path: path to the file that includes "to" training samples.
      from_dev_path: path to the file that includes "from" dev samples.
      to_dev_path: path to the file that includes "to" dev samples.
      from_vocabulary_size: size of the "from language" vocabulary to create and use.
      to_vocabulary_size: size of the "to language" vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for "from language" training data-set,
        (2) path to the token-ids for "to language" training data-set,
        (3) path to the token-ids for "from language" development data-set,
        (4) path to the token-ids for "to language" development data-set,
        (5) path to the "from language" vocabulary file,
        (6) path to the "to language" vocabulary file.
    """
  # Create vocabularies of the appropriate sizes.
  to_vocab_path = os.path.join(data_dir, "vocab%d.to" % to_vocabulary_size)
  from_vocab_path = os.path.join(data_dir, "vocab%d.from" % from_vocabulary_size)
  create_vocabulary(to_vocab_path, to_train_path , to_vocabulary_size, tokenizer)
  create_vocabulary(from_vocab_path, from_train_path , from_vocabulary_size, tokenizer)

  # Create token ids for the training data.
  to_train_ids_path = to_train_path + (".ids%d" % to_vocabulary_size)
  from_train_ids_path = from_train_path + (".ids%d" % from_vocabulary_size)
  data_to_token_ids(to_train_path, to_train_ids_path, to_vocab_path, tokenizer)
  data_to_token_ids(from_train_path, from_train_ids_path, from_vocab_path, tokenizer)

  # Create token ids for the development data.
  to_dev_ids_path = to_dev_path + (".ids%d" % to_vocabulary_size)
  from_dev_ids_path = from_dev_path + (".ids%d" % from_vocabulary_size)
  data_to_token_ids(to_dev_path, to_dev_ids_path, to_vocab_path, tokenizer)
  data_to_token_ids(from_dev_path, from_dev_ids_path, from_vocab_path, tokenizer)
  return (from_train_ids_path, to_train_ids_path, from_dev_ids_path, to_dev_ids_path, from_vocab_path, to_vocab_path)

# test
"""
from_train_path = "./train/100_en.txtaa"
to_train_path = "./train/100_fr.txtaa"
from_dev_path = "./dev/newstest2013.en"
to_dev_path = "./dev/newstest2013.fr"
data_dir = "./"
from_vocabulary_size = 20000
to_vocabulary_size = 20000
from_train, to_train, from_dev, to_dev, from_vocab, to_vocab = prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, from_vocabulary_size,to_vocabulary_size, tokenizer=None)
print("from_train_ids_path: "+from_train)
print("to_train_ids_path: "+to_train)
print("from_dev_ids_path: "+from_dev)
print("to_dev_ids_path: "+to_dev)
print("from_vocab_path: "+from_vocab)
print("to_vocab_path: "+to_vocab)
"""