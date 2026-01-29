
'''
Neighborhood

For each sample, you’re given a neighborhood: a set of target publication IDs (the “true neighbors” for that query patent). The code uses that neighborhood to create and evaluate candidate subqueries.

Candidate subqueries (patterns)

A subquery candidate is a small conjunction-like pattern built from tokens (title/abstract/claim/desc/CPC tokens). Each candidate corresponds to a retrieved set of publications.

Crucially, candidates are constructed to be “clean”:

Zero non-targets constraint (during candidate generation):
The code only keeps a candidate subquery if, for that candidate’s retrieved set, every retrieved publication is in the target neighborhood.
In other words, candidate subqueries are intended to have no negatives (or extremely few, depending on later filtering).

So candidates mostly differ by which targets they cover, not by trading off targets vs. non-targets.

Combine subqueries with OR

The final query for a sample is:

several selected subqueries

combined as:
subquery_1 OR subquery_2 OR ... OR subquery_k

This “OR-of-subqueries” lets you cover different parts of the neighborhood using different token patterns.

Duplicate removal

Many different token patterns can retrieve the same target set. Keeping all of them is redundant and bloats the candidate pool.

So the script deduplicates candidates by a key like:

(sorted target set, query token cost)

If two candidates hit the same targets with the same “cost,” it keeps only one.
This reduces search space for simulated annealing without losing coverage capability.

Simulated Annealing (SA) search over subsets

The code uses simulated annealing to choose which candidate subqueries to OR together.

State representation

A state is basically:

used_subqueries: which candidates are currently selected

unused_subqueries: which are available but not selected

covered_targets: union of targets covered by used subqueries

neg_count: count of non-targets introduced (penalized; often ~0 if candidates are clean)

Neighbor moves (your 50/50 rule)

At each step, SA proposes a neighboring state via:

50% chance: add one unused subquery

50% chance: remove one used subquery

Constraints (hard filters)

Moves are rejected if they violate:

query token limit (≤ 50)

character length limit

max retrieved docs / negative caps

Score function

The score is primarily:

number of targets covered by the current OR-query, i.e.
|union of target sets from selected subqueries|

Because candidates are designed to have zero non-targets, the optimizer can focus on maximizing target coverage.

In the actual code there’s also a small negative penalty term:

score = covered_targets - NEG_WEIGHT * neg_count
…but in spirit, yes: it’s “maximize targets,” since good candidates usually have neg_count ≈ 0.

Acceptance rule

Standard SA acceptance:

Always accept if score improves

Otherwise accept with probability exp((new_score - old_score)/T) where T cools over time
'''

import pandas as pd
import glob
from tqdm import tqdm
import gc
from collections import defaultdict
import datetime
import json
import itertools
import matplotlib.pyplot as plt
import copy
import math
from collections import Counter
import numpy as np
import gc
import time
import random
import pickle
import os
import shutil

import whoosh_utils


"""
a1 = np.arange(100000)
a2 = np.arange(100000)
for _ in tqdm(range(100000)):
    d = np.intersect1d(a1, a2)

a1 = np.arange(100000)
a2 = np.arange(100000)
for _ in tqdm(range(100000)):
    d = np.intersect1d(a1, a2)
"""


# Experiment notes  
# 
# negpick == 0 Only keep as a candidate when  
# Move CPC terms to the end to minimize wildcard usage  
# Limit CPC usage to at most one term  
# Remove second-order neighbors and unnecessary items
# 
# 
# 
# 
# 


# TRAIN_PKL_PATH = '/kaggle/input/sa-cpc-title-abst-clm-desc-10-5-use-allpub-svpkl/'
# TRAIN_PKL_PATH = ''

# TRAIN_PKL_PATH = '/kaggle/input/sa-cpc-title-abst-clm-inf-desc-10-5-save-pkl/'

TRAIN_PKL_PATH = '/kaggle/input/sa-cpc-title-abst-clm-inf-cupy-save/'


IS_TRAIN = False
IS_REDUCE_TRAIN = False

IS_ADD_NEG_0_PATTERN = True


NEG_MAX_COUNT = 1000
PUB_MAX_COUNT = 50 + NEG_MAX_COUNT # 10
SUB_MAX_COUNT = 50 + NEG_MAX_COUNT # 100

PATTERN_NUM_MAX = 10000

CHAR_LIMIT = 9000


NEG_WEIGHT = 0.01 # 0.1


T0 = 1
T1 = 0.1

max_time = 1 # 5


if IS_TRAIN:
    nn_path = '/kaggle/input/create-valid-index-add-claim/nn_df_for_index.csv'
    nn_df = pd.read_csv(nn_path)
    print('len(nn_df)', len(nn_df))

    nn_df.to_csv('nn_df_for_index.csv', index=False)
else:
    nn_df = pd.read_csv('/kaggle/input/uspto-explainable-ai/test.csv')
    nn_df.to_csv('nn_df_for_index.csv', index=False)


import os
os.makedirs('nn_nn', exist_ok=True)
os.makedirs('reduce', exist_ok=True)


# publication_to_word: drop publications not present in nn_df


if IS_TRAIN and TRAIN_PKL_PATH != '':
    shutil.copy(f'{TRAIN_PKL_PATH}reduce/publication_to_word.pkl', 'reduce/publication_to_word.pkl')
else:


# publication to integer id mapping  
# 


if IS_TRAIN and TRAIN_PKL_PATH != '':
    shutil.copy(f'{TRAIN_PKL_PATH}reduce/pub_to_num.pkl', 'reduce/pub_to_num.pkl')
    shutil.copy(f'{TRAIN_PKL_PATH}reduce/num_to_pub.pkl', 'reduce/num_to_pub.pkl')
else:


# word_to_pub_set: drop words not needed for nn_df  
# 


os.makedirs('word_to_pub_set_chunk', exist_ok=True)


if IS_TRAIN and TRAIN_PKL_PATH != '':
    # shutil.copy(f'{TRAIN_PKL_PATH}reduce/word_to_pub_set.pkl', 'reduce/word_to_pub_set.pkl')
    shutil.copy(f'{TRAIN_PKL_PATH}reduce/word_to_number.pkl', 'reduce/word_to_number.pkl')
    shutil.copy(f'{TRAIN_PKL_PATH}reduce/number_to_word.pkl', 'reduce/number_to_word.pkl')
    shutil.copy(f'{TRAIN_PKL_PATH}reduce/word_to_pubcount.pkl', 'reduce/word_to_pubcount.pkl')
    shutil.copy(f'{TRAIN_PKL_PATH}reduce/publication_to_word.pkl', 'reduce/publication_to_word.pkl')
else:


# Load Whoosh index 


if IS_TRAIN:
    # train_idx = whoosh_utils.load_index('/kaggle/input/create-valid-index-add-claim/test_index')
    train_idx = whoosh_utils.load_index('/kaggle/input/uspto-test-index/test_index')
    searcher = whoosh_utils.get_searcher(train_idx)
    qp = whoosh_utils.get_query_parser()

    # query = 'ti:balloons OR ti:string'
    query = '(cpc:A01H6/42 ab:cuttings) OR (ab:regal-ab:rooting) OR (ab:magenta-ab:umbels) OR (ti:geranium-ti:variety-ab:cherry-ab:foliage-ab:pink-clm:geranium) OR (ti:bravo-ab:bicolored) OR (ab:foliage-ab:geranium-ab:regal-clm:geranium-detd:duchess) OR (ti:debutante-ti:pelargonium) OR (ti:geranium-ab:bright-ab:garden-ab:geranium-ab:mounded-ab:zonal-clm:geranium-clm:oglger14007-clm:zonal-detd:oglger14007) OR (ti:geranium-ti:variety-ab:blossoms-ab:foliage-ab:orchid-clm:geranium) OR (ti:louise-ab:ivy) OR (ti:hots) OR (ti:geranium-ti:variety-ab:orange-ab:rain-clm:geranium)'
    print(whoosh_utils.execute_query(query, qp, searcher)[:5])


# Load preprocessed artifacts  
# 


if IS_TRAIN and TRAIN_PKL_PATH != '':
    chunk_path = TRAIN_PKL_PATH + 'reduce/'
else:
    chunk_path = 'reduce/'
word_to_pub_set = {}
for i in tqdm(range(21)):
    _word_to_pub_set = pickle.load(open(chunk_path + f'word_to_pub_set_chunk_{i}.pkl', 'rb'))
    for k, v in _word_to_pub_set.items():
        # word_to_pub_set[k] = v
        word_to_pub_set[k] = np.array(np.sort(v), dtype=np.int32)


word_to_pubcount = pickle.load(open(base_path + 'word_to_pubcount.pkl', 'rb'))


"""
Set/array of publication numbers that contain a given token word(cpc,title,abstract)
Kept as sets/arrays to compute recall quickly
Example: word_to_pub_set[cpc1] → set([pub1, pub2, pub3])
"""

# word_to_pub_set = pickle.load(open(base_path + 'word_to_pub_set.pkl', 'rb'))


publication_to_word = pickle.load(open(base_path + 'publication_to_word.pkl', 'rb'))


pub_to_num = pickle.load(open(base_path + 'pub_to_num.pkl', 'rb'))
num_to_pub = pickle.load(open(base_path + 'num_to_pub.pkl', 'rb'))


counts = []
for _, cnt in word_to_pubcount.items():
    counts.append(cnt)
plt.hist(counts)


nn_df = pd.read_csv('nn_df_for_index.csv')


for i in range(nn_df.values.shape[0]):
    for j in range(nn_df.values.shape[1]):
        nn_df.values[i, j] = pub_to_num[nn_df.values[i, j]]


neighbors = nn_df.values[:, 1:]


word_pattern_to_pub_set_list = [defaultdict(set) for _ in range(len(neighbors))]
word_pattern_to_pub_set_all = dict()
add_count_list = []


for i in tqdm(range(len(neighbors))):

    word_pattern_to_pub_set = word_pattern_to_pub_set_list[i]
    add_list = []


    nn_list = neighbors[i]
    # nn_array = np.array(neighbors[i], dtype=np.int32)

    nn_array = neighbors[i]
    nn_array = np.sort(nn_array)
    nn_array = np.array(nn_array, dtype=np.int32)
    nn_set = set(nn_list)

    # use_words_seen = set()
    use_words_len_max = -1

    # Reset caches per row to prevent memory growth
    pub_set_cache = {}
    use_words_cache = {}

    t_sum = 0
    for r in range(1, 3):
        debug_count=0
        for nn_comb in itertools.combinations(nn_list, r):
            counts = []
            if nn_comb[0] not in use_words_cache:
                use_words_cache[nn_comb[0]] = set([w for w in publication_to_word[nn_comb[0]]])
            use_words = use_words_cache[nn_comb[0]].copy()
            for nn in nn_comb[1:]:
                if not nn in use_words_cache:
                    use_words_cache[nn] = set([w for w in publication_to_word[nn]])
                use_words &= use_words_cache[nn]

            use_words = tuple(sorted(list(use_words)))

            if len(use_words) == 0: # or use_words in use_words_seen
                continue

            use_words_sorted = sorted(use_words, key=lambda x: word_to_pubcount[x])

            use_words = tuple([w for w in use_words if w in set(use_words_sorted)])
            # use_words_len_max = max(len(use_words), use_words_len_max)

            use_words_final = []
            """
            # Previously pre-converted very large arrays to sets to speed up set creation; no longer needed
            if isinstance(word_to_pub_set[use_words_sorted[0]], set):
                all_pub_set = word_to_pub_set[use_words_sorted[0]].copy()
            else:
                if not use_words_sorted[0] in pub_set_cache:
                    pub_set_cache[use_words_sorted[0]] = set(word_to_pub_set[use_words_sorted[0]])
                all_pub_set = pub_set_cache[use_words_sorted[0]].copy()
            """
            all_pub_set = word_to_pub_set[use_words_sorted[0]]

            use_words_final.append(use_words_sorted[0])

            cpc_count = 1 if number_to_word[use_words_sorted[0]][:3] == 'cpc' else 0


            if not len(all_pub_set) <= r:
                for word in use_words_sorted[1:]:

                    if number_to_word[word][:3] == 'cpc' and cpc_count == 1:
                        continue

                    before_len = len(all_pub_set)
                    """
                    if isinstance(word_to_pub_set[word], set):
                        all_pub_set &= word_to_pub_set[word]
                    else:
                        if not word in pub_set_cache:
                            pub_set_cache[word] = set(word_to_pub_set[word])
                        all_pub_set &= pub_set_cache[word]
                    """


                    # all_pub_set = np.intersect1d(all_pub_set, word_to_pub_set[word], assume_unique = True)
                    all_pub_set = np.intersect1d(all_pub_set, word_to_pub_set[word], assume_unique = True)


                    after_len = len(all_pub_set)

                    # Only append the token if it reduces the candidate set size
                    if after_len < before_len:
                        use_words_final.append(word)

                        if number_to_word[word][:3] == 'cpc':
                            cpc_count += 1

                    # No longer needed because CPC wildcards are handled differently
                    """
                    else:
                        # cpcでなければ追加
                        if number_to_word[word][:3] != 'cpc':
                            use_words_final.append(word)
                    """
                    # use_words_final.append(word)

                    counts.append(len(all_pub_set))

                    # Stop once the candidate set shrinks to the neighbor-combination size
                    if len(all_pub_set) <= r: 
                        break

                    # Stop once the candidate set is fully contained in the neighbor set
                    cnt1 = len(all_pub_set)

                    # cnt2 = len(all_pub_set & nn_set)
                    # cnt2 = len(np.intersect1d(all_pub_set, nn_array, assume_unique = True))

                    if cnt1 <= 50:                 
                        t1 = time.time()
                        cnt2 = len(np.intersect1d(all_pub_set, nn_array, assume_unique = True))
                        # cnt2 = np.in1d(all_pub_set ,nn_array, assume_unique = True)
                        # cnt2 = cnt2.sum()

                        t_sum += time.time() - t1

                        if cnt1 == cnt2:
                            break


            use_words = tuple(use_words_final)
            use_words_len_max = max(len(use_words), use_words_len_max)

            if debug_count < -1:
                print(counts)

            debug_count += 1
            if len(all_pub_set) <= 50: # len(all_pub_set) <= 50なしだと、多分OOM + 間に合わない


                all_pub_set = np.asarray(all_pub_set)


                nn_comb_set = set(nn_comb)
                all_pub_set = set(all_pub_set)
                # nn_setで、追加できるなら追加
                add_set = all_pub_set & nn_set
                nn_comb_set = nn_comb_set | add_set

                # neg_pub_set = all_pub_set - nn_comb_set

                # if len(neg_pub_set) == 0:
                if len(nn_comb_set) == len(all_pub_set): # Add only when negatives are zero
                    add_list.append((use_words, nn_comb_set, all_pub_set))

        if i <= 50:
            print(t_sum)
            print(f'r >= {r} len(add_list): {len(add_list)}')        
            print('use_words_len_max ', use_words_len_max )

    add_count = 0
    for word_pattern, pub_set, all_pub_set in add_list:
        if word_pattern in word_pattern_to_pub_set:
            continue

        word_pattern_to_pub_set[word_pattern] = pub_set
        word_pattern_to_pub_set_all[word_pattern] = all_pub_set

        add_count += 1

    if i <= 50:
        print(f'len(add_list): {len(add_list)}, add_count: {add_count}')
    add_count_list.append(add_count)


print(pd.DataFrame(add_count_list).describe())


"""
def count_query_len(word_pattern):
    is_replace_AND = False
    prefix_set = set([number_to_word[word].split(':')[0] for word in word_pattern])
    if 'cpc' not in prefix_set:
        is_replace_AND = True

    if len(word_pattern) == 1 or is_replace_AND:
        token_len = 2
    else:
        token_len = 3
    return token_len
"""
# cpcもtoken_len=2でできるようになったので、変更
def count_query_len(word_pattern):
    token_len = 2 
    return token_len


def filtering_word_pattern_to_pub_set_list(word_pattern_to_pub_set_list):

    word_pattern_to_pub_set_list_new = []
    # for word_pattern_to_pub_set in word_pattern_to_pub_set_list:
    for sample_idx, word_pattern_to_pub_set in enumerate(word_pattern_to_pub_set_list):

        word_pattern_to_pub_set_new = defaultdict(set)
        sort_keys = []

        # データの処理とソートキーの生成
        for word_pattern, pub_set in word_pattern_to_pub_set.items():
            max_count = len(word_pattern_to_pub_set_all[word_pattern])
            if max_count <= PUB_MAX_COUNT:
                # word_pattern_to_pub_set_new[word_pattern] = pub_set

                neg_pub_set = word_pattern_to_pub_set_all[word_pattern] - pub_set                
                neg_weight = len(neg_pub_set)

                sort_key = (len(pub_set), -count_query_len(word_pattern), -neg_weight)
                sort_keys.append((sort_key, word_pattern))
                word_pattern_to_pub_set_new[word_pattern] = pub_set

        # Sort by the sort key
        sorted_word_patterns = sorted(sort_keys, key=lambda x: x[0], reverse=True)

        # Rebuild the list based on the sorted order
        word_pattern_to_pub_set_new = [(word_pattern, word_pattern_to_pub_set_new[word_pattern]) for _, word_pattern in sorted_word_patterns]

        # Deduplicate 
        seen = set()
        new_state = []
        for word_pattern, pub_set in word_pattern_to_pub_set_new:
            # pattern = tuple(sorted(list(pub_set)))
            pattern = (tuple(sorted(list(pub_set))), count_query_len(word_pattern))
            if pattern in seen:
                continue
            seen.add(pattern)
            new_state.append((word_pattern, pub_set))
        word_pattern_to_pub_set_new = new_state


        # Keep top patterns ranked by (#targets in results, #total publications in results) PATTERN_NUM_MAX 件
        word_pattern_to_pub_set_new = word_pattern_to_pub_set_new[:PATTERN_NUM_MAX]

        # This step is redundant (same structure as new_state)
        word_pattern_to_pub_set = []
        for word_pattern, pub_set in word_pattern_to_pub_set_new:
            word_pattern_to_pub_set.append((word_pattern, pub_set))

        word_pattern_to_pub_set_list_new.append(word_pattern_to_pub_set)

    word_pattern_to_pub_set_list = word_pattern_to_pub_set_list_new

    return word_pattern_to_pub_set_list


word_pattern_to_pub_set_list = filtering_word_pattern_to_pub_set_list(word_pattern_to_pub_set_list)


recall_list = []
for word_pattern_to_pub_set in word_pattern_to_pub_set_list:
    true_set = set()
    # for word_pattern, true_pub_set in word_pattern_to_pub_set.items():
    for word_pattern, true_pub_set in word_pattern_to_pub_set:
        true_set |= true_pub_set
    recall_list.append(len(true_set))

print(pd.DataFrame(recall_list).describe())
plt.hist(recall_list)
plt.title('recall distribution')
plt.show()


# cpcの組み合わせ数
lis = []
for word_pattern_to_pub_set in word_pattern_to_pub_set_list:
    lis.append(len(word_pattern_to_pub_set))
print(pd.DataFrame(lis).describe())
plt.hist(lis)
plt.title('combincation count')


from functools import total_ordering

@total_ordering
class State:
    def __init__(self, use_word_pattern_list, not_use_word_pattern_list, use_true_pub_set, neg_count, score=None,
                 is_hard_penalty=False):
        self.use_word_pattern_list = use_word_pattern_list
        self.not_use_word_pattern_list = not_use_word_pattern_list
        # self.use_pub_set = use_pub_set
        self.use_true_pub_set = use_true_pub_set
        self.neg_count = neg_count

        # 簡単にrecall上げられるサンプルなら、negativeの重みを強める
        self.is_hard_penalty = is_hard_penalty

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score

    def calc_score(self):

        # self.score = SUB_MAX_COUNT * (len(self.use_true_pub_set) - 0.25 * self.neg_count)

        if self.is_hard_penalty:
            # self.score = len(self.use_true_pub_set) - 0.25 * self.neg_count
            self.score = len(self.use_true_pub_set) - NEG_WEIGHT * self.neg_count
        else:
            self.score = len(self.use_true_pub_set) - NEG_WEIGHT * self.neg_count


class Timer:
    def __init__(self):
        self.start = time.time()

    def get_current_time(self):
        return (time.time() - self.start)

# Simulated annealingのための確率を計算
def calc_sa_p(new_score, score, T):
    score_diff = new_score - score
    if score_diff >= 0:
        return 1
    else:
        return math.exp(score_diff / T)


def ap50(preds, labels):
    precisions = list()
    n_label = len(labels)
    n_found = 0
    for e, i in enumerate(preds):
        if i in labels:
            n_found += 1
        precisions.append(n_found/(e+1)) # this is the line that is probably incorrect for competition 
    return sum(precisions)/50


# Simulated annealing  
# 
# 
# 


len(neighbors.flatten())


word_pattern_to_neg_list = []

for sample_idx, word_pattern_to_pub_set in tqdm(enumerate(word_pattern_to_pub_set_list)):

    word_pattern_to_neg_count = dict()
    for c, pub_set in word_pattern_to_pub_set:
        neg_pub_set = word_pattern_to_pub_set_all[c] - pub_set

        word_pattern_to_neg_count[c] = len(neg_pub_set)

    word_pattern_to_neg_list.append(word_pattern_to_neg_count)


def word_pattern_to_query(word_pattern):
    query_rm_cpc = [s for s in word_pattern if number_to_word[s].split(':')[0] != 'cpc']
    query_cpc = [s for s in word_pattern if number_to_word[s].split(':')[0] == 'cpc']
    word_pattern = query_rm_cpc + query_cpc

    query = ''
    for i, word in enumerate(word_pattern):
        if number_to_word[word].split(':')[0] == 'cpc':
            if i != len(word_pattern) - 1:
                word = number_to_word[word]

                # cpc:A01H6/77* これだと770~779も含まれるので、この後に対処する
                query += f'{word}*'

                # cpc:A01H6/7? を追加して70~79に制限する。
                base = word.split('/')[0]
                suffix = word.split('/')[1]
                suffix = suffix[:-1] + '?'
                query += f'{base}/{suffix}'
            else:
                # 最後なら、ワイルドカードを追加する必要がない
                query += f'{number_to_word[word]}'
        else:
            query += f'{number_to_word[word]}-'

    # 最後のは取り除く
    # if len(word_pattern) >= 2 and query[-1] != '?':
    if query[-1] == '-':
        query = query[:-1]

    query = "(" + query + ')'
    return query


word_pattern_to_char_len = dict()

for word_pattern_to_pub_set in word_pattern_to_pub_set_list:
    for word_pattern, _ in word_pattern_to_pub_set:
        query = word_pattern_to_query(word_pattern)
        word_pattern_to_char_len[word_pattern] = len(query) + 4


len(word_pattern_to_char_len)


counts = []
for w, c in word_pattern_to_char_len.items():
    counts.append(c)

print(pd.DataFrame(counts).describe())
plt.hist(counts)


query_list = []
default_query = 'ti:bread'

# For visualization
success_count = 0
score_list = []
true_pub_set_len_list = [0] * len(word_pattern_to_pub_set_list)
pub_set_len_list = [0] * len(word_pattern_to_pub_set_list)
neg_count_list = [0] * len(word_pattern_to_pub_set_list)

# for _, word_pattern_counter in tqdm(enumerate(word_pattern_counter_list), total=len(word_pattern_counter_list)):
for sample_idx, word_pattern_to_pub_set in tqdm(enumerate(word_pattern_to_pub_set_list), total=len(word_pattern_to_pub_set_list)):
    word_pattern_to_neg_count = word_pattern_to_neg_list[sample_idx]

    true_set = set(nn_df.values[sample_idx, 1:])
    labels = list(nn_df.values[sample_idx, 1:])
    labels = [num_to_pub[pub] for pub in labels]

    # When there are no query candidates
    if len(word_pattern_to_pub_set) == 0:
        score_list.append(0)
        query_list.append(default_query)
        continue

    word_pattern_to_pub_set_dict = dict()
    for word_pattern, pub_set in word_pattern_to_pub_set:        
        word_pattern_to_pub_set_dict[word_pattern] = pub_set

    use_word_pattern_list = []
    not_use_word_pattern_list = []
    for word_pattern, _ in word_pattern_to_pub_set:
        not_use_word_pattern_list.append(word_pattern)

    # Precompute query token lengths (count_query_len was slow)
    word_pattern_to_query_len = dict()
    for word_pattern, _ in word_pattern_to_pub_set:
        word_pattern_to_query_len[word_pattern] = count_query_len(word_pattern)


    curr_state = State([], [], set(), 0, 0)
    curr_state.score = 0

    # queryの差分更新用
    curr_query_len = -1
    best_query_len = -1
    curr_char_len = -4
    best_char_len = -4


    best_states = []
    timer = Timer()

    # Simulated annealingによるスコアの変化の可視化用
    _score_list = []

    best_state = copy.deepcopy(curr_state)
    # best_state.calc_score()
    best_state.score = 0

    # どの近傍が使われたかをcount
    pattern_count = [0, 0]

    # debugで時間計測用
    time_sum = [0]

    # neg_pickedのペナルティを強めるか
    is_hard_penalty = False

    while True:
        curr_time = timer.get_current_time()

        # if curr_time > max_time:
        if curr_time > max_time:
            break

        t = curr_time / max_time
        T = T0**(1-t) * T1**t

        p = random.random()

        if p >= 0.5:
            act = 'add_pattern'
        else:
            act = 'remove_pattern'

        next_state = State([], 
                           [], 
                           set(), 0, 0,
                           is_hard_penalty)

        if act == 'add_pattern':
            N = len(not_use_word_pattern_list)
            if N == 0:
                continue
            idx = random.randint(0, N-1)
            c = not_use_word_pattern_list[idx]

            # query_len = count_query_len(c)
            query_len = word_pattern_to_query_len[c]
            char_len = word_pattern_to_char_len[c]

            if curr_query_len + query_len > 50 or curr_char_len + char_len > CHAR_LIMIT:
                is_exceed_query_limit = True
            else:
                is_exceed_query_limit = False
        elif act == 'remove_pattern':
            N = len(use_word_pattern_list)
            if N == 0:
                continue
            idx = random.randint(0, N-1)
            c = use_word_pattern_list[idx]

            query_len = word_pattern_to_query_len[c]
            char_len = word_pattern_to_char_len[c]
            is_exceed_query_limit = False

        if is_exceed_query_limit:
            continue

        # Use incremental updates for speed add_pattern 0.6s remove_pattern 1.4s
        if act == 'add_pattern':
            neg_count = curr_state.neg_count + word_pattern_to_neg_count[c]
            use_true_pub_set = curr_state.use_true_pub_set | word_pattern_to_pub_set_dict[c]

            update_count = len(use_true_pub_set) - len(curr_state.use_true_pub_set)
            # recall向上しないならcontinue
            if update_count == 0:
                continue
        else:
            neg_count = 0
            use_true_pub_set = set()

            for word_pattern in use_word_pattern_list:
                if act == 'remove_pattern' and word_pattern == c:
                    continue

                neg_count += word_pattern_to_neg_count[word_pattern]
                if len(use_true_pub_set) + neg_count > SUB_MAX_COUNT:
                    break

                use_true_pub_set |= word_pattern_to_pub_set_dict[word_pattern]


        # Reject candidates when result size exceeds thresholds
        if len(use_true_pub_set) + neg_count > SUB_MAX_COUNT or neg_count > NEG_MAX_COUNT:
            continue

        # next_state.use_pub_set = use_pub_set
        next_state.neg_count = neg_count
        next_state.use_true_pub_set = use_true_pub_set
        next_state.calc_score()

        # Acceptance probability
        sa_p = calc_sa_p(next_state.score, curr_state.score, T)

        if random.random() < sa_p:
            curr_state = next_state
            # curr_query_list = next_query_list
            if act == 'add_pattern':
                c = not_use_word_pattern_list.pop(idx)
                use_word_pattern_list.append(c)
                curr_query_len += query_len
                curr_char_len += char_len
            elif act == 'remove_pattern':
                c = use_word_pattern_list.pop(idx)
                not_use_word_pattern_list.append(c)
                curr_query_len -= query_len
                curr_char_len -= char_len
        else:
            pass

        # _score_list.append(len(curr_state.use_true_pub_set))
        _score_list.append(curr_state.score)


        if curr_state.score > best_state.score:
            best_state = curr_state
            best_state.use_word_pattern_list = copy.deepcopy(use_word_pattern_list)
            # best_query_list = copy.deepcopy(curr_query_list)
            best_query_len = curr_query_len
            best_char_len = curr_char_len

            if best_state.score >= 35 and not is_hard_penalty:
                is_hard_penalty = True
                curr_state.is_hard_penalty = True
                curr_state.calc_score()
                best_state.is_hard_penalty = True
                best_state.calc_score()


        if act == 'add_pattern':
            pattern_count[0] += 1
        else:
            pattern_count[1] += 1
    # print(time_sum)
    # print(pattern_count)
    # Debug
    if sample_idx < 50:
        plt.plot(_score_list)
        plt.show()

    # Build the final Submission query from best_state
    if best_state is None:
        print('error: best_state is None')
        print(1/0)
        # query_list.append(default_query)
    else:
        """
        use_true_pub_set = best_state.use_true_pub_set
        use_pub_set = best_state.use_pub_set
        """
        use_true_pub_set = set()
        use_pub_set = set()
        for c in best_state.use_word_pattern_list:
            use_pub_set |= word_pattern_to_pub_set_all[c]
            use_true_pub_set |= word_pattern_to_pub_set_dict[c]

        use_word_pattern_list = best_state.use_word_pattern_list

        true_pub_set_len_list[sample_idx] = len(use_true_pub_set)
        pub_set_len_list[sample_idx] = len(use_pub_set)
        neg_count_list[sample_idx] = best_state.neg_count

        query = ''
        for i, word_pattern in enumerate(use_word_pattern_list):
            prev_query = query

            _query = word_pattern_to_query(word_pattern)

            query += _query

            if i != len(use_word_pattern_list) - 1:
                query += ' OR '

            if whoosh_utils.count_query_tokens(query) > 50 or len(query) > CHAR_LIMIT + 500:

                # Raise an error and stop
                print('query limit error')
                print(1/0)

                query = prev_query
                break

        # assert best_query_len == len(query.split())

        query_list.append(query)
        success_count += 1

        if IS_TRAIN:
            results = whoosh_utils.execute_query(query, qp, searcher)
            result_set = set(results)
            n_pick = len(set(labels) & result_set)
            score = ap50(results + [-1] * (50 - len(results)), labels)
        else:
            score = 0
            n_pick = 0

        if sample_idx < 50:
            print(query)
            print('ap50', score)
            print('n_pick', n_pick)
            print('true_count', true_pub_set_len_list[sample_idx])
            print('neg_count', neg_count_list[sample_idx])
            print(len(query), best_char_len)

        score_list.append(score)


np.mean(true_pub_set_len_list)


plt.hist(true_pub_set_len_list)
plt.title('true_pub_set_len_list distribution')


np.mean(score_list)


plt.hist(score_list)
plt.title('score_list distribution')


np.mean(pub_set_len_list)


plt.hist(pub_set_len_list)
plt.title('pub_set_len_list distribution')


np.mean(neg_count_list)


plt.hist(neg_count_list)
plt.title('neg_count_list distribution')


# pub_set_allで計算した、正しいneg_count_list


neg_count_list_2 = [pub_set_len_list[i] - true_pub_set_len_list[i]
                  for i in range(len(pub_set_len_list))]
np.mean(neg_count_list_2)


plt.hist(neg_count_list_2)
plt.title('neg_count_list_2 distribution')


neg_count_list_diff = [neg_count_list[i] - neg_count_list_2[i] for i in range(len(neg_count_list))]
print(np.mean(neg_count_list_diff ))
plt.hist(neg_count_list_diff)
plt.title('neg_count_list_diff distribution')


if IS_TRAIN:
    [print(s) for s in true_pub_set_len_list[:50]]
else:
    print(score_list[:10])


if IS_TRAIN:
    [print(s) for s in score_list[:50]]
else:
    print(score_list[:10])


query_count_list = [whoosh_utils.count_query_tokens(query) for query in query_list]
print(max(query_count_list))
plt.hist(query_count_list)

plt.title('token len distribution')
plt.show()


character_len_list = [len(query) for query in query_list]
print(max(character_len_list))
plt.hist(character_len_list)
plt.title('character len distribution')
plt.show()


len(nn_df), len(query_list), success_count


# Create out-of-fold (OOF) diagnostics


if IS_TRAIN:
    columns = ['query_list',
               'true_pub_set_len_list',
               'score_list',
               'pub_set_len_list',
               'neg_count_list',
               'neg_count_list_2',
               'query_count_list',
               'character_len_list']
    values = np.array([ query_list,
               true_pub_set_len_list,
               score_list,
               pub_set_len_list,
               neg_count_list,
               neg_count_list_2,
               query_count_list,
               character_len_list]).T


    oof_df = pd.DataFrame(values, columns=columns)

    print(oof_df.head())
    oof_df.to_csv('oof_df.csv', index=False)


plt.scatter(true_pub_set_len_list, score_list)
plt.xlabel('true_pub_set_len_list')
plt.ylabel('score_list')


idxs = [i for i in range(len(true_pub_set_len_list)) if score_list[i] != 0]
true_pub_set_len_list = [true_pub_set_len_list[i] for i in idxs]
score_list = [score_list[i] for i in idxs]
print(len(true_pub_set_len_list))
plt.scatter(true_pub_set_len_list, score_list)
plt.xlabel('true_pub_set_len_list')
plt.ylabel('score_list')


# Submission

# If 'id:' appears in the query, replace it


default_query = 'ti:bread'

for i in range(len(query_list)):
    if 'id:' in query_list[i]:
        query_list[i] = default_query

"""
for i in range(len(query_list)):
    try:
        result = whoosh_utils.execute_query(query_list[i], qp, searcher)
    except Exception as e:
        query_list[i] = default_query
"""


# Fix query execution errors


for i in range(len(query_list)):
    if query_list[i] == '':
        query_list[i] = default_query


"""
if not IS_TRAIN:
    # train_idx = whoosh_utils.load_index('/kaggle/input/create-valid-index-add-claim/test_index')
    train_idx = whoosh_utils.load_index('/kaggle/input/uspto-test-index/test_index')
    searcher = whoosh_utils.get_searcher(train_idx)
    qp = whoosh_utils.get_query_parser()

    # query = 'ti:balloons OR ti:string'
    query = '(cpc:A01H6/42 ab:cuttings) OR (ab:regal-ab:rooting) OR (ab:magenta-ab:umbels) OR (ti:geranium-ti:variety-ab:cherry-ab:foliage-ab:pink-clm:geranium) OR (ti:bravo-ab:bicolored) OR (ab:foliage-ab:geranium-ab:regal-clm:geranium-detd:duchess) OR (ti:debutante-ti:pelargonium) OR (ti:geranium-ab:bright-ab:garden-ab:geranium-ab:mounded-ab:zonal-clm:geranium-clm:oglger14007-clm:zonal-detd:oglger14007) OR (ti:geranium-ti:variety-ab:blossoms-ab:foliage-ab:orchid-clm:geranium) OR (ti:louise-ab:ivy) OR (ti:hots) OR (ti:geranium-ti:variety-ab:orange-ab:rain-clm:geranium)'
    print(whoosh_utils.execute_query(query, qp, searcher)[:5])
"""


"""
for i in range(len(query_list)):
    try:
        result = whoosh_utils.execute_query(query_list[i], qp, searcher)
    except Exception as e:
        print(e)
        query_list[i] = default_query
"""


if not IS_TRAIN:
    sub = pd.read_csv('/kaggle/input/uspto-explainable-ai/sample_Submission.csv')

    sub['query'] = query_list

    sub.to_csv('Submission.csv', index=False)

    print(sub)


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


