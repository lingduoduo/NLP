# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
from utils import evaluate_places

eval_corpus_path = 'birth_dev.tsv'
n_lines = len(open(eval_corpus_path, 'r', encoding='utf8').readlines())
predictions = ['London'] * n_lines
total, correct = evaluate_places(eval_corpus_path, predictions)
print(f'Total: {total} - Correct: {correct} - Accuracy: {correct / total * 100}')
