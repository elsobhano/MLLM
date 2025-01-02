import pandas as pd
from sacrebleu.metrics import BLEU


File_Name = '/home/sobhan/Documents/Code/MLLM/ckpts/test_outputs_0.csv'


hypotheses = []
tgt_refs = []

df = pd.read_csv(File_Name, sep='|')
hypotheses.extend(df['hypotheses'].tolist())
tgt_refs.extend(df['targets'].tolist())
hypotheses = [text + ' .' for text in hypotheses]
tgt_refs = [text + ' .' for text in tgt_refs]
print(len(tgt_refs), len(hypotheses))
bleu = BLEU()
bleu_s = bleu.corpus_score(hypotheses, [tgt_refs]).score
print(bleu_s)