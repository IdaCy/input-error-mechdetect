import pandas as pd

# Load mean activations for text
clean_text = pd.read_csv('analyses/output/mean_activations_text-sentences_clean.csv', header=None)
syntax_error_text = pd.read_csv('analyses/output/mean_activations_text-sentences_syntax-error.csv', header=None)
typo_text = pd.read_csv('analyses/output/mean_activations_text-sentences_typo.csv', header=None)

# Compute differences for syntax error
differences_syntax_error = syntax_error_text - clean_text
differences_syntax_error.to_csv('analyses/output/differences_text_syntax_error.csv', index=False, header=False)
print("Differences for text (clean vs. syntax error) saved!")

# Compute differences for typo
differences_typo = typo_text - clean_text
differences_typo.to_csv('analyses/output/differences_text_typo.csv',
                        index=False, header=False)
print("Differences for text (clean vs. typo) saved!")
