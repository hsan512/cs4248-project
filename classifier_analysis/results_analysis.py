import pandas as pd

def analyze_df(file_path, threshold=0.1):
    df = pd.read_csv(file_path)

    def get_val(row, label_type):
        label = row[label_type]
        return row[f'prob_{label}']

    df['p_true'] = df.apply(lambda r: get_val(r, 'true_label'), axis=1)
    df['p_pred'] = df.apply(lambda r: get_val(r, 'pred_label'), axis=1)
    df['diff'] = (df['p_true'] - df['p_pred']).abs()

    unique_pairs = df.groupby(['true_label', 'pred_label'])

    for (true_lab, pred_lab), group in unique_pairs:
        count = len(group)
        print(f"\n[CASE] True: {true_lab.upper()} | Predicted: {pred_lab.upper()}")
        print(f"Total occurrences: {count}")

        examples = group.head(15)
        print("Examples:")
        for i, row in examples.iterrows():
            print(f"  - Text: {row['text']}")
            print(f"    P({true_lab}): {row['p_true']:.4f} vs P({pred_lab}): {row['p_pred']:.4f} | Diff: {row['diff']:.4f}")
        print("-" * 30)

# Run the analysis
analyze_df('test_predictions.csv')