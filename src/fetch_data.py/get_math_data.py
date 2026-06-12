from datasets import load_dataset, concatenate_datasets, DatasetDict
import pandas as pd


def read_math_dataset():
    # Load each subject and store in a list
    train_list = []
    test_list = []

    # The list of subjects provided by your error message
    subjects = [
        'algebra', 'counting_and_probability', 'geometry', 
        'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
    ]
    
    for sub in subjects:
        curr_ds = load_dataset("EleutherAI/hendrycks_math", sub)
        train_list.append(curr_ds['train'])
        test_list.append(curr_ds['test'])

    # Combine them into one master dataset
    dataset = DatasetDict({
        'train': concatenate_datasets(train_list),
        'test': concatenate_datasets(test_list)
    })

    print(f"Total problems loaded: {len(dataset['train']) + len(dataset['test'])}")
    return dataset


if __name__ == "__main__":
    dataset = read_math_dataset()
    # Convert the training split to a DataFrame
    df = dataset['train'].to_pandas()

    # Pro-tip: Check how many problems you have per subject
    print(df['type'].value_counts())
    df.to_csv('math_dataset.csv', index=False)