import pandas as pd
from collections import OrderedDict

def save_exp(ori_data, result, output):
    """
    Saves evaluation results and data to a CSV file.

    Args:
        ori_data: List of dictionaries containing original data points.
        result: Dictionary containing evaluation results for each data point.
        output: Path to the output CSV file.
    """

    print(f'Saving results to {output}')

    # Define headers for the CSV file
    init = (('id', []), ('dataset', []), ('qst', []), ('response', []), ('label', []),
            ('type', []), ('comment', []), ('ref', []), ('prompt', []), ('gen', []), ('res', []))
    save = OrderedDict(init)

    # Iterate through each data point
    for data in ori_data:
        id_ = str(data['index'])
        dataset = data['domain']
        qst = data['prompt']
        ans = data['response']
        label = data['labels']
        tp = str(data['type'])
        comment = str(data['comment'])
        ref = str(data['ref'])

        # Retrieve evaluation results
        prompt = result[id_]['prompt']
        gen = result[id_]['raw']
        res = result[id_]['res']

        # Append data to OrderedDict
        save['id'].append(id_)
        save['dataset'].append(dataset)
        save['qst'].append(qst)
        save['response'].append(ans)
        save['label'].append(label)
        save['type'].append(tp)
        save['comment'].append(comment)
        save['ref'].append(ref)
        save['prompt'].append(prompt)
        save['gen'].append(gen)
        save['res'].append(str(res))

    # Create and save DataFrame
    df = pd.DataFrame(data=save)
    df.to_csv(output, index=False)  # index=False prevents saving the row index

# Sample ori_data (you already provided this)
ori_data = [
    {
        "index": 0,
        "prompt": "Show me the answer for the following math question: 143287*534156=?",
        "response": "To find the product of 143287 and 534156, you can use the standard long multiplication method as follows:\\n\\nmarkdown\\nCopy code\\n143287*534156\\n---------\\n715740   (6 x143287)\\n+4294440   (5 x 143287)\\n+11448176   (4 x 143287)\\n---------\\n76529372\\nTherefore, 143287 multiplied by 534156 equals 76529372.",
        "segmented_response": [
            "To find the product of 143287 and 534156, you can use the standard long multiplication method as follows:\n\n\n\n\n143287*534156\n---------\n715740   (6 x143287)\n+4294440   (5 x 143287)\n+11448176   (4 x 143287)\n---------\n76529372\n\n\n\n\nTherefore, 143287 multiplied by 534156 equals 76529372."
        ],
        "labels": [
            False
        ],
        "comment": [
            "76537610772"
        ],
        "type": [
            ""
        ],
        "ref": [
            ""
        ],
        "source": "authors",
        "domain": "math"
    }
]

# Sample result (replace with your actual results)
result = {
    '0': {'id': '0', 'domain': 'math', 'pred': [1], 'raw': 'Answer: ALL_CORRECT', 'prompt': [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'I will show you a question and a list of text segments. All the segments can be concatenated to form a complete answer to the question. Your task is to assess whether each text segment contains errors or not. \nPlease generate using the following format:\nAnswer: List the ids of the segments with errors (separated by commas). Please only output the ids, no more details. If all the segments are correct, output "ALL_CORRECT".\n\nHere is one example:\nQuestion: 8923164*7236571?\nSegments: \n1. The product of 8923164 and 7236571 is: 6,461,216,222,844\n2. So, 8923164 multiplied by 7236571 is equal to 6,461,216,222,844.\n\nBelow are your outputs:\nAnswer: 1,2\nIt means segment 1,2 contain errors.\n\nBelow are my inputs:\n\nQuestion: Show me the answer for the following math question: 143287*534156=?\nSegments: 1. To find the product of 143287 and 534156, you can use the standard long multiplication method as follows:\n\n\n\n\n143287*534156\n---------\n715740   (6 x143287)\n+4294440   (5 x 143287)\n+11448176   (4 x 143287)\n---------\n76529372\n\n\n\n\nTherefore, 143287 multiplied by 534156 equals 76529372.\n'}], 'res': ['FP']}
}

# Output file path
output_path = "evaluation_results.csv" 

# Call the save_exp function
save_exp(ori_data, result, output_path)
