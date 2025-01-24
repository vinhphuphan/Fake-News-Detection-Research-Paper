import argparse
import pandas as pd
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from collections import OrderedDict
import pandas as pd
import regex as re
import json
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from transformers import AutoTokenizer, AutoModelForCausalLM


@retry(wait=wait_random_exponential(min=8, max=50), stop=stop_after_attempt(6))
def gene(eval_prompt, model, tokenizer, max_new_tokens=100):
    tools = load_tools(['wikipedia', 'llm_math', 'wolframalpha'], llm=model)
    
    eval_prompt += "\nBelow are your outputs:"
    eval_prompt += "\nAnswers:"
    
    agent = initialize_agent(
        tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        llm=model,
        handle_parsing_errors=True
    )
    
    input_ids = tokenizer(eval_prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = agent.run(input_ids, max_new_tokens=max_new_tokens)
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return (decoded_output[len(eval_prompt):], eval_prompt)

def extract_letters(string):
    match = re.search("[a-zA-Z]", string)
    if match:
        start_index = match.start()
        result = string[start_index:]
        return result
    else:
        return ""


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--data", type=str, help="dataset name")
    parse.add_argument("--path", type=str, help="dataset path")
    parse.add_argument("--model", type=str, help="model name")
    parse.add_argument("--method", type=str, help="method name")
    parse.add_argument("--key", type=str, help="openai key")
    parse.add_argument("--test", type=bool, help="test mode or not")
    parse.add_argument(
        "--num_cons", type=int, help="sample times for self-consitency method"
    )

    args = parse.parse_args()
    return args


def make_print_to_file(path="logger/"):
    import sys
    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.path = os.path.join(path, filename)
            self.log = open(
                self.path,
                "a",
                encoding="utf8",
            )
            print("save:", os.path.join(self.path, filename))

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime("day" + "%Y_%m_%d")
    sys.stdout = Logger(fileName + ".log", path=path)
    print(fileName.center(60, "*"))


def run(ori_data, model, method, num_cons):
    result = {}

    for data in ori_data:
        _id = str(data["index"])
        prompt = data["prompt"]
        label = data["labels"]
        response = data["response"]
        ref = data["ref"]
        sumsentence = data["segmented_response"]
        dataset = data["domain"]

        if dataset == "math":
            if method == "raw":
                eval_prompt = open("./prompt/segment/math/raw.txt").read() + "\n"
            if method == "cot":
                eval_prompt = open("./prompt/segment/math/cot.txt").read() + "\n"
            if method == "link":
                eval_prompt = open("./prompt/segment/math/raw.txt").read() + "\n"
            if method == "content":
                eval_prompt = open("./prompt/segment/math/raw.txt").read() + "\n"

        if dataset == "reasoning":
            if method == "raw":
                eval_prompt = open("./prompt/segment/reasoning/raw.txt").read() + "\n"
            if method == "cot":
                eval_prompt = open("./prompt/segment/reasoning/cot.txt").read() + "\n"
            if method == "link":
                eval_prompt = open("./prompt/segment/reasoning/raw.txt").read() + "\n"
            if method == "content":
                eval_prompt = open("./prompt/segment/reasoning/raw.txt").read() + "\n"

        if dataset == "science":
            if method == "raw":
                eval_prompt = (
                    open("./prompt/segment/sci/prompt_sci_raw.txt").read() + "\n"
                )
            if method == "cot" or method == "cot_cons":
                eval_prompt = (
                    open("./prompt/segment/sci/prompt_sci_cot.txt").read() + "\n"
                )
            if method == "link":
                eval_prompt = (
                    open("./prompt/segment/sci/prompt_sci_ret_link.txt").read() + "\n"
                )
            if method == "content":
                eval_prompt = (
                    open("./prompt/segment/sci/prompt_sci_ret_content.txt").read()
                    + "\n"
                )

        if dataset == "wk" or dataset == "writing_rec":
            if method == "raw":
                eval_prompt = (
                    open("./prompt/segment/wk/prompt_wk_raw.txt").read() + "\n"
                )
            if method == "cot":
                eval_prompt = (
                    open("./prompt/segment/wk/prompt_wk_cot.txt").read() + "\n"
                )
            if method == "link":
                eval_prompt = (
                    open("./prompt/segment/wk/prompt_wk_ret_link.txt").read() + "\n"
                )
            if method == "content":
                eval_prompt = (
                    open("./prompt/segment/wk/prompt_wk_ret_content.txt").read() + "\n"
                )

        eval_prompt += "\nQuestion: "
        eval_prompt += prompt
        # eval_prompt+="\n\nAnswer: "
        eval_prompt += "\nSegments: "
        for j in range(len(sumsentence)):
            no_number = extract_letters(sumsentence[j])
            b = str(j + 1) + ". " + no_number + "\n"
            eval_prompt += b
        a = ""
        if method == "link":
            eval_prompt += "\nRerefence links: "
            for ref_out in data["ref"]:
                a += ref_out + "\n"
        if method == "content":
            eval_prompt += "\nRerefence doc: "
            for ref_out in data["source"]:
                a += ref_out + "\n"

        if method == "cot":
            raw_generates = []
            for _ in range(num_cons):
                raw_generate, prefix = gene(eval_prompt, model=model, method=method)
                raw_generates.append(raw_generate)

        else:
            raw_generate, prefix = gene(eval_prompt, model=model, method=method)

        if method == "cot":
            print(_id, raw_generates, label)
            ress = []
            for i in range(len(raw_generates)):
                generate = raw_generates[i]
                gen = [1 for x in range(len(sumsentence))]
                if "ALL_CORRECT" not in generate:
                    an = "".join(re.findall(r"(?<=Answer: )[\s\S]*", generate))
                    generate = [
                        int(x)
                        for x in re.findall(r"\d+", an)
                        if int(x) <= len(sumsentence)
                    ]
                    gen = [1 for x in range(len(sumsentence))]
                    for _ in generate:
                        gen[_ - 1] = 0
                res = []
                for i in range(len(sumsentence)):
                    if (gen[i] == 1) and label[i]:
                        print("TP")
                        res.append("TP")
                    elif (gen[i] == 0) and not label[i]:
                        print("TN")
                        res.append("TN")
                    elif (gen[i] == 1) and not label[i]:
                        print("FP")
                        res.append("FP")
                    elif (gen[i] == 0) and label[i]:
                        print("FN")
                        res.append("FN")
                ress.append(res)
            final_res = [[] for _ in range(len(sumsentence))]
            res = []
            for i in range(len(sumsentence)):
                for j in range(len(ress)):
                    final_res[i].append(ress[j][i])
            for i in range(len(sumsentence)):
                res.append(max(final_res[i], key=final_res[i].count))

        else:

            print(_id, raw_generate, label)
            generate = raw_generate
            gen = [1 for x in range(len(sumsentence))]

            # if 'ALL_CORRECT' not in generate:
            if "ALL\_CORRECT" not in generate:
                # generate=''.join(re.findall(r'(?<=Answer: )[\s\S]*',generate))
                generate = [
                    int(x)
                    for x in re.findall(r"\d+", generate)
                    if int(x) <= len(sumsentence)
                ]
                gen = [1 for x in range(len(sumsentence))]
                for _ in generate:
                    gen[_ - 1] = 0

            res = []
            for i in range(len(sumsentence)):
                if (gen[i] == 1) and label[i]:
                    print("TP")
                    res.append("TP")
                elif (gen[i] == 0) and not label[i]:
                    print("TN")
                    res.append("TN")
                elif (gen[i] == 1) and not label[i]:
                    print("FP")
                    res.append("FP")
                elif (gen[i] == 0) and label[i]:
                    print("FN")
                    res.append("FN")
        # pdb.set_trace()
        result[_id] = {
            "id": _id,
            "domain": dataset,
            "pred": gen,
            "raw": raw_generate,
            "prompt": prefix,
            "res": res,
        }
    return result


def compute_accuracy(domain, res):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for k, v in res.items():
        if v["domain"] == domain or domain == "ALL":
            for r in v["res"]:
                if r == "TP":
                    TP += 1
                elif r == "TN":
                    TN += 1
                elif r == "FP":
                    FP += 1
                elif r == "FN":
                    FN += 1

    return {
        "class 1": TN / (TN + FP) if TN + FP != 0 else None,
        "class 0": TP / (TP + FN) if TP + FN != 0 else None,
        "true num": TP + FN,
        "false num": TN + FP,
        "balanced": (
            0.5 * (TN / (TN + FP) + TP / (TP + FN))
            if TN + FP != 0 and TP + FN != 0
            else None
        ),
        "TN,TP,FN,FP": (TN, TP, FN, FP),
        "P": TP / (2 * (TP + FP)) if TP + FP != 0 else None,
        "R": TP / (TP + FN) if TP + FN != 0 else None,
        "F1": (
            (TP / (TP + FP)) * (TP / (TP + FN)) / ((TP / (TP + FP)) + (TP / (TP + FN)))
            if (TP + FP != 0 and TP + FN != 0)
            else None
        )
    }


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


def print_saveresult(data, result, method, model):
    print("ALL" + str(compute_accuracy("ALL", result)))
    print("wk" + str(compute_accuracy("wk", result)))
    print("sci" + str(compute_accuracy("science", result)))
    print("math" + str(compute_accuracy("math", result)))
    print("reasoning" + str(compute_accuracy("reasoning", result)))
    print("writing_rec" + str(compute_accuracy("writing_rec", result)))
    time_ = time.strftime("%m-%d-%H-%M-%S", time.localtime(time.time()))
    output = ("/content/drive/MyDrive/COMP8851/eval/res/"+ time_+ "_"+ str(method)+ "_"+ str(model)+ ".csv")

    save_exp(data, result, output)


if __name__ =='__main__':
    time_ = time.strftime("%m-%d-%H-%M-%S", time.localtime(time.time()))
    make_print_to_file(path='/content/drive/MyDrive/COMP8851/eval/res')

    args = parse_args()

    model_name = args.model
    if model_name == 'vicuna':
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        model_ = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", device_map="auto")
    elif model_name == 'llama':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model_ = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
    elif model_name == 'mistralai':
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        model_ = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
    elif model_name == 'llama-3':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        model_ = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map="auto")
    elif model_name == 'fast-chat':
        tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
        model_ = AutoModelForCausalLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0", device_map="auto")
    else:
        raise ValueError("Invalid model name. Please provide 'vicuna', 'llama', or 'mistralai'.")

    res = set()
    path = args.path

    with open(path, 'r') as json_file:
        data = json.load(json_file)

    result = run(ori_data=data, model=model_name, method=args.method, num_cons=1)
    print_saveresult(data, result, args.method, model_name)