import glob
import random

import requests
import copy
import re, time
import os, argparse, ast, json, tqdm
from pydantic import BaseModel, Extra, root_validator
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from time import sleep
from collections import defaultdict
import numpy as np

from util import gen_from_prompt, load_model, process_args_for_models, helm_process_args
from tool_util import _generate_lm_answers, extract_json_v2, search_related_pages, search_step, get_pageviews
from wiki_autobencher import fast_compare_answers

DEFAULT_JSON_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your reasoning and language skills.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
Reply "TERMINATE" in the end when everything is done.
"""


def _generate_python_answers(problem_json, agent_lm, agent_tokenizer, agent_client, outfile_prefix='att1'):
    context = """Your goal is to generate the python code that solves the provided math questions. 
The provided math problems are in the format of json. You should write the python code for all the problems in python coding blocks. You should NOT need to write a large solve function that can solve all the problems and handle all cases, just provide problem specific solution for each question. 
Do not use string matching to figure out which problem to solve. Just do as follows, write the expression for each question inline, then print the answer. 
```python
ans1 = 1+1
print("1. ANSWER:", ans1)
ans2 = np.sin(1)
print("2. ANSWER:", ans2)
``` 
Also, make sure to answer questions **sequentially**, in the same order as question id. If there is a question that's not answerable, set the answer to be "N/A".
*You should not omit any questions, and DO NOT use ellipsis*
For questions with a numeric answer, make sure to simplify the final result and report the final answer as a decimal number. For example, for a sympy expression, you should simplify it and report the final answer as a decimal number via evalf(). But note that 'float' object has no attribute 'evalf'. 
If you are solving a trigonometric equation using sympy, make sure to use sympy.sin(x) for variable x. Do not use math.sin(x) or np.sin(x), which cause errors. Also, remember that sympy.sin takes in radians, not degrees. So you want to judge the unit and then convert degrees into radians. 
Implement error handling for each function, if there is an error (in input or in execution), output "N/A" for that question.
The python code should print the answer of each problem in the following format: 
1. ANSWER: <answer1>
2. ANSWER: <answer2>
3. ANSWER: <answer3>
...
"""
    if os.path.exists(f"{outfile_prefix}.python_answers.json"):
        print(f"{outfile_prefix}.python_answers.json exists. Skipping.")
        with open(f"{outfile_prefix}.python_answers.json", "r") as f:
            logs = f.read()
        return f'Processed Already in {outfile_prefix}.python_code.json', logs

    elif os.path.exists(f"{outfile_prefix}.python_code.json"):
        with open(f"{outfile_prefix}.python_code.json", "r") as f:
            extracted_python_code = f.read()
        exit_code, logs, image = execute_code(extracted_python_code,
                                              lang="python")
        print(logs)
        assert exit_code == 0
        return extracted_python_code, logs

    if isinstance(problem_json, list) or isinstance(problem_json, dict):
        problem_json_str = json.dumps(problem_json, indent=2)
    else:
        problem_json_str = problem_json
    context += problem_json_str + "Please write the python code for all the problems in a python coding block."
    context = DEFAULT_SYSTEM_MESSAGE + context

    request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
                                     echo_prompt=False, temperature=0.0, max_tokens=2000,
                                     process_func=None, service=agent_client,
                                     terminate_by_linebreak='no', )
    response = request_result.completions[0].text
    # parse the json file
    response = response.replace('TERMINATE', '')
    extracted_python_code = extract_code(response)
    # print(extracted_python_code, 'extracted_python_code')
    extracted_python_code = extracted_python_code[0][1]
    with open(f"{outfile_prefix}.python_code.json", "w") as f:
        print(extracted_python_code, file=f)
    exit_code, logs, image = execute_code(extracted_python_code, lang="python3.9")
    print(logs)
    if exit_code != 0:
        # error
        silver_string = "\n".join(
            [f"{idx + 1}. {line['answer']}" for idx, line in enumerate(problem_json)])
        return None, silver_string

    else:
        with open(f"{outfile_prefix}.python_answers.json", "w") as f:
            print(logs, file=f)

        return extracted_python_code, logs

def solve_with_python(question_json, outfile_prefix, agent_info):
    # solve for 20 questions. at a time.
    agent_lm, agent_tokenizer, agent_client = agent_info
    question_json_copy = copy.deepcopy(question_json)

    python_log_lst = []
    for idx in range(0, len(question_json), 20):
        question_json = question_json_copy[idx:idx+20]
        question_json_2 = []
        for line in question_json:
            line2 = {}
            line2['question'] = line['question']
            line2['id'] = line['id']
            question_json_2.append(line2)

        python_output, python_logs = _generate_python_answers(question_json_2,
                                                              agent_lm, agent_tokenizer, agent_client,
                                                              outfile_prefix = outfile_prefix + f".{idx}")
        print('python_output', python_output)
        print('python_logs', python_logs)
        # process python_logs into answers.
        ans_lst = python_logs.strip().split('\n')
        assert len(ans_lst) == len(question_json)
        for idx, ans in enumerate(ans_lst):
            answer = ans.split("ANSWER:")[1].strip()
            python_log_lst.append(answer)
    print(python_log_lst)
    assert len(python_log_lst) == len(question_json_copy)
    for line, ans in zip(question_json_copy, python_log_lst):
        line['python_answer'] = ans

    with open(outfile_prefix + '.full_python_answers.json', 'w') as f:
        json.dump(question_json_copy, f, indent=2)
    return python_log_lst, question_json_copy

def get_summary_of_results(json_dict, gold_key="answer", verbose=False):
    # a summary of the results.
    # summarize by each category.
    category2correct_count = defaultdict(list)
    category2question = defaultdict(list)
    str_summary = 'In the following, we summarize the evaluation results by each category in this agent iteration. \n We will report the accuracy for each category, and list the questions that are answered correctly and incorrectly. \n'
    for line in json_dict:
        line['category2'] = f"{line['category']} || {line['subcat']}" if 'subcat' in line else line['category'] # for the new format.
        category2correct_count[line['category2']].append(line['is_correct'])
        category2question[(line['category2'], line['is_correct'])].append(line)
    for category in category2correct_count:
        acc_temp = sum([1 if x == 'true' else 0 for x in category2correct_count[category]]) / len(category2correct_count[category])
        str_summary += f"category: {category}, accuracy: {round(acc_temp, 3)} " \
                       f"|| {sum([1 if x == 'true' else 0 for x in category2correct_count[category]])} out of {len(category2correct_count[category])}" + "\n"
        if verbose:
            str_summary += "# Questions answered correctly:\n"
            for qq in category2question[(category, 'true')]:
                str_summary += f"{qq['question']} || gold: {qq[gold_key]} || pred: {qq['test_taker_answer']}" + "\n"

                # str_summary += f"{qq['question']} || {qq['difficulty']} || gold: {qq['python_answer']} || pred: {qq['test_taker_answer']}" + "\n"
            str_summary += "# Questions answered incorrectly:\n"
            for qq in category2question[(category, 'false')]:
                str_summary += f"{qq['question']} || gold: {qq[gold_key]} || pred: {qq['test_taker_answer']}" + "\n"
            str_summary += "\n + ------------------------------------ + \n"
    # print(str_summary)
    return str_summary

def summarize_over_history(history_json_dict, gold_key="python_answer", verbose=True):
    '''
    :param history: a list of dictionaries. Each dictionary corresponds to a run.
    :return: a summary of the results.
    '''
    # augment each line of the dictionary with the iteration number.
    for idx, json_dict in enumerate(history_json_dict):
        for line in json_dict:
            line['iteration'] = idx
    # concatenate the dictionaries.
    json_dict = [line for json_dict in history_json_dict for line in json_dict]
    # a summary of the results.
    str_summary = get_summary_of_results(json_dict, gold_key=gold_key, verbose=verbose)
    # print(str_summary)
    return str_summary

def _generate_cat_with_aim(aim, agent_lm, agent_tokenizer, agent_client, history, iters, outfile_prefix='att1'):
    context = """ Your goal is to come up with comprehensive categories of math questions, and then efficiently find the subcategories within that achieve close to AIM accuracy, for each category of math questions. 
In each iteration, you should come up with a plan towards generating questions, and write the plan in a json file. 

To generate the plan, you should first come up with the set of categories to expand.
Then, for each category (e.g., arithmetic), you should expand it with a few subcategory description that's detailed, precise and concrete (e.g., two-digit addition of two terms). 
Note that you should design carefully to efficiently find the subcategories within that achieve close to AIM accuracy. For example, don't get stuck in questions that are overall too simple, and you should first broadly explore various difficulties levels, and gradually pin down the subcategories that are close to AIM accuracy.
Make sure that the subcategory you generated are concrete such that it allows some LM to generate around 50 math questions that instantiate this subcategory. 

# Output format:
Each question should be a dictionary with the following keys: id, question, category, difficulty levels (1-10).
The questions should be exactly in the following format (a list of dictionaries): 
```json
[
{"id": "1", "category": "Arithmetic", "subcategory_description": "single-digit addition of two terms", "difficulty": "1"}, 
{"id": "2", "category": "Arithmetic", "subcategory_description": "single-digit subtraction of two terms", "difficulty": "1"},
...
]
``` 
Do not use python code block. Write concrete categories and subcategories. Do not leave place holders or ellipsis.
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


The goal is to search for (category, subcategory_description) pair that achieves AIM accuracy. You do not need to answer the questions. 
For iteration 1, you can start with a variety of different categories. 
Make sure the (category, subcategory_description) that you come up with are concrete, not just place holders, and come up with 5 (category, subcategory_description) pairs.

In later iterations you should 
1. Think about breadth. Brainstorm more categories, if there are missing categories to make the evaluation more comprehensive and have broader coverage. 
2. Adjust the difficulty level of the subcategories to match the AIM accuracy. If all the subcategories tend to achieve accuracy greater than AIM, generate subcategory description with increased difficulties.
3. If all the subcategories tend to achieve lower accuracy than AIM, make the subcategory description easier. 


Note: do not come up with repetitive subcategories. 
It's helpful to first come up with a plan for this iteration, and then write the questions.

"""
    context = context.replace("AIM", aim)
    if iters is None:
        iters = len(history) + 1
    if len(history) == 0:
        context += "Please start with iteration 1."
    else:
        context += "\n".join(history) + "\nPlease start with iteration {}. Remember your goal is to find the subcategory with accuracy {} for each category.".format(iters, aim)
    context = DEFAULT_JSON_MESSAGE + context
    # extract the json file from the message
    request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
                                     echo_prompt=False, temperature=0.0, max_tokens=2000,
                                     process_func=None, service=agent_client,
                                     terminate_by_linebreak='no', )
    response = request_result.completions[0].text

    extracted_json = extract_json_v2(response, f"{outfile_prefix}.question_plan_with_aim.json")

    return extracted_json



def _generate_question_from_description(description_json, agent_lm, agent_tokenizer, agent_client, outfile_prefix='att1',
                                        questions_old=None):
    context = """Your goal is to come up with math questions that match the description. 
In each iteration, you receive as input a subcategory_description that describes the types of question to ask.  
You should come up with around 50 math questions that matches the subcategory description, and write these questions in a json file.
Each question should be a dictionary with the following keys: id, question, category, answer, difficulty levels (1-10).
You do not need to answer the questions. 

Note: do not come up with repetitive questions. If you have asked a question, do not ask it again! 
Come up with 100 concrete questions, and write them in the following format. 
Do not leave place holders or ellipsis!!!
It's helpful to first come up with a plan for this iteration, and then write the questions.
The questions should be exactly in the following format (a list of dictionaries): 
```json
[
{"id": "1", "question": "What is 5 + 3?", "category": "Arithmetic", "answer": "8", "difficulty": "1"}, 
{"id": "2", "question": "What is 5 - 3?", "category": "Arithmetic", "answer": "2", "difficulty": "1"},
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.
"""

    context = DEFAULT_JSON_MESSAGE + context + f"\nSubcategory Description:{description_json['subcategory_description']}"
    if questions_old is not None:
        old_q_string = ''
        for q in questions_old:
            old_q_string += str(q) + '\n'
        context += f"\nQuestions already generated: {old_q_string}. You need to keep generating the same amount of questions for this iteration."
    # extract the json file from the message
    request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
                                     echo_prompt=False, temperature=0.0, max_tokens=4096,
                                     process_func=None, service=agent_client,
                                     terminate_by_linebreak='no', )
    response = request_result.completions[0].text

    extracted_json = extract_json_v2(response, f"{outfile_prefix}.questions.json")
    for line in extracted_json[0]:
        line['category'] = description_json['category']
        line['subcat'] = description_json['subcategory_description']
    return extracted_json

def _ask_question_v3(agent_info, history, iters, outfile_prefix, aim_acc=None):
    agent_lm, agent_tokenizer, agent_client = agent_info
    plan_outfile = f"{outfile_prefix}.question_plan_with_aim.json"
    if not os.path.exists(plan_outfile):
        plan_json = _generate_cat_with_aim(aim_acc, agent_lm, agent_tokenizer, agent_client, history, iters=iters,
                                        outfile_prefix=outfile_prefix)

    else:
        print('FOUND THE PLAN FILE', plan_outfile)
        with open(plan_outfile, 'r') as f:
            plan_json = json.load(f)
    if len(plan_json) == 1: #remove the outer bracket.
        plan_json = plan_json[0]

    question_json_full = []
    for idx, plan_line in enumerate(plan_json):
        outfile_prefix2 = outfile_prefix + '.subcat{}'.format(idx)
        if os.path.exists(f"{outfile_prefix2}.questions.json"):
            print('found the question file', f"{outfile_prefix2}.questions.json")
            if os.path.exists(f"{outfile_prefix2}.questions_final.json"):
                print('existed already,', f"{outfile_prefix2}.questions_final.json")
                with open(f"{outfile_prefix2}.questions_final.json", 'r') as f:
                    question_json = json.load(f)
                    if len(question_json) == 1:
                        question_json = question_json[0]
                    question_json_full.extend(question_json)
                continue
            with open(f"{outfile_prefix2}.questions.json", 'r') as f:
                print('loading for more questions', f"{outfile_prefix2}.questions.json")
                question_json = json.load(f)
            if len(question_json) == 1:
                question_json = question_json[0]

        else:
            question_json = _generate_question_from_description(plan_line, agent_lm, agent_tokenizer, agent_client,
                                                                outfile_prefix2)
            if len(question_json) == 1:
                question_json = question_json[0]

        while len(question_json) < 50:
            question_json_new = _generate_question_from_description(plan_line, agent_lm, agent_tokenizer,
                                                                    agent_client, outfile_prefix2,
                                                                    questions_old=question_json)
            question_json_new = question_json_new[0]
            question_json.extend(question_json_new)
        with open(f"{outfile_prefix2}.questions_final.json", 'w') as f:
            json.dump(question_json, f)

        question_json_full.extend(question_json)

    with open(outfile_prefix + '.all_questions.json', 'w') as f:
        json.dump(question_json_full, f, indent=2)
    return question_json_full, plan_json



def test_and_eval(question_json, outfile_prefix, test_taker_info, agent_info, tool_info, gold_ans_key='answer'):

    if os.path.exists(f"{outfile_prefix}.compare_answers.json"):
        print('redundant computation, reuse the cache.')
        with open(f"{outfile_prefix}.compare_answers.json", 'r') as f:
            history_json = json.load(f)
        return history_json

    print(len(question_json), 'number of questions.')
    question_json_copy2 = copy.deepcopy(question_json)
    test_taker_output = _generate_lm_answers(question_json,
                                             test_taker_info,
                                             agent_info,
                                             outfile_prefix = outfile_prefix)
    if gold_ans_key == 'python_answer':
        question_json_copy2 = solve_with_python(question_json, outfile_prefix, agent_info)

    summary_prev_iteration, history_json = fast_compare_answers(question_json_copy2, test_taker_output,
                                                                tool_info,
                                                                outfile_prefix = outfile_prefix,
                                                                gold_ans_key=gold_ans_key)
    return history_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')

    # parser.add_argument('--model', default='gpt-3.5-turbo')  # option that takes a value
    parser.add_argument('--test_taker_modelname', default='gpt-3.5-turbo')  # option that takes a value
    parser.add_argument('--test_taker_modelname2', default=None)  # option that takes a value
    parser.add_argument('--agent_modelname', default='gpt-4-turbo-preview')  # option that takes a value
    parser.add_argument('--tool_modelname', default=None)  # option that takes a value
    parser.add_argument('--temperature', type=float, default=0.001)  # option that takes a value
    parser.add_argument('--pairwise', type=str, default='no')  # option that takes a value
    parser.add_argument('--exp_mode', type=str, default='ki_wiki')  # option that takes a value
    parser.add_argument('--theme', type=str, default='history')  # option that takes a value
    parser.add_argument('--use_helm', type=str, default='yes')  # option that takes a value
    parser.add_argument('--top_p', type=float, default=0.9)  # option that takes a value
    parser.add_argument('--acc_target', type=str, default="0.3--0.5")  # option that takes a value

    parser.add_argument('--outfile_prefix1', type=str, default='att1')  # option that takes a value

    args2 = parser.parse_args()
    args = copy.deepcopy(args2)

    if args.use_helm == 'yes':
        test_taker_info = helm_process_args(args.test_taker_modelname)
        print('loaded helm models')
    else:
        # load the test taker model.
        test_taker_lm, test_taker_tokenizer, modelpath_name, test_taker_client = process_args_for_models(
            args.test_taker_modelname)
        test_taker_info = (test_taker_lm, test_taker_tokenizer, test_taker_client)

        if args.test_taker_modelname2 is not None:
            test_taker_lm2, test_taker_tokenizer2, modelpath_name2, test_taker_client2 = process_args_for_models(
                args.test_taker_modelname2)
            test_taker_info2 = (test_taker_lm2, test_taker_tokenizer2, test_taker_client2)


    agent_lm, agent_tokenizer, agent_name, agent_client = process_args_for_models(args.agent_modelname)

    if args.tool_modelname is None:
        tool_lm, tool_tokenizer, tool_name, tool_client = agent_lm, agent_tokenizer, agent_name, agent_client
    else:
        tool_lm, tool_tokenizer, tool_name, tool_client = process_args_for_models(args.tool_modelname)

    evaluator_info = (tool_lm, tool_tokenizer, tool_client)
    agent_info = (agent_lm, agent_tokenizer, agent_client) # agent model


    if  args.exp_mode == 'autobencher':

        reuse_starting_qs = False #check_reuse()
        history = []
        history_dict = []
        historical_psg = []
        for iters in range(8):
            args.outfile_prefix = args.outfile_prefix1 + str(iters + 1)
            summarized_content = summarize_over_history(history_dict, gold_key='answer', verbose=False)
            history = [summarized_content]
            if iters == 0 and reuse_starting_qs:
                assert False, 'no need to reuse questions yet.'
            else:
                # category_gen_func = _generate_categories_targetacc_augmented
                historical_psg = _ask_question_v3(agent_info, history, iters + 1,
                                                  outfile_prefix=args.outfile_prefix,
                                                  aim_acc=args.acc_target)
                with open(f"{args.outfile_prefix}.all_questions.json", "r") as f:
                    json_category = json.load(f)
            if len(json_category) == 1:
                json_category = json_category[0]
            gold_answer_json = copy.deepcopy(json_category)
            json_dict = test_and_eval(gold_answer_json, args.outfile_prefix, test_taker_info, agent_info,
                                      evaluator_info, gold_ans_key='answer') # python_answer
            history_dict.append(json_dict)
            # compute the success rate of the qustions.
            # acc_lst = get_acc_lst(json_dict)
            verbose_description = get_summary_of_results(json_dict, verbose=False)
            print(verbose_description)


    elif args.exp_mode == 'python_solve':
        datafile = args.outfile_prefix1
        full_lst = []
        try:
            with open(datafile, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    line['test_taker_response'] = "PLACEHOLDER"
                    full_lst.append(line)
        except:
            with open(datafile, 'r') as f:
                line_lst = json.load(f)
                for line in line_lst:
                    line['test_taker_response'] = "PLACEHOLDER"
                    full_lst.append(line)
        solve_with_python(full_lst, args.outfile_prefix1, agent_info)
