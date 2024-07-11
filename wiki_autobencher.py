import re, time, os, argparse, ast, json, tqdm
import glob
import random
import copy
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from time import sleep
from collections import defaultdict
import numpy as np
from util import gen_from_prompt, load_model, process_args_for_models, helm_process_args
from tool_util import _generate_lm_answers, extract_json_v2, search_related_pages, search_step, get_pageviews

DEFAULT_JSON_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your reasoning and language skills.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
Reply "TERMINATE" in the end when everything is done.
"""



def get_summary_of_results(json_dict, gold_key="python_answer", verbose=False):
    # a summary of the results.
    # summarize by each category.
    category2correct_count = defaultdict(list)
    category2question = defaultdict(list)
    str_summary = 'In the following, we summarize the evaluation results by each category in this agent iteration. \n We will report the accuracy for each category, and list the questions that are answered correctly and incorrectly. \n'
    for line in json_dict:
        line['category2'] = f"{line['category']} || {line['wiki_entity']} [{line['additional_requirement']}]" if 'additional_requirement' in line else line['category']
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


def get_acc_lst(json_dict, gold_key="python_answer"):
    # a summary of the results.
    # summarize by each category.
    category2correct_count = defaultdict(list)
    for line in json_dict:
        category2correct_count[line['category']].append(line['is_correct'])
    acc_lst = []
    for category in category2correct_count:
        acc = sum([1 if x == 'true' else 0 for x in category2correct_count[category]]) / len(category2correct_count[category])
        acc_lst.append(acc)
    return acc_lst



def solve_and_compare_questions(test_taker_info, agent_info, question_json, gold_answer, outfile_prefix, gold_ans_key='gold_answer'):
    test_taker_output = _generate_lm_answers(question_json,
                         test_taker_info,
                         agent_info,
                         outfile_prefix=outfile_prefix)
    summary_prev_iteration, history_json = fast_compare_answers(gold_answer, test_taker_output,
                                                                agent_info, outfile_prefix=outfile_prefix,
                                                                gold_ans_key=gold_ans_key)

    return history_json


def fast_compare_answers(gold_output, test_taker_output, agent_model_info, outfile_prefix='att1', gold_ans_key='gold_answer'):
    if os.path.exists(f"{outfile_prefix}.compare_answers.json"):
        print('FOUND compare_answers.json')
        json_dict = json.load(open(f"{outfile_prefix}.compare_answers.json", "r"))
        str_summary = get_summary_of_results(json_dict, gold_key="gold_answer")
        return str_summary, json_dict

    print("Comparing the answers generated by the python code and the test taker...")
    agent_lm, agent_tokenizer, agent_client = agent_model_info
    print(len(gold_output), len(test_taker_output))
    assert len(gold_output) == len(test_taker_output)
    context_str = """Your goal is to compare the prediction with the gold answer, and judge the correctness of the prediction. 
We'd still consider the prediction to be correct if 
1. the prediction is semantically the same as the gold answer: formating or different way of reference shouldn't affect correctness. For example, if the gold answer is Jan 21, and the test taker output is 01/21, we would still consider the prediction to be correct. For example, United States and USA refer to the same entity.
2. the prediction refers a broader entity that contains the gold answer. For example, if the gold answer is Beijing, and the test taker output is Asia, we will then consider correctness based on the question.
3. If the question is slightly ambiguous, such that there are multiple correct answers: For example, if the question asks for reasons why something happens, and it could be caused by multiple reasons, we will consider the prediction to be correct if the prediction contains one of the correct answers.

You should output a short and succinct reasoning for the your correctness prediction. Then, you should output delimiter "##" and output "true" if the prediction is correct, and "false" if the prediction is incorrect.
Example Format: 
Question: What is 1+1?
pred=2 || gold=2.0
reason: identical numbers ## true
"""
    out_handle = open(f"{outfile_prefix}.compare_answers.jsonl", 'w')
    final_lst = []
    correct_count2 = 0
    for idx, (line_gold, line_pred) in tqdm.tqdm(enumerate(zip(gold_output, test_taker_output))):
        # print(line_gold, line_pred)
        line = {'id': str(idx + 1), 'question': line_gold['question'], 'gold_answer': line_gold[gold_ans_key],
                "test_taker_answer": line_pred['test_taker_response']}
        # add other fields in line_gold to line.
        for k, v in line_gold.items():
            if k not in line:
                line[k] = v
        pred = line_pred['test_taker_response'].strip()
        gold = line_gold[gold_ans_key].strip()
        q_str = f"Question {idx+1}: {line_gold['question']}\npred={pred} || gold={gold}\nreason:"
        context = context_str + q_str
        request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
                                         echo_prompt=False, temperature=0.0, max_tokens=3000,
                                         process_func=None, service=agent_client,
                                         terminate_by_linebreak='no', verbose=False)
        response = request_result.completions[0].text
        line['reasons'] = response.strip()
        line['is_correct'] = response.strip().split('##')[-1].strip()
        test_taker_line = test_taker_output[idx]
        line['question'] = test_taker_line['question']
        if 'category' in test_taker_line:
            line['category'] = test_taker_line['category']
        else:
            line['category'] = 'None'
        if 'difficulty' in test_taker_line:
            line['difficulty'] = test_taker_line['difficulty']
        if line["is_correct"] == 'true':
            correct_count2 += 1

        print(json.dumps(line), file=out_handle)

        final_lst.append(line)
    json_dict = final_lst
    accuracy = correct_count2 / len(json_dict)
    print("accuracy: ", accuracy)
    assert len(json_dict) == len(test_taker_output)
    out_handle.close()


    with open(f"{outfile_prefix}.compare_answers.json", 'w') as out_handle:
        json.dump(json_dict, out_handle, indent=2)

    str_summary = get_summary_of_results(json_dict, gold_key="gold_answer")
    return str_summary, json_dict

def gen_qa_without_docs(topic, agent_info, additional_req):
    context = """You will generate a few question and answer pairs on the topic: {{TOPIC}}
Make sure not to ask subjective questions, and let the question's correct answer be a concise short phrase. 
Make the answer concise. 

You will also receive additional requirements on the questions. You should follow these additional requirements when generating the questions.
For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.

Try to generate a diverse set of 50 questions, and make sure that the questions are not too similar to each other while satisfying the additional requirements. If you can't generate 50 questions, generate as many as you can.

Formatting: 
Each question should be a dictionary with the following keys: id, question, answer, estimated difficulty. 
The questions should be exactly in the following format (a list of dictionaries): 
```json
{"id": "1", "question": "<question>", "answer": "<answer>", "difficulty": "1"}, 
{"id": "2", "question": "<question>", "answer": "<answer>", "difficulty": "1"}, 
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets. 
If you are generating double quotes as content of <question> or <answer>, make sure to escape them with a backslash. 
    """
    context.replace("{{TOPIC}}", topic)
    agent_lm, agent_tokenizer, agent_client = agent_info

    context += f"Topic: {topic}\nAdditional requirements: {additional_req}\n"
    # extract the json file from the message
    request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
                                     echo_prompt=False, temperature=0.0, max_tokens=4096,
                                     process_func=None, service=agent_client,
                                     terminate_by_linebreak='no', verbose=False)
    response = request_result.completions[0].text

    extracted_json = extract_json_v2(response, None)
    extracted_json = extracted_json[0]
    return extracted_json

def gen_qa_pairs_augmented(paragraph, agent_info, additional_req):
    context = """Conditioned on the wikipedia paragraph, you will generate a few question and answer pairs. 
Make sure not to ask subjective questions, and let the question's correct answer be a concise short phrase. 
Make sure that the question you selected is answerable by the given wikipedia paragraph, and make the answer concise. It's recommended to use the exact text from the paragraph as answers.
Make sure that the questions are also answerable by an expert **without the wikipedia paragraph**. For example, dont ask questions that are too specific to the paragraph, like "what are the three locations mentioned in the paragraph?". Or "who's the most famous soldier, according to the paragraph?".

You will also receive additional requirements on the questions. You should follow these additional requirements when generating the questions.
For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.

Try to generate a diverse set of 15 questions, and make sure that the questions are not too similar to each other while satisfying the additional requirements. If you can't generate 15 questions, generate as many as you can.

Formatting: 
Each question should be a dictionary with the following keys: id, question, answer, estimated difficulty. 
The questions should be exactly in the following format (a list of dictionaries): 
```json
{"id": "1", "question": "<question>", "answer": "<answer>", "difficulty": "1"}, 
{"id": "2", "question": "<question>", "answer": "<answer>", "difficulty": "1"}, 
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets. 
If you are generating double quotes as content of <question> or <answer>, make sure to escape them with a backslash. 
"""
    agent_lm, agent_tokenizer, agent_client = agent_info

    context += f"Wiki paragraph: {paragraph}\nAdditional requirements: {additional_req}\n"
    # extract the json file from the message
    request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
                                     echo_prompt=False, temperature=0.0, max_tokens=2000,
                                     process_func=None, service=agent_client,
                                     terminate_by_linebreak='no', verbose=False)
    response = request_result.completions[0].text


    extracted_json = extract_json_v2(response, None)
    extracted_json = extracted_json[0]
    return extracted_json

def _refine_categories_random_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
    category_json = _generate_categories_random_augmented(theme, agent_info, history, iters, outfile_prefix=outfile_prefix+'.brainstorm', acc_target=acc_target)
    # given the json_lst, refine the categories to achieve the target accuracy.
    full_cat_lst = []
    for line in category_json:
        cat_lst = search_related_pages(line['category'])
        full_cat_lst.extend(cat_lst)
    context = """ Your goal is to select from a list of categories for knowledge intensive questions so that the selected subset are not repetitive from prior selectioins and covers a wide range of topics that are important.
The categories should be selected based on three criteria: (1) aligned with THEME, (2) salient and cover important topics.
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.

Output Formatting: 
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement. 
Make sure the categories are similar to wikipedia categories. 
The categories should be exactly in the following format (a list of dictionaries): 
```json 
[
{"id": "1", "category": "Ancient Philosophers", "parent_category": "History", "additional_requirement": "only ask about famous people and their ideologies"}, 
{"id": "2", "category": "Second World War", "parent_category": "History", "additional_requirement": "major battles"}, 
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration: 
The goal is to find a set of categories that have broad coverage of topics and are not repetitive from prior selections. 

At every iteration, you are given a list of categories that you have already explored and their respective accuracy. Also, you are given a larger set of candidate categories for this iteration, and you should use the information from previous iterations to select the top 10 categories from the list. 
DO NOT REPEAT any of the categories that you have already explored.
"""
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _refine_categories(theme, context, agent_info, history, iters, full_cat_lst, outfile_prefix=outfile_prefix + '.refine')

def _refine_categories_targetacc_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
    category_json = _generate_categories_targetacc_augmented(theme, agent_info, history, iters, outfile_prefix=outfile_prefix+'.brainstorm', acc_target=acc_target)
    # given the json_lst, refine the categories to achieve the target accuracy.
    full_cat_lst = []
    for line in category_json:
        cat_lst = search_related_pages(line['category'])
        full_cat_lst.extend(cat_lst)
    context = """ Your goal is to select from a list of categories for knowledge intensive questions so that the selected subset are likely to achieve the target accuracy of {ACC_TARGET}.
The categories should be selected based on three criteria: (1) aligned with THEME, (2) likely to obtain the target accuracy of {ACC_TARGET}, you can judge this based on the accuracy statistics from previous iterations. and (3) salient and cover important topics.
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.

Output Formatting: 
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement. 
Make sure the categories are similar to wikipedia categories. 
The categories should be exactly in the following format (a list of dictionaries): 
```json 
[
{"id": "1", "category": "Ancient Philosophers", "parent_category": "History", "additional_requirement": "only ask about famous people and their ideologies"}, 
{"id": "2", "category": "Second World War", "parent_category": "History", "additional_requirement": "major battles"}, 
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration: 
The goal is to find a set of categories that with accuracy close to the target accuracy level of {ACC_TARGET}. 

At every iteration, you are given a list of categories that you have already explored and their respective accuracy. Also, you are given a larger set of candidate categories for this iteration, and you should use the information from previous iterations to select the top 10 categories from the list, that are most likely to achieve the target accuracy level, while still being relevant and salient. 
In later iterations you should receive as input the categories that you have already explored and their respective accuracy. You should
DO NOT REPEAT any of the categories that you have already explored.
"""
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _refine_categories(theme, context, agent_info, history, iters, full_cat_lst, outfile_prefix=outfile_prefix + '.refine')




def _generate_categories_targetacc_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
    context = """ Your goal is to come up with a list of categories for knowledge intensive questions that achieve the target accuracy of {ACC_TARGET}.
The categories should be diverse and cover important topics, under the theme of THEME. 
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.
Constructing the categories is like building a tree structure of history, and (category, parent_category) is like specifying a node and its parent. We should select the most precise parent category, for example if you are trying to expand the category "second world war" to make it more specific by adding the node "famous battles in second world war", you should specify the parent category as "second world war" instead of "history".

Output Formatting: 
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement. 
Make sure the categories are similar to wikipedia categories. 
The categories should be exactly in the following format (a list of dictionaries): 
```json 
[
{"id": "1", "category": "Ancient Philosophers", "parent_category": "History", "additional_requirement": "only ask about famous people and their ideologies"}, 
{"id": "2", "category": "Second World War", "parent_category": "History", "additional_requirement": "major battles"}, 
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration: 
The goal is to find a set of categories that with accuracy close to the target accuracy level of {ACC_TARGET}. 

For iteration 1, you can start with a wide variety of categories for us to build upon later. 
In later iterations you should receive as input the categories that you have already explored and their respective accuracy. You should
1. Think about breadth. Brainstorm questions with different categories to have broader coverage. Coming up with new categories that can are likely to achieve the target accuracy level.
2. For example, If you find the model now lacks categories of 0.3 -- 0.5 accuracy, you should come up with more categories that would yield accuracy in that range, by either reducing the difficulty of questions that achieve lower accuracy (via subcategory or via additional requirement), or increasing the difficulty of questions that achieve higher accuracy.
3. DO NOT REPEAT any of the categories that you have already explored.
"""
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _generate_categories(theme, context, agent_info, history, iters, outfile_prefix=outfile_prefix)

def _generate_categories_random_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
    context = """ Your goal is to come up with a list of categories for knowledge intensive questions that have broad coverage and are salient. 
The categories should be diverse and cover important topics, under the theme of THEME. 
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.
Constructing the categories is like building a tree structure of history, and (category, parent_category) is like specifying a node and its parent. We should select the most precise parent category, for example if you are trying to expand the category "second world war" to make it more specific by adding the node "famous battles in second world war", you should specify the parent category as "second world war" instead of "history".

Output Formatting: 
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement. 
Make sure the categories are similar to wikipedia categories. 
The categories should be exactly in the following format (a list of dictionaries): 
```json 
[
{"id": "1", "category": "Ancient Philosophers", "parent_category": "History", "additional_requirement": "only ask about famous people and their ideologies"}, 
{"id": "2", "category": "Second World War", "parent_category": "History", "additional_requirement": "major battles"}, 
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration: 
The goal is to find a set of categories that have broad coverage of topics and are salient. 

For iteration 1, you can start with a wide variety of categories for us to build upon later. 
In later iterations you should receive as input the categories that you have already explored and their respective accuracy. You should
1. Think about breadth. Brainstorm questions with different categories to have broader coverage.
2. DO NOT REPEAT any of the categories that you have already explored.
"""
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _generate_categories(theme, context, agent_info, history, iters, outfile_prefix=outfile_prefix)

def _refine_categories(theme, context, agent_info, history, iters, candidate_lst, outfile_prefix='att1'):
    if os.path.exists(f"{outfile_prefix}.categories.json"):
        print("FOUND categories.json")
        return json.load(open(f"{outfile_prefix}.categories.json", "r"))[0]
    agent_lm, agent_tokenizer, agent_client = agent_info
    context = context.replace("THEME", theme)
    if iters is None:
        iters = len(history) + 1
    if iters == 1:
        context += "Please start with iteration 1." + "Here are the category candidates to select from (delimited by ||): " + " || ".join(candidate_lst) + "\n"
    else:
        context += "\n".join(history) + "Please start with iteration {}.".format(iters) + "Here are the category candidates to select from (delimited by ||): " + "||".join(candidate_lst) + "\n"
    context = DEFAULT_JSON_MESSAGE + context
    # extract the json file from the message
    request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
                                     echo_prompt=False, temperature=0.0, max_tokens=2000,
                                     process_func=None, service=agent_client,
                                     terminate_by_linebreak='no',)
    response = request_result.completions[0].text

    with open(f"{outfile_prefix}.full_thoughts.txt", 'w', encoding='utf-8') as out_handle:
        out_handle.write(context)
        out_handle.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        out_handle.write(response)

    extracted_json = extract_json_v2(response, f"{outfile_prefix}.categories.json")
    if len(extracted_json) == 1:
        extracted_json = extracted_json[0]
    return extracted_json

def _generate_categories(theme, context, agent_info, history, iters, outfile_prefix='att1'):
    if os.path.exists(f"{outfile_prefix}.categories.json"):
        print("FOUND categories.json")
        return json.load(open(f"{outfile_prefix}.categories.json", "r"))[0]
    agent_lm, agent_tokenizer, agent_client = agent_info
    context = context.replace("THEME", theme)
    if iters is None:
        iters = len(history) + 1
    if iters == 1:
        context += "Please start with iteration 1."
    else:
        context += "\n".join(history) + "Please start with iteration {}.".format(iters)
    context = DEFAULT_JSON_MESSAGE + context
    # extract the json file from the message
    request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
                                     echo_prompt=False, temperature=0.0, max_tokens=2000,
                                     process_func=None, service=agent_client,
                                     terminate_by_linebreak='no', )
    response = request_result.completions[0].text



    with open(f"{outfile_prefix}.full_thoughts.txt", 'w', encoding='utf-8') as out_handle:
        out_handle.write(context)
        out_handle.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        out_handle.write(response)

    extracted_json = extract_json_v2(response, f"{outfile_prefix}.categories.json")
    if len(extracted_json) == 1:
        extracted_json = extracted_json[0]
    return extracted_json




def _ask_question(theme, agent_info, history, iters, outfile_prefix='att1'):
    if os.path.exists(f"{outfile_prefix}.questions.json"):
        print("FOUND questions.json")
        return json.load(open(f"{outfile_prefix}.questions.json", "r"))
    agent_lm, agent_tokenizer, agent_client = agent_info
    context = """
Your goal is to comprehensively evaluate the knowledge of a language model. 
In each iteration, you should output 30 ** knowledge-intensive ** questions of different categories, and write these questions in a json file. All the question need to pertain to the them of THEME.

To generate a question, you will follow the following steps: 
1. Come up with a category (e.g. physical phenomenon, capitals of countries, etc.)
2. This category will be used to search for a wikipedia page.
3. Conditioned on the wikipedia paragraph, you will generate a question and answer pair. Make sure that the question you selected is answerable by the given wikipedia paragraph.

Output formatting: 
Each question should be a dictionary with the following keys: id, question, answer, category, estimated difficulty.
Note: do not come up with repetitive questions. If you have asked a question, do not ask it again! 
Come up with 30 concrete questions, and write them in the following format. It's helpful to first come up with a plan for this iteration, and then write the questions.
The questions should be exactly in the following format (a list of dictionaries): 
```json
[
{"id": "1", "question": "What's the tallest mountain in the world?", "answer": "Mount Everest", "category": "Geography", "difficulty": "1"}, 
{"id": "2", "question": "Who discovered green tea", "answer":"Shen Nong", "category": "History", "difficulty": "1"},
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.

Iteration: 
The goal is to search for a category of questions that the language model is weak at. 
For iteration 1, you can start with questions of different categories, and start with a difficulty level of 1-2. Make sure the questions that you come up with are concrete questions that has a concrete solution, not just place holders, and come up with 30 questions. Do not leave place holders.  
In later iterations you should 
1. Think about breadth. Brainstorm questions with different categories if there are missing categories to make the evaluation more comprehensive and have broad coverage. 
2. For the categories that the model is strong at, increase the difficulty level of the questions. 
3. For the categories that the model is weak at, try to probe for diverse types of failure modes. Remember the goal is to get a comprehensive evaluation of the model. We want to know all the failure modes of the model, and all its strength.  
"""
    context = context.replace("THEME", theme)
    if iters is None:
        iters = len(history) + 1
    if iters == 1:
        context += "Please start with iteration 1."
    else:
        context += "\n".join(history) + "Please start with iteration {}. Remember: Always output 30 questions, DO NOT just terminate directly".format(iters)
    context = DEFAULT_JSON_MESSAGE + context
    # extract the json file from the message
    request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
                                     echo_prompt=False, temperature=0.0, max_tokens=2000,
                                     process_func=None, service=agent_client,
                                     terminate_by_linebreak='no', )
    response = request_result.completions[0].text

    extracted_json = extract_json_v2(response, f"{outfile_prefix}.questions.json")
    return extracted_json


def generate_dataset_without_docs(line_, agent_info, outfile_prefix,
                            total_num_questions=50):
    if os.path.exists(f"{outfile_prefix}.KI_questions.json"):
        print("found ", f"{outfile_prefix}.KI_questions.json")
        full_lst = []
        with open(f"{outfile_prefix}.KI_questions.json", "r") as f:
            for line in f:
                line = json.loads(line)
                full_lst.append(line)
        return full_lst

    f = open(f"{outfile_prefix}.KI_questions.json", "w")

    full_lst = []

    try:
        json_questions = gen_qa_without_docs(line_['category'], agent_info, line_['additional_requirement'])
    except Exception as e:
        print(e)
        print("error in generating more questions, skipping...")
        print(f'generated {len(ful_lst)} questions')

    for json_question in json_questions:
        line = copy.deepcopy(line_)
        line['question'] = json_question['question']
        line['gold_answer'] = json_question['answer']
        line['difficulty'] = json_question['difficulty']
        line['wiki_entity'] = 'None'
        full_lst.append(line)
        print(json.dumps(line), file=f)
    f.close()
    return full_lst

def generate_long_questions(line_, agent_info, outfile_prefix, generate_qa_func=gen_qa_pairs_augmented,
                            total_num_questions=50):
    if os.path.exists(f"{outfile_prefix}.KI_questions.json"):
        print("found ", f"{outfile_prefix}.KI_questions.json")
        full_lst = []
        with open(f"{outfile_prefix}.KI_questions.json", "r") as f:
            for line in f:
                line = json.loads(line)
                full_lst.append(line)
        return full_lst

    f = open(f"{outfile_prefix}.KI_questions.json", "w")
    paragraph, wiki_entity = search_step(line_['category'], output_more=True)
    print(len(paragraph), 'length of paragraph')
    if len(paragraph) == 0:
        print("empty paragraph, skipping...")
        return {}

    full_lst = []
    for start_idx in range(0, len(paragraph), 20):
        if start_idx > total_num_questions: break
        end_idx = start_idx + 20 if start_idx + 20 < len(paragraph) else len(paragraph)
        try:
            json_questions = generate_qa_func(paragraph[start_idx:end_idx], agent_info, line_['additional_requirement'])
            # json_questions = generate_qa_func(paragraph[start_idx:end_idx], agent_info, line_['additional_requirement'])
        except:

            print("error in generating more questions, skipping...")
            print(f'generated {len(ful_lst)} questions')
            continue  # skip the empty paragraph.

        for json_question in json_questions:
            line = copy.deepcopy(line_)
            line['question'] = json_question['question']
            line['gold_answer'] = json_question['answer']
            line['difficulty'] = json_question['difficulty']
            line['wiki_entity'] = wiki_entity
            full_lst.append(line)
            print(json.dumps(line), file=f)
    f.close()
    return full_lst


def saliency_rerank(json_lst, num_keep = 5 ):
    for line_ in json_lst:
        page_title = line_['category'].replace(' ', '_')
        pageviews = get_pageviews(page_title)
        line_['salience'] = pageviews  if pageviews is not None else 0 # add the pageviews to the line.
    # sort by the saliency
    json_lst = sorted(json_lst, key=lambda x: x['salience'], reverse=True)
    for line in json_lst:
        print(f'salience of {line["category"]}: ', round(line['salience'] / 1000000, 2),  'M')
    return json_lst[:num_keep]



def generate_full_qa(theme, agent_info, history, iters, outfile_prefix='att1',
                                   historical_psg=None,
                                   category_gen_func=_refine_categories_targetacc_augmented,
                                   generate_qa_func=generate_long_questions, acc_target=None,
                                   apply_saliency_rerank=True):
    if os.path.exists(f"{outfile_prefix}.KI_questions.json"):
        print("FOUND KI_questions.json")
        return

    if acc_target is not None:
        json_category = category_gen_func(theme, agent_info, history, iters, outfile_prefix=outfile_prefix,
                                          acc_target=acc_target)
    else:
        json_category = category_gen_func(theme, agent_info, history, iters, outfile_prefix=outfile_prefix)
    if apply_saliency_rerank:
        json_category = saliency_rerank(json_category, 5)
    full_lst = []
    historical_psg = []
    for line_ in json_category:
        paragraph, wiki_entity = search_step(line_['category'])
        if wiki_entity in historical_psg:
            print('found repetitive wiki entity, skipping...', wiki_entity)
            continue
        if len(paragraph) == 0:
            print("empty paragraph, skipping...")
            continue  # skip the empty paragraph.
        historical_psg.append(wiki_entity)
        if 'additional_requirement' not in line_: continue # skip the empty paragraph.
        page_title = line_['category'].replace(' ', '_')
        pageviews = get_pageviews(page_title)
        line_['salience'] = pageviews if pageviews is not None else 0 # add the pageviews to the line.
        print(f'salience of {page_title}: ', round(line_['salience'] / 1000000, 2), 'M')
        try:
            json_questions = generate_qa_func(line_, agent_info, outfile_prefix+f'__{page_title}')
        except Exception as e:
            print(e)
            print("error in generating questions, skipping...")
            continue # skip the empty paragraph.

        for line in json_questions:
            full_lst.append(line)
        line_['paragraph'] = paragraph
        line_['wiki_entity'] = wiki_entity

    with open(f"{outfile_prefix}.KI_questions.json", "w") as f:
        json.dump(full_lst, f)

    with open(f"{outfile_prefix}.categories_augmented.json", "w") as f:
        json.dump(json_category, f)
    return historical_psg




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
    parser.add_argument('--acc_target', type=str, default="0.1--0.3")  # option that takes a value
    parser.add_argument('--num_iters', type=int, default=8)  # option that

    parser.add_argument('--outfile_prefix1', type=str, default='att1')  # option that takes a value

    args2 = parser.parse_args()
    args = copy.deepcopy(args2)

    if args.use_helm == 'yes': # use the helm model server.
        test_taker_info = helm_process_args(args.test_taker_modelname)
        print('loaded helm models')
    else:
        test_taker_lm, test_taker_tokenizer, modelpath_name, test_taker_client = process_args_for_models(
            args.test_taker_modelname)
        test_taker_info = (test_taker_lm, test_taker_tokenizer, test_taker_client)

    agent_lm, agent_tokenizer, agent_name, agent_client = process_args_for_models(args.agent_modelname)

    if args.tool_modelname is None:
        tool_lm, tool_tokenizer, tool_name, tool_client = agent_lm, agent_tokenizer, agent_name, agent_client
    else:
        tool_lm, tool_tokenizer, tool_name, tool_client = process_args_for_models(args.tool_modelname)

    evaluator_info = (tool_lm, tool_tokenizer, tool_client)
    agent_info = (agent_lm, agent_tokenizer, agent_client) # agent model

    if args.exp_mode == 'autobencher':
        history_dict = []
        historical_psg = []
        for iters in range(args.num_iters):
            args.outfile_prefix = args.outfile_prefix1 + str(iters + 1)
            summarized_content = summarize_over_history(history_dict, gold_key='gold_answer', verbose=False)
            history = [summarized_content]
            historical_psg = generate_full_qa(args.theme, agent_info, history, iters + 1,
                                                            outfile_prefix=args.outfile_prefix,
                                                            historical_psg=historical_psg,
                                                            category_gen_func=_refine_categories_targetacc_augmented,
                                                            generate_qa_func=generate_long_questions,
                                                            acc_target=args.acc_target)
            with open(f"{args.outfile_prefix}.KI_questions.json", "r") as f:
                json_category = json.load(f)
            if len(json_category) == 1: # remove the outer embedded list.
                json_category = json_category[0]
            gold_answer_json = copy.deepcopy(json_category)
            json_dict = solve_and_compare_questions(test_taker_info, evaluator_info, json_category, gold_answer_json,
                                                    args.outfile_prefix, 'gold_answer')
            history_dict.append(json_dict)

            verbose_description = get_summary_of_results(json_dict, verbose=False)
            print(verbose_description)

    elif args.exp_mode == 'naive_baseline':
        '''
        This is the most naive version of AutoBencher, 
        There is no previleged information (wiki) and no adaptive search.
        '''
        history_dict = []
        for iters in range(args.num_iters):
            args.outfile_prefix = args.outfile_prefix1 + str(iters+1)
            summarized_content = summarize_over_history(history_dict, gold_key='gold_answer')
            history = [summarized_content]

            json_category = _ask_question(args.theme, agent_info, history, iters+1, outfile_prefix=args.outfile_prefix)
            gold_answer_json = copy.deepcopy(json_category[0])

            json_dict = solve_and_compare_questions(test_taker_info, agent_info, json_category, gold_answer_json,
                                                    args.outfile_prefix, "answer")
            history_dict.append(json_dict)


    elif args.exp_mode == "baseline_without_privileged_info":
        '''
        The topic proposal component is the same as AutoBencher, meaning that this uses adaptive search. 
        The dataset generation component does not use privileged information. 
        '''
        history_dict = []
        historical_psg = []
        for iters in range(args.num_iters):
            args.outfile_prefix = args.outfile_prefix1 + str(iters + 1)
            summarized_content = summarize_over_history(history_dict, gold_key='gold_answer', verbose=False)
            history = [summarized_content]

            historical_psg = generate_full_qa(args.theme, agent_info, history, iters + 1,
                                                                 outfile_prefix=args.outfile_prefix,
                                                                 historical_psg=historical_psg,
                                                                 category_gen_func=_refine_categories_targetacc_augmented,
                                                                 generate_qa_func=generate_dataset_without_docs,
                                                                 apply_saliency_rerank=True)
            with open(f"{args.outfile_prefix}.KI_questions.json", "r") as f:
                json_category = json.load(f)
            if len(json_category) == 1:
                json_category = json_category[0]
            gold_answer_json = copy.deepcopy(json_category)
            json_dict = solve_and_compare_questions(test_taker_info, agent_info, json_category, gold_answer_json,
                                                    args.outfile_prefix, 'gold_answer')
            history_dict.append(json_dict)



    elif args.exp_mode == 'baseline_without_adaptive_search':
        '''
        This baseline do not use adaptive search, but it uses privileged information. 
        '''
        history_dict = []
        historical_psg = []
        for iters in range(args.num_iters):
            args.outfile_prefix = args.outfile_prefix1 + str(iters + 1)
            summarized_content = summarize_over_history(history_dict, gold_key='gold_answer',  verbose=False)
            history = [summarized_content]
            historical_psg = generate_full_qa(args.theme, agent_info, history, iters + 1,
                                                            outfile_prefix=args.outfile_prefix,
                                                            historical_psg=historical_psg,
                                                            category_gen_func=_refine_categories_random_augmented,
                                                            generate_qa_func=generate_long_questions,
                                                                 apply_saliency_rerank=False)
            with open(f"{args.outfile_prefix}.KI_questions.json", "r") as f:
                json_category = json.load(f)
            if len(json_category) == 1:
                json_category = json_category[0]
            gold_answer_json = copy.deepcopy(json_category)
            json_dict = solve_and_compare_questions(test_taker_info, agent_info, json_category, gold_answer_json,
                                                    args.outfile_prefix, 'gold_answer')
            history_dict.append(json_dict)
