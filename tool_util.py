import copy
import re
import requests
import os, argparse, ast, json, tqdm
from pydantic import BaseModel, Extra, root_validator
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from time import sleep
from collections import defaultdict
from autogen.code_utils import UNKNOWN, extract_code, execute_code, infer_lang
import numpy as np
from bs4 import BeautifulSoup
from util import gen_from_prompt


DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply "TERMINATE" in the end when everything is done.
"""

DEFAULT_JSON_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your reasoning and language skills.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
Reply "TERMINATE" in the end when everything is done.
"""

DEFAULT_DESCRIPTION = "A helpful and general-purpose AI assistant that has strong language skills, Python skills, and Linux command line skills."



def extract_json_v2(json_text, outfilename):
    response = json_text.replace("TERMINATE", "")
    if "```json" in response:
        # parse the json file
        try:
            extracted_json = extract_code(response)
            combined_json = sum([], [ast.literal_eval(xx[1]) for xx in extracted_json])
        except:
            if '...' in response:
                response = response.replace('...', '')
                extracted_json = extract_code(response)
                combined_json = sum([], [ast.literal_eval(xx[1]) for xx in extracted_json])
            else:
                response2 = "\n".join(response.split('\n')[:-1]) + "]\n```"
                extracted_json = extract_code(response2)
                combined_json = sum([], [ast.literal_eval(xx[1]) for xx in extracted_json])
        # load the json_string.
        json_dict = combined_json
        # json_dict = ast.literal_eval(combined_json)
        if outfilename is not None:
            with open(outfilename, "w") as f:
                json.dump(json_dict, f)

    else:
        assert False, "fail to output json file."
    return json_dict

def test_taker_inference(test_model_info, problem_json, outfile, bsz=1, temperature=0.01, max_length=50):
    if len(test_model_info) == 3:
        model_choice, tokenizer_choice, client_choice = test_model_info
        auth = None
        use_helm = False
    elif len(test_model_info) == 4:
        model_choice, tokenizer_choice, client_choice, auth = test_model_info
        use_helm = True

    print(f'writing to {outfile}')
    out_handle = open(outfile, 'w')
    full_result_lst = []
    batch_lst, line_lst = [], []
    for line in tqdm.tqdm(problem_json):
        line['prompt'] = "Output just with the final answer to the question.\nQuestion:" + line[
            'question'] + "\n" + "Answer:"
        line_lst.append(line)
        batch_lst.append(line['prompt'])
        if len(batch_lst) < bsz:
            continue  # batch not full yet
        request_result = gen_from_prompt(model=model_choice, tokenizer=tokenizer_choice, prompt=batch_lst,
                                         echo_prompt=False, temperature=temperature, max_tokens=max_length,
                                         service=client_choice,
                                         terminate_by_linebreak='no', use_helm=use_helm, auth=auth,
                                         verbose=False)

        for line, xx in zip(line_lst, request_result.completions):
            # print(line['prompt'])
            # print('-' * 100)
            # print(xx.text)
            line['test_taker_response'] = xx.text
            print(json.dumps(line), file=out_handle)
            full_result_lst.append(line)
        batch_lst, line_lst = [], []
    if len(batch_lst) > 0:
        request_result = gen_from_prompt(model=model_choice, tokenizer=tokenizer_choice, prompt=batch_lst,
                                         echo_prompt=False, temperature=temperature, max_tokens=max_length,
                                         service=args.model_auth, terminate_by_linebreak='no', use_helm=use_helm,
                                         auth=auth, verbose=False)
        for line, xx in zip(line_lst, request_result.completions):
            line['test_taker_response'] = xx.text
            print(json.dumps(line), file=out_handle)
            full_result_lst.append(line)
    out_handle.close()
    return full_result_lst


def _generate_lm_answers(question_inputs, test_model_info, agent_model_info, outfile_prefix='att1'):
    if os.path.exists(f"{outfile_prefix}.test_taker_inference.json"):
        full_result_lst = []
        with open(f"{outfile_prefix}.test_taker_inference.json", 'r') as in_handle:
            for line in in_handle:
                line = json.loads(line.strip())
                full_result_lst.append(line)
        return full_result_lst

    # test_taker_lm, test_taker_tokenizer, test_taker_client = test_model_info
    if isinstance(question_inputs, list) or isinstance(question_inputs, dict):
        question_inputs_str = json.dumps(question_inputs, indent=2)
    else:
        assert False

    if isinstance(question_inputs, list) and isinstance(question_inputs[0], list):
        json_dict = question_inputs[0]
    elif isinstance(question_inputs, list):
        json_dict = question_inputs
    else:
        print('question_inputs should be a list.')
        assert False

    full_result_lst = test_taker_inference(test_model_info, json_dict,
                                           outfile=f"{outfile_prefix}.test_taker_inference.json")

    return full_result_lst



def search_related_pages(search_query):
    # URL for Wikipedia API search action
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={search_query}&format=json&cmlimit=max"

    # Making the request
    response = requests.get(url)

    # Checking if request was successful
    if response.status_code == 200:
        data = response.json()
        search_results = data['query']['search']

        # Extracting titles of the search results
        related_pages = [result['title'] for result in search_results]
        return related_pages
    else:
        print("Failed to retrieve data from Wikipedia API.")
        return []


def get_pageviews(page_title, start_date="2020040100", end_date="2023040700"):
    access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIwMDFkMTFmNmQ2MzVmMGY4YmI3MDlkNWViN2ZhNDRlYiIsImp0aSI6IjMzOGQ0Mzc0YzNmZjE5NjBlZDkzNjIwNTdiYjMwYjExOWYzZTY2MzVkZjM3NmY3NDcyZjczMDcyMjNiYzU4ODFjODBkOTliOTZmMjAzZGNkIiwiaWF0IjoxNzEyNjEwMTg0LjY4OTIyNywibmJmIjoxNzEyNjEwMTg0LjY4OTIzLCJleHAiOjMzMjY5NTE4OTg0LjY4NzY1Mywic3ViIjoiNzUzODczODIiLCJpc3MiOiJodHRwczovL21ldGEud2lraW1lZGlhLm9yZyIsInJhdGVsaW1pdCI6eyJyZXF1ZXN0c19wZXJfdW5pdCI6NTAwMCwidW5pdCI6IkhPVVIifSwic2NvcGVzIjpbImJhc2ljIl19.YN0ZvSzsBuYe3Mg-r0C63cWxDXPU3GOCyspUqg4mMv27Qw1FJq9F9H6JKJAUMrqQxB-xyWZqpu8mekvMoxb3Ha5S2fpPbuM4gMB0JketqG2obaDd4QqgtJjg8KDYKwR8ieKoPRLDSHv3Tv4NcvIL-EvzjkRybqrukzQwttwuBUwxmlY8vhC1BZed7URt_-KhMYPsnNfJLSBeWivYJOmrqF2S04AOS0Egjul8Pz_yXAQ7q7aqpIwg6X2jod0ZN5h1gnmAvZmoLB7mKSAxrHEUL2zaQ8BVERWostWVA9ek556cuUJe5NusQ0XW7pcsYIi0YpFjKOBuq-tXzuOlbxFhlbwrp6xkhE_grQGNs1IxyT-w_sjQc2gI48FDe0ldDrTg6ZmgLELsjJM8xOxBy1ng1fY73p-QnaDdxX4hqRw2ZBDlZ1E2j84lvVrv62x_SHPiBNAeywEPcOqDRV_XbU6ArOyJ7QTZXRu9UOT0XDQ-Fx3maCRGb35W4aOtLSWL-SSXYLI8ZuOQ2BwKQQYYbEDMp0W7NjHWzh8YPv6Y2wDaMzsAqaxk2c36pNvTToiTc_P6_a56lydQwoT8ACx1kzzw5lTNPKPEPxPGNiMgtsL3VqtxJWMR7Lgq-ZKwI7cwQ5FTp2YriQDBYuvoaDQeG_eVh8BlNlyg26OYojtYbNos3os"
    client_id = "001d11f6d635f0f8bb709d5eb7fa44eb"
    client_secret = "630b434daa4c8f6cce03b1c294b59574c1ce9431"  # Example client secret
    headers = {
        'Authorization': f'Bearer {access_token}',
        'User-Agent': 'wikipagerank',
    }

    # Construct the API URL with the appropriate parameters
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{page_title}/daily/{start_date}/{end_date}"
    # Make the HTTP GET request to the API
    response = requests.get(url, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Extract the pageview data
        print('retrieved for ', page_title)
        views = sum(item['views'] for item in data['items'])
        return views
    else:
        print(f"Failed to retrieve pageviews data for {page_title}. Status code: {response.status_code}")
        return 0

def clean_str(p):
  try:
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
  except:
    return ''

def filter_paragraph(paragraph_lst):
    return [p for p in paragraph_lst if len(p.split(" ")) > 2 and len(p.split(".")) > 1]


def get_page_obs(page):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def search_step(entity, output_more=False):
    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    response_text = requests.get(search_url).text
    soup = BeautifulSoup(response_text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    if result_divs:  # mismatch
      result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
      # obs = f"Could not find {entity}. Similar: {result_titles[:5]}."
      print(f"Could not find {entity}. Search for similar entities, {result_titles[0]}, instead")
      obs, entity = search_step(result_titles[0])
    else:
      print('found entity', entity)
      page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
      if any("may refer to:" in p for p in page):
        obs, entity = search_step("[" + entity + "]")
      else:
        page_ = ""
        for p in page:
          if len(p.split(" ")) > 2:
              page_ += clean_str(p)
              if not p.endswith("\n"):
                  page_ += "\n"
        obs = get_page_obs(page_)
        if output_more:
            obs = filter_paragraph(obs)
        else:
            obs = filter_paragraph(obs[:10])

    return obs, entity