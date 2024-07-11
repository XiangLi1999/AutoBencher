import os, sys, json
import glob, torch
import numpy as np
import random, re, string, argparse
from collections import Counter
import getpass
from anthropic import Anthropic

from helm.common.authentication import Authentication
from helm.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from helm.common.request import Request, RequestResult
from helm.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from helm.proxy.accounts import Account
from helm.proxy.services.remote_service import RemoteService

import datasets
from collections import namedtuple, defaultdict
from transformers import set_seed,  AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import datasets
import openai, time
from openai import OpenAI


# CRFM_KEY = "TODO"
# openai.api_key = "TODO"
# anthropic_api_key = "TODO"



def load_model(modelpath):
    print(f'loading from {modelpath}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(modelpath)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    print('---' * 100, modelpath, '---' * 100)
    model = transformers.AutoModelForCausalLM.from_pretrained(modelpath, torch_dtype = torch.float16,
                                                              low_cpu_mem_usage = True,).cuda()
    return model, tokenizer


def load_via_deepspeed(model_name):
    from transformers.deepspeed import HfDeepSpeedConfig
    config = AutoConfig.from_pretrained(model_name)
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    dtype = config.torch_dtype  # torch.bfloat16 if model_name in ["bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16
    model_hidden_size = config.hidden_size
    train_batch_size = 1 * world_size

    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": model_hidden_size * model_hidden_size,
            "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
            "stage3_param_persistence_threshold": 0,
        },
        "steps_per_print": 2000,
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.eval()
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module
    return model


def gen_from_prompt(model, tokenizer, prompt, echo_prompt=False,
                    temperature=0., max_tokens=20, num_completions=1, output_scores=False,
                    service=None, seed=101, process_func=None, terminate_by_linebreak=True,
                    verbose=False, use_helm=False, auth=None):
    if service is None:
        assert model is not None
        if process_func is not None:
            prompt = process_func(prompt)
        prompt_ids = tokenizer(prompt, return_tensors='pt', padding=True)
        attention_mask = prompt_ids['attention_mask'].to(model.device)
        prompt_ids = prompt_ids['input_ids'].to(model.device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,): # if True: # LISA_DEBUG
            generated_ids = model.generate(input_ids=prompt_ids, attention_mask=attention_mask,
                                           temperature=temperature, do_sample=True,
                                           max_length=max_tokens + prompt_ids.size(1),
                                           num_return_sequences=num_completions,
                                           eos_token_id=2, pad_token_id=2)
        generated_text = tokenizer.batch_decode(generated_ids[:, prompt_ids.size(1):], skip_special_tokens=True)
        if terminate_by_linebreak == 'no':
            generated_text = [x for x in generated_text]
        else:
            generated_text = [x.split('\n')[0] for x in generated_text]

        Completion = namedtuple('Completion', ['text'])
        compl = [Completion(text=x) for x in generated_text]
        RequestResult = namedtuple('RequestResult', ['completions', 'success', 'embedding', 'cached'])
        request_result = RequestResult(completions=compl, success=True, embedding=None,
                                       cached=False)

    elif model.startswith('gpt'):
        generated_text = query_gpt4(client=service, model=model, prompt_lst=prompt, temperature=temperature, max_tokens=max_tokens,
                             num_completions=num_completions, random=str(seed), verbose=verbose)  # stop_sequences=['\n']) #
        # print(generated_text, 'turbo')
        Completion = namedtuple('Completion', ['text'])
        compl = [Completion(text=x) for x in generated_text]
        RequestResult = namedtuple('RequestResult', ['completions', 'success', 'embedding', 'cached'])
        request_result = RequestResult(completions=compl, success=True, embedding=None,
                                       cached=False)
    elif model.startswith('claude'):
        generated_text = query_claude(client=service, model=model, prompt_lst=prompt, temperature=temperature,
                                      max_tokens=max_tokens, num_completions=num_completions, random=str(seed), verbose=verbose)
        # print(generated_text, 'turbo')
        Completion = namedtuple('Completion', ['text'])
        compl = [Completion(text=x) for x in generated_text]
        RequestResult = namedtuple('RequestResult', ['completions', 'success', 'embedding', 'cached'])
        request_result = RequestResult(completions=compl, success=True, embedding=None,
                                       cached=False)
    elif use_helm:
        assert len(prompt) == 1 # only one prompt
        request_result = None
        num_retry = 0
        max_retry = 5
        while num_retry < max_retry:
            try:
                request = Request(model=model, prompt=prompt[0], echo_prompt=echo_prompt,  ##"openai/text-davinci-003",
                                  temperature=temperature, max_tokens=max_tokens,
                                  num_completions=num_completions, random=str(seed), stop_sequences=['\n']) #
                request_result = service.make_request(auth, request)
                break
            except Exception as e:
                print(e)
                print('retrying...')
                num_retry += 1
                time.sleep(10)
        if request_result is None:
            raise RuntimeError(f"Could not get completion after {max_retry} retries.")
    else:
        raise NotImplementedError
    return request_result

def query_claude(client, model, prompt_lst, temperature, max_tokens, num_completions, random, verbose, max_num_retries=5):
    num_retries = 0
    result_lst = []
    # Repeat the query until we get a valid response.
    message = None
    for prompt in prompt_lst:
        while num_retries < max_num_retries:
            try:
                if verbose:
                    print(f"+++++++++++ Model Prompt +++++++++++\n {prompt}")
                message = client.messages.create(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model,
                )
                break
            except:  # noqa
                print("Retrying...")
                num_retries += 1
                time.sleep(10)
        if message is None:
            raise RuntimeError(f"Could not get completion after {max_num_retries} retries.")

        result_txt = message.content[0].text
        if verbose:
            print(f"+++++++++++ Model Output +++++++++++\n {result_txt}")
        result_txt = result_txt.strip()

        result_lst.append(result_txt)
    return result_lst



def query_gpt4(client, model, prompt_lst, temperature, max_tokens, num_completions, random, verbose, max_num_retries=5):
    # Randomly select one assistant to be presented first.
    num_retries = 0
    result_lst = []
    # Repeat the query until we get a valid response.
    completion = None
    for prompt in prompt_lst:
        while num_retries < max_num_retries:
            try:
                if verbose:
                    print(f"+++++++++++ Model Prompt +++++++++++\n {prompt}")
                completion = client.chat.completions.create(
                    model=model, #"gpt-4", #"gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful AI agent.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=num_completions,
                    # stop=["\n"],
                )
                break
            except Exception as e:  # noqa
                print(e)
                print("Retrying...")
                num_retries += 1
                time.sleep(10)

        if completion is None:
            raise RuntimeError(f"Could not get completion after {max_num_retries} retries.")

        result_txt = completion.choices[0].message.content

        if verbose:
            usage = completion.usage
            print(usage)
            total_tokens = usage.total_tokens
            print(total_tokens, 'total tokens')

        if verbose:
            print(f"+++++++++++ Model Output +++++++++++\n {result_txt}")
        result_txt = result_txt.strip()

        result_lst.append(result_txt)
    return result_lst




def helm_process_args(experiment_model):

    # An example of how to use the request API.
    # api_key = getpass.getpass(prompt="Enter a valid API key: ")
    auth = Authentication(api_key=CRFM_KEY)
    service = RemoteService("https://crfm-models.stanford.edu")

    # Access account and show my current quotas and usages
    account: Account = service.get_account(auth)
    print(account.usages)
    return experiment_model.lower(), None, service, auth



def process_args_for_models(experiment_model):

    if experiment_model.startswith('gpt'):
        modelpath = experiment_model
        model_choice = modelpath
        tokenizer_choice = None
        modelpath_name = modelpath
        model_client = OpenAI(api_key=openai.api_key, organization=openai.organization)


    elif experiment_model.startswith('claude'):
        client = Anthropic(
            api_key=anthropic_api_key,
        )
        model_choice = experiment_model
        tokenizer_choice = None
        model_client = client
        modelpath_name = experiment_model
    else:
        model_choice, tokenizer_choice = load_model(experiment_model)
        modelpath_name = os.path.basename(experiment_model).replace('/', '_')
        model_client = None

    return model_choice, tokenizer_choice, modelpath_name, model_client
