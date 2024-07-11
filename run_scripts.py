import os, sys

# model_lst = ["claude-2.0",  "mistralai/Mixtral-8x7B-Instruct-v0.1", 'claude-3-opus-20240229', 'gpt-4-turbo-2024-04-09', 'claude-3-sonnet-20240229', "gpt-3.5-turbo-0613", "google/gemini-pro"]
# model_lst = ['claude-3-opus-20240229', 'gpt-4-turbo-2024-04-09', 'claude-3-sonnet-20240229'] #'mistralai/Mixtral-8x22B-Instruct-v0.1',
             # 'databricks/dbrx-instruct',  'google/gemma-7b-it',  'meta-llama/Llama-3-70b-chat-hf',
             # 'Qwen/Qwen1.5-72B-Chat', 'zero-one-ai/Yi-34B-Chat', 'deepseek-ai/deepseek-llm-67b-chat',  ]

model_lst = ["gpt-4-turbo-preview",]
subject_name = ["history"] # science, economy
simple_name = ["history"] # science, economy

#
# multilingual_lst = ["mistralai/Mistral-7B-Instruct-v0.1", 'lmsys/vicuna-7b-v1.5',
#                     'openagi-project/OpenAGI-7B-v0.1', 'openchat/openchat-3.5-0106',
#                     '/juice5/scr5/nlp/llama-2-hf-latest/Llama-2-7b-chat-hf',
#                     'google/gemma-7b']
#
# code_model_lst = ['codellama/CodeLlama-70b-Instruct-hf', 'deepseek-ai/deepseek-coder-33b-instruct',
#                   'databricks/dbrx-instruct',  'meta-llama/Llama-3-70b-chat-hf',  'mistralai/Mistral-7B-Instruct-v0.1', ]
#
#
# base_lst = [ 'mistralai/Mistral-7B-Instruct-v0.1',
#            'openagi-project/OpenAGI-7B-v0.1',
#             'lmsys/vicuna-7b-v1.5',
#            '/juice5/scr5/nlp/llama-2-hf-latest/Llama-2-7b-chat-hf',
#            'Xwin-LM/Xwin-Math-7B-V1.0',
#            'WizardLM/WizardMath-7B-V1.0',
#            'EleutherAI/gpt-neo-2.7B',
#            'alpaca-7b',
#            'HuggingFaceH4/zephyr-7b-beta',
#            'openchat/openchat-3.5-0106' # add phi,
#             ]


assert len(sys.argv) == 2, "please specify the mode (multilingual, wiki, math)"

if sys.argv[1] == 'wiki':
    for model in model_lst:
        if "claude" in model or 'gpt' in model:
            use_helm = "no"
            modelname_base = model
        elif 'Mistral-7B' in model or 'vicuna' in model:
            use_helm = 'no'
            modelname_base = model.split("/")[1]
        else:
            modelname_base = model.split("/")[1]
            use_helm = "yes"
        for subject, simple in zip(subject_name, simple_name):
            command = (
                "HF_MODULES_CACHE=/scr/biggest/xlisali/cache TRANSFORMERS_CACHE=/scr/biggest/xlisali/cache  HF_DATASETS_CACHE=/scr/biggest/xlisali/cache  "
                "python wiki_autobencher.py --exp_mode autobencher "
                f"--test_taker_modelname {model}  --use_helm {use_helm} "
                f"--agent_modelname gpt-4-turbo-preview --theme {subject} "
                f"--outfile_prefix1 KI/{simple}.tgtacc_5word+gpt-4-turbo-preview+{modelname_base}0.1--0.3. "
                f"--tool_modelname gpt-4-turbo-preview --acc_target '0.1--0.3'")
            print(command)
            os.system(command)

elif sys.argv[1] == 'multilingual':
    for model in model_lst:
        if "claude" in model or 'gpt' in model:
            use_helm = "no"
            modelname_base = model
        elif 'Mistral-7B' in model or 'vicuna' in model:
            use_helm = 'no'
            modelname_base = model.split("/")[1]
        else:
            use_helm = "yes"
            modelname_base = model.split("/")[1]

        command = (f"TRANSFORMERS_CACHE=/scr/biggest/xlisali/cache HF_DATASETS_CACHE=/scr/biggest/xlisali/cache "
                   f"HF_DATASETS_CACHE=/scr/biggest/xlisali/cache   "
                   f"python multilingual_autobencher.py --exp_mode autobencher  --agent_modelname gpt-4-turbo-preview "
                   f" --test_taker_modelname {model} --outfile_prefix1 multilingual/5word_v3_{modelname_base}  "
                   f" --use_helm {use_helm} ")
        print(command)
        os.system(command)

elif sys.argv[1] == 'math':
    for model in model_lst:
        if "claude" in model  or 'gpt' in model:
            use_helm = "no"
            modelname_base = model
        elif 'Mistral-7B' in model or 'vicuna' in model:
            use_helm = 'no'
            modelname_base = model.split("/")[1]
        else:
            use_helm = "yes"
            modelname_base = model.split("/")[1]
        command_math = ("HF_MODULES_CACHE=/scr/biggest/xlisali/cache TRANSFORMERS_CACHE=/scr/biggest/xlisali/cache "
                        " HF_DATASETS_CACHE=/scr/biggest/xlisali/cache "
                        "python math_autobencher.py --exp_mode autobencher --agent_modelname gpt-4-turbo-preview "
                        f"--test_taker_modelname {model} "
                        f"--outfile_prefix1 math_v5/tgtacc_5word+gpt-4-turbo-preview+{modelname_base}0.1--0.3. "
                        f"--use_helm {use_helm}")

        print(command_math)
        os.system(command_math)
