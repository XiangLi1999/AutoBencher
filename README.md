# AutoBencher

To quick start, you need to install the following dependencies:

```bash
pip insall -r requirements.txt 
```

Then, you can run the following command to start the benchmark to experiment with the kowledge intensive tasks:
```bash
python run_scripts.py wiki
python run_scripts.py multilingual
python run_scripts.py math
```

Specifically, the above scripts run the following command: 
    
```bash
python wiki_autobencher.py --exp_mode autobencher --test_taker_modelname gpt-4-turbo-preview  --use_helm no --agent_modelname gpt-4-turbo-preview --theme history --outfile_prefix1 KI/history. 
```
