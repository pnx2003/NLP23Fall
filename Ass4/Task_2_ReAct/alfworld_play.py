import os
import openai
from generate_reflexion import update_memory
import yaml
import alfworld
import alfworld.agents.environment
import json
import sys
import time
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='gpt-3.5-turbo-instruct')
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default='../data')
    parser.add_argument("--type", type=str, default='react')
    parser.add_argument('is_reflexion', type=bool, default=False)
    args = parser.parse_args()
    return args

def llm(prompt, stop=["\n"]):
    response = openai.Completion.create(
      model=args.model,
      prompt=prompt,
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop
    )
    return response["choices"][0]["text"]




def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

    
def alfworld_run(prompt, to_print=True, ob='', game_name = 0):
    init_prompt = prompt + ob + '\n>'
    prompt = ''
    if to_print:
        print(ob)
        sys.stdout.flush()
    import pdb;pdb.set_trace()
    for i in range(1, 50):
        # sleep for 20s
        time.sleep(20)
        action = llm(init_prompt + prompt, stop=['\n']).strip()
        done = False
        if "Action" in action:
            action = action.split(':')[1].strip()

            
        if action.startswith('Think:') or action.startswith('think:'):
            observation = 'OK.'
            
        else:
            observation, reward, done, info = env.step([action])
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        prompt += f' {action}\n{observation}\n>'
        input_prompt = init_prompt + prompt
        with open(f'prompt_{args.type}_{game_name}.txt', 'w') as f:
            f.write(input_prompt)
        if done:
            return reward
    return 0

def main():
    for _ in range(134):
        # obtain specific game
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0]\
            .split('/')[-3:-2])
        print(name)
        for i, (k,v) in enumerate(prefixes.items()):
            if name.startswith(k):

                prompt = 'Interact with a household to solve a task. Here are two examples.\n' \
                    + d[f'{k}_1'] + d[f'{k}_0'] + '\nHere is the task.\n'
                    
                r = alfworld_run(prompt, ob=ob, game_name=name)
                if r != 1:
                    print(f'Failed {name}')
                    if args.is_reflexion:
                        print(f'Allows the Agent to reflect upon a past experience.')
                        query = update_memory(f'./prompt_{args.type}_{name}.txt')
                        prompt = 'Interact with a household to solve a task. Here are two examples.\n' \
                            + d[f'{k}_1'] + d[f'{k}_0'] + '\nHere is the task.\n' + query
                        alfworld_run(prompt, ob=ob, game_name=name)
                    
      
      
if __name__ == '__main__': 
    args = get_args()
    openai.api_key = args.key#"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" 
    os.environ['ALFWORLD_DATA'] = args.data_dir
    
    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    split = "eval_out_of_distribution"
    env = alfworld.agents.environment.AlfredTWEnv(config, train_eval=split)
    env = env.init_env(batch_size=1)
    folder = './'
    prompt_file = f'alfworld_{args.type}.json'
    with open(folder + prompt_file, 'r') as f:
        d = json.load(f)
    prefixes = {
        'pick_and_place': 'put',
        'pick_clean_then_place': 'clean',
        'pick_heat_then_place': 'heat',
        'pick_cool_then_place': 'cool',
        'look_at_obj': 'examine',
        'pick_two_obj': 'puttwo'
    }
    
    main()