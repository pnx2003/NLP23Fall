import os
import openai
import yaml
import alfworld
import alfworld.agents.environment
import json
import sys
import time

def llm(prompt, stop=["\n"]):
    response = openai.Completion.create(
      model="gpt-3.5-turbo-instruct",
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



    



def alfworld_run(prompt, to_print=True, ob='', game_id = 0):
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

            
        if action.startswith('Think:'):
            observation = 'OK.'
            
        else:
            observation, reward, done, info = env.step([action])
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        prompt += f' {action}\n{observation}\n>'
        input_prompt = init_prompt + prompt
        with open(f'prompt_act_{game_id}.txt', 'w') as f:
            f.write(input_prompt)
        if done:
            return reward
    return 0

def main(is_resume=False):
    for _ in range(134):
        # obtain specific game
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0]\
            .split('/')[-3:-2])
        print(name)
        for i, (k,v) in enumerate(prefixes.items()):
            if name.startswith(k):
                if not i==3:
                    continue
                # with open(f'prompt_react_{i}.txt', 'r') as f:
                #     prompt = f.read()
                # action_space  = prompt.split('New plan:')[-1]
                # for line in action_space.split('\n'):
                #     if line.startswith('>'):
                #         observation, reward, done, info = env.step([line[1:]])
                # r = alfworld_run(prompt, ob='', game_id=i)
                # import pdb; pdb.set_trace()
                
                if os.path.exists(f'./prompt_act_{name}.txt'):
                    with open(f'./prompt_act_{name}.txt', 'r') as f:
                        prompt = f.read()
                    done = (False,)
                    # find all the lines start with '>'
                    # for line in prompt.split('\n'):
                    #     if line.startswith('>'):
                            
                    #         observation, reward, done, info = env.step([line[1:]])
                    if not done[0]:
                        r = alfworld_run(prompt, ob='', game_id=name)
                        
                        
                else:
                    prompt = 'Interact with a household to solve a task. Here are two examples.\n' \
                     + d[f'{k}_1'] + d[f'{k}_0'] + '\nHere is the task.\n'
                    
                    r = alfworld_run(prompt, ob=ob, game_id=name)
                
        print('------------\n')
      
      
if __name__ == '__main__': 
    openai.api_key = "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" 
    os.environ['ALFWORLD_DATA'] = "../data"
    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    
    split = "eval_out_of_distribution"
    env = alfworld.agents.environment.AlfredTWEnv(config, train_eval=split)
    env = env.init_env(batch_size=1)
    folder = './'
    prompt_file = 'alfworld_act.json'
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
    cnts = [0] * 6
    rs = [0] * 6
    
    main(is_resume=False)