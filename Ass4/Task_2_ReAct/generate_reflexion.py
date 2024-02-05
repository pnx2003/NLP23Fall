#from new import llm
from typing import List, Dict, Any
from tmp import llm
import openai
with open("./reflection_2shot.txt", 'r') as f:
    FEW_SHOT_EXAMPLES = f.read()

def _generate_reflection_query(log_str: str) -> str:
    """Allows the Agent to reflect upon a past experience."""
    query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task.
    Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake 
    with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need 
    this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{FEW_SHOT_EXAMPLES}

{log_str}"""
    query += '\n\nNew plan:'
    return query

def update_memory(trial_log_path):
    """Updates the given env_config with the appropriate reflections."""
    # with open(trial_log_path, 'r') as f:
    #     full_log: str = f.read()
    # reflection_query: str = _generate_reflection_query(full_log)
    # NEW_PLAN = llm(reflection_query, stop=['\n'])
    # with open(trial_log_path, 'a') as f:
    #     f.write(f'\nNew plan: {NEW_PLAN}')
    with open(trial_log_path, 'r') as f:
        prompt = f.read()
        prompt = 'Here is the task.\n' + prompt.split('Here is the task.\n')[-1]\
            + '\nSTATUS:Fail'
        query = _generate_reflection_query(prompt)
        action = llm(query)
        query += action
        
    return query.split('Here is the task.\n')[-1]
