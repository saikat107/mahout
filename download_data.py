questions = []
taken_question_ids = set()
from tqdm import tqdm
import time
from stackapi import StackAPI

SITE = StackAPI('stackoverflow')
for pi in range(1, 50):
    print("Sleeping for 30 seconds!")
    for _ in tqdm(range(120), desc=f"Iteration : {pi}"):
        time.sleep(0.25)
    print("Querying")
    data = SITE.fetch(
            "search/advanced", 
            q="pandas,dataframe", 
            accepted=True, 
            tagged="pandas;dataframe;python", 
            page=pi*5, 
            filter="withbody"
        )
    d = data['items']
#     'quota_max', 'quota_remaining'
    quota_max = data['quota_max']
    qr = data['quota_remaining']
    for itm in d:
        if (
            'accepted_answer_id' in itm.keys() and \
            'pandas' in itm['tags'] and \
            'dataframe' in itm['tags'] and \
            itm['question_id'] not in taken_question_ids
        ):
            taken_question_ids.add(itm['question_id'])
            questions.append(itm)
    print(len(questions), quota_max, qr, sep='\t')


import os
import json
os.makedirs('ten_thousand_questions', exist_ok=True)
ten_thousand_questions = open('ten_thousand_questions/questions_with_bodies.json', 'w')
json.dump(questions, fp=ten_thousand_questions, indent=4)
ten_thousand_questions.close()
