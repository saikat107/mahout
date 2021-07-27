import json
from bs4 import BeautifulSoup as bs
from tqdm.notebook import tqdm

import sys
import os

input_path = sys.argv[1]
out_dir = sys.argv[2]

#out_dir = output_path[:output_path.rindex("/")]
os.makedirs(out_dir, exist_ok=True)

accepted_answers = json.load(open(input_path))
print(len(accepted_answers))

from stackapi import StackAPI

SITE = StackAPI('stackoverflow')
aids = [i['accepted_answer_id'] for i in accepted_answers]
print(len(aids))

answers = []

for idx in range(100):
    try:
        data = SITE.fetch(
            'answers/{ids}',
            ids=aids[idx*100:(idx+1)*100],
            filter='withbody'
        )
        answers.extend(
            data['items']
        )
        print(len(answers), data['quota_remaining'], sep='\t')
    except:
        pass
    
final_answers = answers

id_to_answer = {
    answer['answer_id']: answer for answer in final_answers
}

for idx in range(len(accepted_answers)):
    try:
        accepted_answers[idx]['answer_body'] = id_to_answer[accepted_answers[idx]['accepted_answer_id']]['body']
        accepted_answers[idx]['question_body'] = accepted_answers[idx]['body']
    except:
        pass
    
import requests

urls = json.load(open("pandas_apis.json"))
data_frame_url = urls["df"]

def get_api_list(_url):
    page = requests.get(_url)
    soup = bs(page.content, 'html.parser')
    codes = soup.find_all("code", {"class": "xref py py-obj docutils literal notranslate"})
    apis = []
    for code in codes:
        api_name = code.find("span").text
        if "." in api_name:
            name = api_name[api_name.index(".") + 1:]
            apis.append(name)
            pass
        else:
            apis.append(api_name)
        pass
    return apis

data_frame_apis = get_api_list(data_frame_url)
# print(data_frame_apis)

list_of_apis = []
complete_api_sets = []

for key in urls:
    api_from_url = get_api_list(urls[key])
    list_of_apis.append({
        "url_key": key,
        "url": urls[key],
        "apis": api_from_url
    })
    complete_api_sets.extend(api_from_url)
    pass

# print("Dataframe APIs only")
# print(data_frame_apis)
# print("=" * 100)
# print("Pandas APIS")
complete_api_sets = list(set(complete_api_sets))
# print(complete_api_sets)
# print("=" * 100)


def dataframe_api_exists(line):
    for api in data_frame_apis:
        if api in line:
            return True
        pass
    return  False

def api_exists(line):
    for api in complete_api_sets:
        if api in line:
            return True
        pass
    return False


import re 
import keyword
import sys

keywords = keyword.kwlist

def keyword_exists(line):
    for kw in keywords:
        if (kw + " ") in line or (" " + kw) in line:
            return True
    return False


def min_word_filter(texts, min_word=5):
    filtered_texts = []
    for t in texts:
        if len(t.split()) >= min_word:
            filtered_texts.append(t)
            pass
        pass
    return filtered_texts

def extract_code(text, filters):
    soup = bs(text)
    all_code = [code.text for code in soup.find_all('code')]
    for f in filters:
        all_code = f(all_code)
    return all_code
    pass

def data_frame_exists(code):
    lines = [l.strip() for l in code.split("\n")]
    data_frame_re = "[0-9]+[, \t]+[.]*"
    matches = []
    for l in lines:
        if len(re.findall(data_frame_re, l)) > 0 \
            and not keyword_exists(l) \
            and not dataframe_api_exists(l) \
            and not api_exists(l):
            matches.append(True)
        else:
            matches.append(False)
    return any(matches)

def code_exists(code):
    lines = [l.strip() for l in code.split("\n")]
    matches = []
    for l in lines:
        if dataframe_api_exists(l) or keyword_exists(l) or api_exists(l):
            return True
    return False

def print_formatted_code(code, fp=sys.stdout):
    lines = code.split('\n')
    lines = ['\t\t' + l for l in  lines]
    print('\n'.join(lines), file=fp)
    fp.flush()
    
def extract_description(html):
    soup = bs(html)
    [code.extract() for code in soup.find_all('code')]
    text = re.sub("[ \n\t]+", " ", soup.text)
    return text
    pass
    
taken_answers = []

for a in accepted_answers:
    try:
        title = a['title']
        code_from_body = extract_code(a['question_body'], filters=[min_word_filter])
        code_from_answer = extract_code(a['answer_body'], filters=[min_word_filter])
        data_frames = []
        for cid, c in enumerate(code_from_body):
            is_df = data_frame_exists(c)
            is_code = code_exists(c)
            if is_df and not is_code:
                data_frames.append(c)
        
        taken_code = []
        for cid, c in enumerate(code_from_answer):
            is_df = data_frame_exists(c)
            is_pandas_code = api_exists(c)
            if is_pandas_code and not is_df:
                taken_code.append(c)
        if len(data_frames) == 2 and len(taken_code) > 0:
            a["formatted_input"] = {
                "qid": a['question_id'],
                'link': a['link'],
                "question": {
                    "title": a["title"],
                    "ques_desc" : extract_description(a['question_body'])
                },
                "io": data_frames,
                "answer" : {
                    "ans_desc" : extract_description(a['answer_body']),
                    "code": taken_code
                }
            }
            taken_answers.append(a)
    except:
        pass

print(len(taken_answers))


import json 

fp = open(out_dir + "/formatted_output_takens.txt", "w")

for a in taken_answers:
    fmt_input = a["formatted_input"]
    print("\"qid\": %s" % fmt_input["qid"], file=fp)
    print("\"link\": %s" % fmt_input["link"], file=fp)
    print("\"question\": {", file=fp)
    print("\t\"title\": %s" % fmt_input["question"]["title"], file=fp)
    print("\t\"desc\": %s" % fmt_input["question"]["ques_desc"], file=fp)
    print("}", file=fp)
    print("\"io\": {", file=fp)
    print("\t\"Frame-1\": ", file=fp)
    print_formatted_code(fmt_input["io"][0], fp)
    print("\t\"Frame-2\":", file=fp)
    print_formatted_code(fmt_input["io"][1], fp)
    print("}", file=fp)
    print("\"answer\": {", file=fp)
    print("\t\"desc\": %s", fmt_input["answer"]["ans_desc"], file=fp)
    print("\t\"code-snippets\": [", file=fp)
    for t in fmt_input["answer"]["code"]:
        print_formatted_code(t, fp)
        print("\t\t" + ("-" * 70), file=fp)
    print("\t]", file=fp)
    print("}", file=fp)
    print("=" * 100, file=fp)
    print("\n", file=fp)

fp.close()

all_answers_file = open(out_dir + "/all_accepted_answers_with_all_details.json", 'w')
json.dump(obj=accepted_answers, fp=all_answers_file, indent=4)
all_answers_file.close()

taken_answer_file = open(out_dir + "/taken_answers_with_all_details.json", "w")
json.dump(obj=taken_answers, fp=taken_answer_file, indent=4)
taken_answer_file.close()
                         
