import json
import keyword
import os
import re

import nltk
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def get_details_about_apis(which_type_of_url, _url):
    invalid_count = 0
    page = requests.get(_url)
    soup = bs(page.content, 'html.parser')
    all_about_apis = []
    codes = soup.find_all("code", {"class": "xref py py-obj docutils literal notranslate"})
    for code in codes:
        api_name = code.find("span").text
        if "." in api_name:
            api_name = api_name[api_name.index(".") + 1:]
        documentation = list(code.parent.parent.parent.parent.children)[2].find("p").text
        try:
            category = code.parent.parent.parent.parent.parent.parent.parent.find("h2").text[:-1]
        except:
            category = which_type_of_url
        if len(api_name.strip()) == 0 or len(documentation.strip()) == 0 or len(category.strip()) == 0:
            invalid_count += 1
            continue
        all_about_apis.append({
            "api": re.sub(",", " ", str(api_name.encode("ascii", errors="ignore").decode())),
            "doc": re.sub(",", " ", str(documentation.encode("ascii", errors="ignore").decode())),
            "category": re.sub(",", " ", str(category.encode("ascii", errors="ignore").decode()))
        })
    # print(invalid_count)
    return all_about_apis


def dataframe_api_exists(line, data_frame_apis):
    tokens = nltk.wordpunct_tokenize(line)
    for t in tokens:
        if t in data_frame_apis:
            return True
    return False


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


def data_frame_exists(code, df_apis):
    lines = [l.strip() for l in code.split("\n")]
    data_frame_re = "[0-9]+[, \t]+[.]*"
    matches = []
    for l in lines:
        if len(re.findall(data_frame_re, l)) > 0 \
                and not keyword_exists(l) \
                and not dataframe_api_exists(l, df_apis):
            matches.append(True)
        else:
            matches.append(False)
    return any(matches)


def code_exists(code, df_apis):
    lines = [l.strip() for l in code.split("\n")]
    matches = []
    for l in lines:
        if dataframe_api_exists(l, df_apis) or keyword_exists(l):
            return True
    return False


def print_formatted_code(code):
    lines = code.split('\n')
    lines = ['\t\t' + l for l in lines]
    print('\n'.join(lines))


def extract_description(html):
    soup = bs(html)
    [code.extract() for code in soup.find_all('code')]
    text = re.sub("[ \n\t]+", " ", soup.text)
    return text
    pass


def get_train_data(api_lists, taken_test_qids):
    train_examples = []
    example_id = 1
    for part in range(1, 4):
        data = json.load(open(f"25_K_Examples/part-{part}-output/all_accepted_answers_with_all_details.json"))
        for point in tqdm(data):
            if point['question_id'] in taken_test_qids:
                continue
            qtitle = point['title']
            question = point['question_body']
            answer = point['answer_body']
            code_from_answer = extract_code(answer, filters=[min_word_filter])
            qdesc = extract_description(question)
            taken_code = []
            for cid, c in enumerate(code_from_answer):
                is_df = data_frame_exists(c, api_lists)
                is_pandas_code = dataframe_api_exists(c, api_lists)
                if is_pandas_code and not is_df:
                    taken_code.append(c)

            apis = list()
            for c in taken_code:
                tokens = nltk.wordpunct_tokenize(c)
                for tidx, token in enumerate(tokens):
                    token = token.strip()
                    if tidx >= 0:
                        prev_token = tokens[tidx - 1].strip()[-1]
                        if token == "T":
                            token = "transpose"
                        if (token in api_lists and prev_token == "."):
                            apis.append(token)
            if len(apis) <= 0:
                continue
            api_seq = apis

            train_examples.append({
                'qid': point['question_id'],
                'id': example_id,
                'q': qtitle.strip().lower(),
                'd': qdesc.strip().lower(),
                "apis": api_seq,
                'link': point['link']
            })
            example_id += 1
            pass
        pass
    return train_examples
    pass


def get_test_data(api_lists):
    examples = []
    fail_count = 0
    example_id = 1
    id_to_link = {}
    taken_api_list = set()
    for i in range(1, 4):
        example_file = f"25_K_Examples/part-{i}-output/taken_answers_with_all_details.json"
        data = json.load(open(example_file))
        for e in tqdm(data):
            try:
                ques_id = e['question_id']
                qtitle = e['formatted_input']['question']['title']
                qdesc = e['formatted_input']['question']['ques_desc']
                codes = e['formatted_input']['answer']['code']
                apis = list()
                for c in codes:
                    tokens = nltk.wordpunct_tokenize(c)
                    for tidx, token in enumerate(tokens):
                        token = token.strip()
                        if tidx >= 0:
                            prev_token = tokens[tidx - 1].strip()[-1]
                            if token == "T":
                                token = "transpose"
                            if (token in api_lists and prev_token == "."):
                                apis.append(token)
                                taken_api_list.add(token)
                api_seq = apis
                if len(api_seq) <= 0:
                    continue
                examples.append({
                    'qid': ques_id,
                    'id': example_id,
                    'q': qtitle.strip().lower(),
                    'd': qdesc.strip().lower(),
                    "apis": api_seq,
                    'link': e['link']
                })
                id_to_link[example_id] = e['link']
                example_id += 1
            except Exception as ex:
                print(ex)
                fail_count += 1
    return examples, taken_api_list


def find_doc(all_about_apis, api):
    for a in all_about_apis:
        if a["api"] == api:
            return a["doc"].lower()

    print(api)
    return None


def remove_punc(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        data = np.char.replace(data, i, ' ')
    return str(data)


def remove_single_char(data):
    new_text = ""
    words = data.split()
    for w in words:
        if len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_stopword(data):
    stop_words = stopwords.words("english")
    words = data.split()
    new_text = ""
    for word in words:
        if word not in stop_words:
            new_text = new_text + " " + word
    return new_text


def stem(data):
    ps = PorterStemmer()
    words = data.split()
    new_text = ""
    for w in words:
        new_text += (" " + ps.stem(w))
    return new_text


def lematize(data):
    ps = WordNetLemmatizer()
    words = data.split()
    new_text = ""
    for w in words:
        new_text += (" " + ps.lemmatize(w))
    return new_text


def dataframe(data):
    words = data.split()
    new_text = ""
    for word in words:
        if word == "df" or word == "dataframe":
            word = "data frame"
        new_text += (" " + word)
    return new_text


def preprocess(data):
    words = nltk.word_tokenize(data)
    data = " ".join(words)
    data = np.char.lower(data)
    data = remove_punc(data)
    data = str(np.char.replace(data, "'", ""))
    data = remove_single_char(data)
    data = remove_stopword(data)
    data = stem(data)
    data = lematize(data)
    data = dataframe(data)
    data = remove_punc(data)
    return data


def calculate_scores(answers, predictions):
    Acc = []
    preds = []
    labels = []
    for key in answers:
        if key not in predictions:
            print("Missing prediction for index {}.".format(key))
        Acc.append(answers[key] == predictions[key])
        preds.append(int(predictions[key]))
        labels.append(int(answers[key]))
    from sklearn.metrics import accuracy_score as acc, precision_score as pr, recall_score as rc, f1_score as f1
    scores = {
        "Acc": acc(labels, preds),
        "Pr": pr(labels, preds),
        "Rc": rc(labels, preds),
        "F1": f1(labels, preds),
    }
    return scores


if __name__ == '__main__':
    api_lists = list(json.load(open("dpr_exp/data/api_list.json")).keys())
    print(api_lists)
    test_examples, _ = get_test_data(api_lists)
    train_examples = get_train_data(api_lists=api_lists, taken_test_qids=[a["qid"] for a in test_examples])
    token_to_index = {token:idx for idx, token in enumerate(api_lists)}
    token_to_index["<s>"] = len(token_to_index.keys())
    token_to_index["</s>"] = len(token_to_index.keys())
    os.mkdir("dpr_exp/data/ngram")
    with open("dpr_exp/data/ngram/vovab.txt", "w") as fp:
        json.dumps(token_to_index, indent=4)
        fp.close()
    with open("dpr_exp/data/ngram/train.txt", "w") as fp:
        for ex in train_examples:
            fp.write((" ".join([s.strip() for s in ex["apis"]])) + "\n")
        fp.close()
    with open("dpr_exp/data/ngram/test.txt", "w") as fp:
        for ex in test_examples:
            fp.write((" ".join([s.strip() for s in ex["apis"]])) + "\n")
        fp.close()
                     
    
    
