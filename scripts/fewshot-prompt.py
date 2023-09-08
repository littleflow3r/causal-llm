import os
import sys
import time
import jsonlines as jl
import openai
import pandas as pd
import numpy as np
import re

f_i = sys.argv[1]  # input folder (training data)
f_o = sys.argv[2]  # output folder
i_cv = sys.argv[3]  # cross-val index
n_samples = int(sys.argv[4])  # output folder
m_samples = int(sys.argv[5])  # cross-val index
# # print(f"> {i_cv} (Type: {type(i_cv)}, Length: {len(i_cv)})")
# # sys.exit()
# n_samples = int(input("Number of training samples (n): "))  # 5, 10, 15, 20; 5, 10
# # while not n_samples:
# #   print("Number of training samples cannot be undefined.")
# #   n_samples = input("Number of training samples (n): ")
# m_samples = int(input("Number of test samples (m): "))  # 5, 10, 15, 20; 5, 10
#max_tokens = int(4097 - (n_samples * (4000/(n_samples+m_samples))))

openai.api_key = "" #put your API key here
# path_to_oai_key = "/home/chatGPT_API-key"
# openai.api_key = open(path_to_oai_key, mode="r").readline()

# Function to translate int into str relation
relation_int2str = lambda x: "non-causal" if x==0 else ("causal" if x==1 else x)  

# Reading the training data into a dataframe
#training_data = pd.read_json(f'{f_i}{i_cv}/train2.json', lines=True)  # COMAGC
training_data = pd.read_json(f'{f_i}train2.json', lines=True)

# create the prompt using the real data (data_comagc) with fraction of training data (n=1,5,10,20......)
def prepare_prompt(n=5): 
  # Picking n random rows from the training data
  p_n_ratio = 0
  while not 0.4 <= p_n_ratio < 0.6:  # Inefficient, but does its job: Getting a pos-neg ratio between 40~60%
    n_data = [[training_data.iloc[i]['sentence'], training_data.iloc[i]['relation']] for i in np.random.randint(0, training_data.shape[0], size=n)]
    p_n_ratio = sum([d[1] for d in n_data])/n_samples
    print(f"Training sample pos-neg ratio: {p_n_ratio}", end="\r")
  print()
  #n_data = [[training_data.iloc[i]['sentence'], training_data.iloc[i]['relation']] for i in np.random.randint(0, training_data.shape[0], size=n)]

  # Extract e1 and e2 to provide them in the desired result
  # examples = ""
  training_contexts = ""
  training_answers = ""
  # for sentence, relation in n_data:
  for i, data in enumerate(n_data):
    sentence, relation = data[0], data[1]
    e1a, e1b = sentence.index("<e1>"), sentence.index("</e1>")
    e2a, e2b = sentence.index("<e2>"), sentence.index("</e2>")
    e1 = ' '.join([word for i, word in enumerate(sentence) if e1b > i > e1a])
    e2 = ' '.join([word for i, word in enumerate(sentence) if e2b > i > e2a]) 
    # examples += f"[{' '.join(sentence)}]\n'e1': '{e1}', 'relation': '{relation_int2str(r)}', 'e2': '{e2}'\n\n"
    training_contexts += f"Context #{i+1}: [{' '.join(sentence)}]\n"
    training_answers += f"Result #{i+1}: ['e1': '{e1}', 'relation': '{relation_int2str(relation)}', 'e2': '{e2}]'\n"
    
  # return examples
  return f"{training_contexts}\n{training_answers}\n"


def extract_relation(text: str):
   if "'relation': 'non-causal'" in text: return 0
   elif "'relation': 'causal'": return 1
   else: return "NA"

def send_prompt (text):
  response = openai.Completion.create(
    engine="text-davinci-003",
    #engine="text-davinci-002",
    # engine="text-curie-001",
    prompt=text,
    temperature=0.7,
    # max_tokens=800, #n5,m20, gene
    #max_tokens=550, #n15,m10,gene
    max_tokens=250, #n20,m5,gene
    # max_tokens=max_tokens, 
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  return response.choices[0].text.lstrip()

try:  # create directory
  os.makedirs(f_o)
except FileExistsError:
  pass

n_rows = None
with jl.open(f'{f_i}{i_cv}/test2.json', 'r') as f_test:
#with jl.open(f'{f_i}{i_cv}/test_new.json', 'r') as f_test:
  n_rows = sum([1 for line in f_test.iter()])

with jl.open(f'{f_i}{i_cv}/test2.json', 'r') as f_test, open(f'{f_o}result_{i_cv}.tsv', 'w') as f_out:
#with jl.open(f'{f_i}{i_cv}/test_new.json', 'r') as f_test, open(f'{f_o}result_{i_cv}.tsv', 'w') as f_out:
  f_out.write('Index\tGold\t0/1_Response\tOriginal_Response\tContext\n')
  
  test_data = ""
  test_relations = []
  test_ids = []
  instruction = "Given the context sentence, classify the relationship between the entities marked with e1 and e2 as 'causal' or 'non-causal' relation.\n\n"
  # instruction = "Given each context sentence, classify the relationship between the respective entities (marked with e1 and e2) as 'causal' or 'non-causal' relationship.\n\n"
  # instruction = f"Given each context sentence (#{user_n+1} through #{(user_n*2)+1}), classify the relationship between the respective entities (marked with e1 and e2) as 'causal' or 'non-causal' relationship.\n\n"

  for i, row in enumerate(f_test):
    if (i+1) % m_samples == 0 or (i+1) == n_rows:
        test_data += f"Context #{(i%m_samples)+1}: [" + " ".join(row['sentence']) + "]\n"
        test_relations.append(row['relation'])
        test_ids.append(row['id'])
        prompt = instruction + prepare_prompt(int(n_samples)) + test_data
        print(f"PROMPT:\n\n{prompt}")
        
        # buffer = input("REPLY (press any key): ")
      #  print("REPLY:")
        reply = send_prompt(prompt)
        
      #  [print(l, relation_int2str(r)) for l, r in zip(str(reply).split("\n"), test_relations)]
        result = lambda p,c: f"\033[92m{p}\033[00m" if str(p).split("'relation': ")[1].split(", ")[0].strip("'") == relation_int2str(c) else f"\033[91m{p}\033[00m"
        # [print(result(p, c)) for p, c in zip(str(reply).split("\n"), test_relations)]
        for p, c in zip(str(reply).split("\n"), test_relations):
          try:
            print(result(p, c))
          except:
            pass
        # [f_out.write(f"{t_id}\t{row['relation']}\t{extract_relation(r)}\t{r}\t{str(c).split('[')[1].split(']')[0]}\n") for t_id, r, c in zip(test_ids, str(reply).split("\n"), test_data.split("\n"))]
        for t_id, r, c in zip(test_ids, str(reply).split("\n"), test_data.split("\n")):
          try: 
            f_out.write(f"{t_id}\t{row['relation']}\t{extract_relation(r)}\t{r}\t{str(c).split('[')[1].split(']')[0]}\n")
          except:
            print(f"Couldn't write to file:\n")
  
        print("#"*100)        
        del(prompt)
        test_data = ""
        test_relations = []
        test_ids = []
    else:
      test_data += f"Context #{(i%m_samples)+1}: [" + " ".join(row['sentence']) + "]\n"
      test_relations.append(row['relation'])
      test_ids.append(row['id'])

print()
del(training_data)
