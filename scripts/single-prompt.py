import os
import sys
import openai
import jsonlines
import time
openai.api_key = "" #put your API key here
#path_to_oai_key = "/home/chatGPT_API-key"
#openai.api_key = open(path_to_oai_key, mode="r").readline()
llmodel = "gpt-3.5-turbo"

infd = sys.argv[1]  # input folder (without cross-val idx)
outfd = sys.argv[2]  # output folder
idcv = sys.argv[3]  # cross-val index

def send_request(prompt):
    response = openai.ChatCompletion.create(
        model=llmodel,
        # model="text-davinci-003",
        messages=[
                # message_0
                {"role": "system", "content": "You are a chatbot"},
                # message_1
                # {"role": "system", "content": "You are a cancer expert."},
                # message_2:
                # {"role": "system", "content": "You are a cancer expert. Your job is to assess whether entities in a context sentence are in a causal relationship, or just a correlation at best."},
                # message_3
                #{"role": "system", "content": "You are a medical expert."},
                # 
                {"role": "user", "content": prompt},
            ]
    )
    return response

def prompt_3(infd, outfd, idcv):
    try:  # create directory
        os.makedirs(outfd)
    except FileExistsError:
        pass

    with jsonlines.open(infd+str(idcv)+'/test2.json', 'r') as fop, open(outfd+'result_'+str(idcv)+'.tsv', 'w') as fout:
    # with jsonlines.open(infd+str(idcv)+'/test.json', 'r') as fop, open(outfd+'result_'+str(idcv)+'.tsv', 'w') as fout:
        head = 'Index\tGold\t0/1_Response\tOriginal_Response\tContext\te1\te2\n'
        fout.write(head)
        
        for en, row in enumerate(fop):
            # if row['id'] < 3: #check on the first 3 rows
            # if row['id'] > -1:
            if en > -1:
                try:
                    sentence = row['sentence']
                    e1a, e1b = sentence.index("<e1>"), sentence.index("</e1>")
                    e2a, e2b = sentence.index("<e2>"), sentence.index("</e2>")
                    e1 = ' '.join([w for idx, w in enumerate(sentence) if e1b > idx > e1a])
                    e2 = ' '.join([w for idx, w in enumerate(sentence) if e2b > idx > e2a])

                    remove = ["<e1>","</e1>","<e2>","</e2>" ]
                    sentence = ' '.join([x for x in sentence if x not in remove])
                    
                    # prompt_C:
                    # prompt = f"Given the gene-research related context below, is there a causal relationship between {e1} and {e2}? "
                    # prompt += f"In case the relationhip shows only a correlation, but no strict causation between {e1} and {e2}, answer only, and only, with 'False.' "
                    # prompt += f"In case of uncertainty, answer only, and only, with 'Maybe.' "
                    # prompt += f"In case where there is clearly a causal relationship, and not just a correlation between {e1} and {e2}, answer only, and only, with 'True.'"
                    # prompt += f"\n\nContext: '{sentence}'"

                    #prompt_A:
                    prompt = f"There is a causal relationship between {e1} and {e2}. Answer only, and only, with 'True.' or 'False.'"
                    
                    # prompt_B:
                    # prompt = f"Given the following context, classify the relationship between {e1} and {e2} as a causal or non-causal relationship. Answer only, and only, with 'causal.', or 'non-causal.'.\nContext: '{sentence}'"

                    t_start = time.time()  # starting time
                    response = send_request(prompt)
                    print(f"Requesting #{en} {row['id']} (CrossVal-ID: {idcv}):", e1, e2)
                    print (f"{prompt}\n")
                    result = ''
                    for choice in response.choices:
                        result += choice.message.content

                    if 'false.' in result.lower():
                        response = '0'
                    elif 'true.' in result.lower():
                        response = '1'
                    else:
                        response = 'NA'

                    result = " ".join(result.split())
                    line = f"{row['id']}\t{row['relation']}\t{response}\t{result}\t{sentence}\t{e1}\t{e2}\n"
                    print (row['id'], row['relation'], response, f"-- {e1} -&- {e2} --", result)
                    print(sentence)
                    print("-"*100)
                    fout.write(line)
                    t_passed = t_start - time.time()  #  time passed since start of the request
                    if t_passed < 20:
                        time.sleep(20 - t_passed)
                except:
                    print ('*****\nError, skipped:', row['id'], "\n*****\n")
                    print("-"*100)
                    pass

#prompt_3(infd, outfd, idcv)
#prompt_3('data_sem/', 'eval/sp_sem_A/', 1)
