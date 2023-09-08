import os
import sys
import openai
import jsonlines
import time

openai.api_key = "" #put your API key here
#ft_model = '' # put your model classification format 
#ft_model = '' # put your model extraction format 

infd = sys.argv[1]  # input folder (without cross-val idx)
outfd = sys.argv[2]  # output folder
idcv = sys.argv[3]  # cross-val index

def send_request(prompt, ft_model):
    res = openai.Completion.create(model=ft_model, prompt=prompt, max_tokens=50, temperature=0)
    return (res['choices'][0]['text'])

def run_prompt(infd, outfd, idcv, ft_model):
    try:  # create directory
        os.makedirs(outfd)
    except FileExistsError:
        pass

    #with jsonlines.open(infd+str(idcv)+'/testft.jsonl', 'r') as fop, open(outfd+'result_'+str(idcv)+'.tsv', 'w') as fout:
    with jsonlines.open(infd+str(idcv)+'/testftext.jsonl', 'r') as fop, open(outfd+'result_'+str(idcv)+'.tsv', 'w') as fout:
    
        head = 'Index\tGold\t0/1_Response\tOriginal_Response\tPrompt\n'
        fout.write(head)
        
        for idx, row in enumerate(fop):
            if idx > -1:
            #if idx > 29:
            # if idx < 1:
                try:
                    prompt = row['prompt']
                    print(f"Requesting #{row['id']} (CrossVal-ID: {idcv}):")
                    t_start = time.time()  # starting time
                    result = send_request(prompt, ft_model).strip()
                    # print (f"{prompt}\n")

                    if '\nfalse' in result.lower():
                        response = '0'
                    elif '\ntrue' in result.lower():
                        response = '1'
                    else:
                        response = 'NA'

                    prompt = prompt.replace('\n', '#')
                    line = f"{row['id']}\t{row['relation']}\t{response}\t{result}\t{prompt}\n"
                    # line = str(row['id'])+'\t'+str(row['relation'])+'\t'+response+'\t'+result+'\t'+sentence+f"\t{e1}\t{e2}\n"
                    print (f"{prompt}\n")
                    print (f"ID:{row['id']} Gold:{row['relation']} RESULT:{response}/{result}")
                    print("-"*100)
                    fout.write(line)
                    t_passed = t_start - time.time()  #  time passed since start of the request
                    if t_passed < 20:
                        time.sleep(20 - t_passed)
                except:
                    print ('*****\nError, skipped:', row['id'], "\n*****\n")
                    print("-"*100)
                    continue
        fout.close()


# t_start = time.time()  # starting time
run_prompt(infd, outfd, idcv)

# print (F"Start: {time.time()}")
# ft_model = ''
# run_prompt('data_sem/', 'eval/ft_sem_gptext/', 1, ft_model)
# print (F"End: {time.time()}")
