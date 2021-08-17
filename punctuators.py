# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 20:57:42 2021

@author: shashank
"""
import torch
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from youtube_transcript_api import YouTubeTranscriptApi

def isValidURL(str):
    
    # Regex to check valid URL
    regex = ("https://youtu.be/\w+")
     
    # Compile the ReGex
    p = re.compile(regex)
     
    # If the string is empty
    # return false
    if (str == None):
        return False
     
    # Return if the string
    # matched the ReGex
    if(re.search(p, str)):
        return True
    else:
        return False

def predict(transcript_text):
    PATH = "shashank2123/t5-base-fine-tuned-for-Punctuation-Restoration"

    tokenizer = AutoTokenizer.from_pretrained( PATH)

    token_data = tokenizer.encode(transcript_text)[:-1]
  
    MAX_LEN = 256
  
    batch_token = [token_data[i:i+MAX_LEN-1] + [1] for i in range(0 , len(token_data) , MAX_LEN-1) ]
  
    batch_token[-1] = batch_token[-1] + [0]*(MAX_LEN - len(batch_token[-1]))
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForSeq2SeqLM.from_pretrained( PATH )

    model.to(torch.device(device))
  
    ids = torch.tensor(batch_token).to(device, dtype = torch.long)
    batch_list = []
    batch_size = 16
    for i in range(0,len(batch_token),batch_size):
        batch_list.append(batch_token[i:i+batch_size])
  
    pred_batch = []
    for batch in batch_list:
        ids = torch.tensor(batch).to(device, dtype = torch.long)
        generated_ids = model.generate(
                        input_ids = ids,
                        max_length=256, 
                        num_beams=2,
                        repetition_penalty=2.5, 
                        length_penalty=1.0, 
                        early_stopping=True
                        )
        pred_batch.append(generated_ids)
  
  
  
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for batch in pred_batch for g in batch]
  
    output_text = " ".join(preds)
  
    return output_text

class URLError(Exception):
    def __init__(self , message):
        super().__init__(message)
        

def get_transcript_by_url(youtube_url):

    video_id = youtube_url.split('/')[-1]

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    except:
        raise URLError("Youtube URL is Incorrect or Video transcript is not accessible")
  
    transcript_text = ''
    for temp in transcript_list:
        if 'text' in temp.keys():
            transcript_text = transcript_text + ' '+ temp['text']

    return transcript_text



if __name__ == '__main__':
    import sys

    args = sys.argv[1:]

    if len(args)!=0:
        url = args[0]
        if isValidURL(url):
            print("Getting transcript from URL : {}".format(url))
            transcript_text = get_transcript_by_url(url)
            print("transcript text is collected....")
            print("Starting Inference...")
            print(predict(transcript_text))
        else:
            raise URLError("URL pattern not matched [Ex :- https://youtu.be/BrmyEWrp22A ]")

    else:
        raise URLError("URL not provided!")
