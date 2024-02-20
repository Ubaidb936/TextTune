## Set pdf path (------------Required)
pdfPath = None
    #examples
        #1-pdfPath = "https://www.dobs.pa.gov/Documents/Publications/Brochures/The%20Basics%20for%20Investing%20in%20Stocks.pdf"
        #2-pdfPath = "doc.pdf"







##Set hugginface token (----------------------Required)
hf_token = None
    #examples
        #hf_token = "hf_................"







## Set huggingFace llm model id (-------------------Required)
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    #examples
        #1-model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"  (Recommended.........)
        #2-model_id = "mistralai/Mistral-7B-Instruct-v0.2"








##What the document is about (---------------Required)
title = None
    # examples
       # title = "stock market Basics"







##Local file path where you want to store Generated QNA CSV from generateQNA.py (---------------Required)
file_path = "qna.csv"




##datasetName to push it into huggingface (Required only if you want to push the dataset to HuggingFaceHub)
datasetName = "stock_market_basics"
     #examples
        #datasetName = "stock_market_basics"
            #Note: datasetName must use alphanumeric chars or '-', '_', '.', '--' and '..' are forbidden, '-' and '.' cannot start or 
            #end the name, max length is 96








##Some keywords in the doc separated by, (Optional)
keywordsRelatedToPdfDoc = None
    #examples
       #keywordsRelatedToPdfDoc = "stock, stop order, start order."
    