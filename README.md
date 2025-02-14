
# Medical-Chatbot


# How to run?
## STEPS:

### STEP 01- Create a conda environment

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```



### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a .env file in the root directory and add your Pinecone credentials as follows:


```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GEMINI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```




```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

NOW,
```bash
open up localhost:
```

### Techstack Used:


 - Python
 - LangChain
 - Flask
 - Meta Llama2
 - Pinecone
 - HTML
 - CSS
