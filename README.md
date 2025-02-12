# Getting-started-with-GPT-querying
Install OpenAI and LlamaIndex for augmenting and running LLM queries.
```bash
pip install openai llama-index llama-index-readers-web llama-index-embeddings-huggingface
```
For running simple LLM queries on ChatGPT, you can use the following piece of code. Note that you need to generate an OpenAI API token for getting responses from GPT4:
```python
import openai
import sys


class GPT4:
    def __init__(self, system_prompt, temperatrue=0.0):
        openai.api_key = "<YOUR-API-KEY>"
        self.messages = list()
        self.temperature = temperatrue
        self.system_prompt = system_prompt
        system_message = {
            'role': 'system',
            'content': system_prompt
        }    
        self.messages.append(system_message)
    
    def print_system_message(self):
        print("System message:", self.system_prompt)
        
    def get_chat_result(self, user_prompt):
        user_message = {
            'role': 'user',
            'content': user_prompt
        }
        self.messages.append(user_message)
        
        completion_text = openai.chat.completions.create(
            model = "gpt-4o",
            temperature = self.temperature,
            messages = self.messages
        )
        
        response = completion_text.choices[0]
        
        if response.finish_reason != 'stop':
            sys.exit(
                f'Model did not finish properly: {response.finish_reason}')
        
        gpt_message = {
            'role': 'assistant',
            'content': response.message.content
        }
        self.messages.append(gpt_message)
        
        return gpt_message['content']   
```
For our work, the system prompt was hard-coded and the user-prompt was synthesized from the data structures and function descriptions defined in Python using the Eywa API. You can build your own API for this purpose, based on the kind of utilities you wish to support.

LLMs can be expected to produce reasonable code if they have “seen” related text and code during their training time. But if you are working with protocols that are relatively new, you might have to provide relevant information at the time of querying. For this, we can use a technique called Retrieval Augmented Generation (or RAG).

First, we should prepare the document index that will be queried based on the user prompt. If we were using RFCs, we could use the LlamaIndex Webpage reader for this purpose. This library offers many kinds of sophisticated webpage parsers. We are going to use a simple one. 

For other options, check this link: 

https://llamahub.ai/l/readers/llama-index-readers-web?from=readers

Here is the code for building a simple document index:
```python
# import the libraries for loading your embedding model for building the document index
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings


# import the necessary utilities
from llama_index.core import SummaryIndex
# There are many forms of document indexing available
# Follow this link for details:
# https://docs.llamaindex.ai/en/v0.10.18/module_guides/indexing/index_guide.html
# Here we will use SummaryIndex which may not be the most appropriate

from llama_index.readers.web import SimpleWebPageReader

# Choose the embedding model
# If the model does not already exist on your machine, then it will be downloaded
# automatically
Settings.embed_model = HuggingFaceEmbedding(
	model_name="BAAI/bge-small-en-v1.5"
)
# For more embedding models, check this link:
# https://huggingface.co/spaces/mteb/leaderboard

# Alternatively, you could also the OpenAI embeddings, although
# they are known to be not very good. However, if you still want
# to check them out, remove the previous line and uncomment the
# following line:
# Settings.embed_model = OpenAIEmbedding()

# Internally, Llama-Index breaks each document into a smaller chunks each containing
# a fixed number of tokens. There will be overlaps too, but you can adjust both these
# settings as per your wish.
Settings.chunk_size = 1024
Settings.chunk_overlap = 50
# Smaller chunk sizes lead to more precise embeddings, but loss of long-range contextual
# information. However, if you increase the chunk size too much, then it could lead to
# unnecessary details being captured. 
# chunk_overlap helps to reinforce certain parts of the text.
# But it can lead to increased redundancy

# Retrieve the web pages by including them in a list
# Suppose we want to include all the relevant HTTP 3.0 RFCs
# Although here I only use webpage reading, you can have a mixture of different
# document formats too. In that case, you will need to use different types of
# document parsers.
# https://llamahub.ai/l/readers/llama-index-readers-file
documents = SimpleWebPageReader(html_to_text=True).load_data(
	[
		"https://datatracker.ietf.org/doc/html/rfc9114",
		"https://datatracker.ietf.org/doc/html/rfc9110",
		"https://datatracker.ietf.org/doc/html/rfc9111",
		"https://datatracker.ietf.org/doc/rfc9218/"
	]
)
index = SummaryIndex.from_documents(documents)

# Now we can retrieve the relevant context in response to a user query.
# Normally, you can directly use the query engine as such:
# query_engine = index.as_query_engine()
# response = query_engine.query("<your question">)
# But this is not very useful in our case, because we might want to reformat the
# additionally retrieved information to suit our needs.

retriever = index.as_retriever()
response = retriever.retrieve("A function that checks for request cancellation and rejection")

# Let us print the context to see what was generated
print(response)

# Once you have the contextual information from the RFCs, you can attach it to your
# user-prompt and query the LLM using the code shown above.
# This webiste contains a great tutorial on how to build a retriever engine from scratch
# https://docs.llamaindex.ai/en/stable/examples/low_level/retrieval/
```
