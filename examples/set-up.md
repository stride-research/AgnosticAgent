1. Run with 
```python
      python3 -m examples.LLM.02-tools.run   
```  
2. If using tool calling make sure u the functions are loaded by the python interpreter. Its needed to be included (automatically) into the toolkit. \
If you have the tools in its own indepedent module, say its named `toolkit.py` you can simply do `from . import toolkit`. You can also integrate everything into classes and create \
instances of it. 