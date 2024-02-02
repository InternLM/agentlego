Welcome to the documentation of AgentLego!
============================================

*AgentLego* is an open-source library of versatile tool APIs to extend and enhance large language model (LLM) based agents, with the following highlight features:

- **Rich set of tools for multimodal extensions of LLM agents** including visual perception, image generation and editing, speech processing and visual-language reasoning, etc.
- **Flexible tool interface** that allows users to easily extend custom tools with arbitrary types of arguments and outputs.
- **Easy integration with LLM-based agent frameworks** like `LangChain <https://github.com/langchain-ai/langchain>`_, `Transformers Agents <https://huggingface.co/docs/transformers/transformers_agents>`_ and `Lagent <https://github.com/InternLM/lagent>`_.
- **Support tool serving and remote accessing**, which is especially useful for tools with heavy ML models (e.g. ViT) or special environment requirements (e.g. GPU and CUDA).


.. _Getting Started:
.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   get_started.md


.. _Modules:
.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules/apis.md
   modules/tool.md
   modules/tool-server.md


.. _Tool APIs:
.. toctree::
   :maxdepth: 1
   :caption: Tool APIs
   :glob:

   _tmp/tools/*


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
