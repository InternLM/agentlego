欢迎使用 AgentLego！
=======================

*AgentLego* 是一个开源的工具 API 库，用于扩展和增强基于大型语言模型（LLM）的智能体程序，具有以下突出特点：

- **丰富的工具集，用于扩展 LLM 代理程序的多模态功能** ，包括视觉感知、图像生成和编辑、语音处理和视觉语言推理等。
- **灵活的工具接口** ，允许用户轻松扩展具有多种类型参数和输出的自定义工具。
- **与基于 LLM 的代理程序框架轻松集成** ，如 `LangChain <https://github.com/langchain-ai/langchain>`_, `Transformers Agents <https://huggingface.co/docs/transformers/transformers_agents>`_ 和 `Lagent <https://github.com/InternLM/lagent>`_
- **支持工具服务和远程访问** ，这对于资源消耗较大的机器学习模型（例如 ViT）或有特殊环境要求（例如 GPU 和 CUDA）的工具特别有用。

.. _Getting Started:
.. toctree::
   :maxdepth: 2
   :caption: 快速上手

   get_started.md


.. _Modules:
.. toctree::
   :maxdepth: 2
   :caption: 模块指南

   modules/apis.md
   modules/tool.md
   modules/tool-server.md


.. _Tool APIs:
.. toctree::
   :maxdepth: 1
   :caption: 工具列表
   :glob:

   _tmp/tools/*


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
