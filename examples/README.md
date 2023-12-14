# AgentLego examples

## Lagent

1. Prepare environment.

```bash
pip install lagent
pip install agentlego
```

2. Specify your OpenAI key in environment variables.

```bash
export OPEN_API_KEY=sk-xxx
```

3. Start chatting in terminal.

```bash
python examples/lagent_example.py --model gpt-3.5-turbo --tools Calculator
```

> If your want other tools, install dependencies at first and specify them in the parameters.

```text
User: Please tell me the result of cosine pi/6.
Bot:
Thought: To find the result of cosine pi/6, I can use the calculator tool.

Action: Calculator
Action Input: {"expression": "cos(pi/6)"}
System:Response:0.8660254037844387

Bot:
Final Answer: The result of cosine pi/6 is approximately 0.8660254037844387.
```

## LangChain

1. Prepare environment.

```bash
pip install langchain
pip install agentlego
```

2. Specify your OpenAI key in environment variables.

```bash
export OPEN_API_KEY=sk-xxx
```

3. Start chatting in terminal.

```bash
python examples/langchain_example.py --model gpt-4-1106-preview --tools Calculator
```

> If your want other tools, install dependencies at first and specify them in the parameters.

````text
User: Please tell me the result of cosine pi/6.

> Entering new AgentExecutor chain...
Action:
```
{
  "action": "Calculator",
  "action_input": {
    "expression": "math.cos(math.pi/6)"
  }
}
```
Observation: 0.8660254037844387
Thought:Action:
```
{
  "action": "Final Answer",
  "action_input": "The result of cosine pi/6 is approximately 0.8660254037844387."
}
```

> Finished chain.
gpt-4-1106-preview: The result of cosine pi/6 is approximately 0.8660254037844387.
````

## Streamlit demo

1. Prepare environment.

```bash
pip install lagent
pip install agentlego
pip install streamlit==1.29.0
```

2. Specify your OpenAI key in environment variables.

```bash
export OPEN_API_KEY=sk-xxx
```

3. Start streamlit demo.

```bash
streamlit run examples/lagent_example.py -- --tools Calculator
```

> If your want other tools, install dependencies at first and specify them in the parameters.
