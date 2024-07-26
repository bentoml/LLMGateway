prepare:
```
export BASE_URL=http://......
export OPENAI_API_KEY=xxx
```

gpt3.5 request:
```
python test.py
```

run again to hit cache:
```
python test.py
```

route to another model:
```
MODEL=llama3.1 python test.py
```

test toxic detection:
```
MODEL=llama3.1 PROMPT="you suck" python test.py
```
