# LLMGateway

LLMGateway is an example project that demonstrates how to build a gateway application that works with different LLM APIs using [BentoML](https://github.com/bentoml/BentoML). LLMGateway supports private LLM APIs like OpenAI and open-source LLM deployments such as Llama and Mistral. The project offers a unified API interface that makes it easier to work with different LLMs. In addition, LLMGateway demonstrates how to integrate with tools for detecting harmful prompts to keep usage safe and caching to make the LLMs more efficient.

# Prerequisites

- You have installed Python 3.8+ and pip. See the Python downloads page to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read Quickstart first.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the Conda documentation or the Python documentation for details.

# Install dependencies

```
git clone https://github.com/bentoml/BentoSentenceTransformers.git
cd BentoSentenceTransformers
pip install -r requirements.txt
```
# Run the LLM gateway

We have defined a BentoML Service in service.py. Run bentoml serve in your project directory to start the Service on your laptop.

```
$ bentoml serve .
```

The server is now active at http://localhost:3000.

# Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

# Test

Prepare the test client.
```
export BASE_URL=[Local or BentoCloud URL]
export OPENAI_API_KEY=xxx
```

Send a GPT-3.5 request:
```
python test.py
```

Run again to hit cache:
```
python test.py
```

Route to another model:
```
MODEL=llama3.1 python test.py
```

Test toxic detection:
```
MODEL=llama3.1 PROMPT="You are a worthless AI agent!" python test.py
```
