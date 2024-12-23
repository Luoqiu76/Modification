from llm import *

config = {
        "azure_endpoint":"https://82634-m50u3n0v-swedencentral.cognitiveservices.azure.com/openai/deployments/gpt-4-32k/chat/completions?api-version=2024-08-01-preview",
        "model":"gpt-4-32k",
        "api_version":"2024-08-01-preview"
    }

client = AsyncAzureOpenAI(
    azure_endpoint=config["azure_endpoint"],
    api_version=config["api_version"],
    api_key="FRMrim2mAZHZFIDuWGjvcMdF7GPax1fth6DYkXc1lq1IPGIinmlrJQQJ99ALACfhMk5XJ3w3AAAAACOGsvYV"
)
sync_client = AzureOpenAI(
    azure_endpoint=config["azure_endpoint"],
    api_version=config["api_version"],
    api_key="FRMrim2mAZHZFIDuWGjvcMdF7GPax1fth6DYkXc1lq1IPGIinmlrJQQJ99ALACfhMk5XJ3w3AAAAACOGsvYV"
)


prompt = "What is the capital of France? "


if __name__ == "__main__":
    llm = AzureLLM("test", None)
    # response = llm.get_response_sync(
    #     prompt=prompt,
    #     max_tokens=100,
    #     is_stream=True,
    #     temperature=0.5,
    #     log=False,
    #     log_message="None"
    # )
    # print(response)
    print("===============================================================")
    response = asyncio.run(llm.get_batch_response(
        prompts=prompt,
        max_tokens=10,
        is_stream=True,
        temperature=0.5,
        log=False,
        log_message="None"
    ))
    print(response)
    print("===============================================================")
    response = asyncio.run(llm.get_response(
        prompt=prompt,
        max_tokens=10,
        is_stream=True,
        temperature=0.5,
        log=False,
        log_message="None"
    ))
    print(response)
    print("===============================================================")

