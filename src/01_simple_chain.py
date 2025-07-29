from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

try:
    # load environment variables from .env file (requires `python-dotenv`)
    load_dotenv()
except ImportError:
    pass

model = init_chat_model("gpt-4o-mini", model_provider="openai")

messages = [
    SystemMessage("Translate the following from English into Chinese"),
    HumanMessage(
        "I Have a Dream is a public speech that was delivered by American civil rights activist and Baptist minister Martin Luther King Jr. during the March on Washington for Jobs and Freedom on August 28, 1963.[2] In the speech, King called for civil and economic rights and an end to racism in the United States. Delivered to over 250,000 civil rights supporters from the steps of the Lincoln Memorial in Washington, D.C., the speech was one of the most famous moments of the civil rights movement and among the most iconic speeches in American history.[3][4]"),
]

# 打印发送给GPT的完整prompt
print("\n===== 发送给GPT的Prompt内容 =====")
for message in messages:
    print(f"\n[{message.type.upper()}]")
    print(message.content)

print("\n")

print(model.invoke(messages))
