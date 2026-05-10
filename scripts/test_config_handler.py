from utils.config_handler import rag_config,prompts_config,chroma_config

print(rag_config)
print(prompts_config)
print(chroma_config)
print("+"*50)
print(rag_config["chat_model_name"])