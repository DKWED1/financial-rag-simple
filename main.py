from service.rag_service import RAGService
from utils.logger import logger

if __name__ == "__main__":
    logger.info("=== 金融风控 RAG 系统启动 ===")

    # 初始化服务（现在是秒级启动）
    rag_service = RAGService()

    while True:
        try:
            question = input("\n[输入问题 (输入'quit'退出)]: ")
            if question.lower() == 'quit':
                break

            answer = rag_service.ask(question)
            print(f"\n🤖 助手: {answer}")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break