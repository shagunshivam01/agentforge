class ChatbotApp:
    def __init__(self, runtime):
        self.runtime = runtime

    async def run(self, user_id: str, message: str):
        return await self.runtime.execute(
            user_id=user_id,
            input=message
        )
