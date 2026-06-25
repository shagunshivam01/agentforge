from collections import defaultdict


class ConversationMemory:
    def __init__(self):
        self._messages = defaultdict(list)

    def add(self, user_id: str, role: str, content: str):
        self._messages[user_id].append(
            {
                "role": role,
                "content": content,
            }
        )

    def get(self, user_id: str):
        return self._messages[user_id]

    def clear(self, user_id: str):
        self._messages[user_id] = []
