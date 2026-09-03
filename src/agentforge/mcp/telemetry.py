

class MCPTelemetry:
    def __init__(self):
        self.logs = []

    def log_step(self, user_id: str, step):
        self.logs.append({
            "user_id": user_id,
            "step": str(step)
        })

    def get_logs(self):
        return self.logs
