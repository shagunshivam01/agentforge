const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function sendMessage(sessionId, message) {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      message,
    }),
  });

  if (!response.ok) {
    let detail = "Failed to send message.";

    try {
      const data = await response.json();

      if (data.detail) {
        detail = data.detail;
      }
    } catch {
      // Ignore JSON parsing errors.
    }

    throw new Error(detail);
  }

  return response.json();
}

