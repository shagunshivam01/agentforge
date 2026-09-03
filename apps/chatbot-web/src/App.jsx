import { useState } from "react";

import { sendMessage } from "./api";

function getSessionId() {
  const existingSessionId = sessionStorage.getItem(
    "agentforge_session_id",
  );

  if (existingSessionId) {
    return existingSessionId;
  }

  const newSessionId = crypto.randomUUID();

  sessionStorage.setItem(
    "agentforge_session_id",
    newSessionId,
  );

  return newSessionId;
}

function App() {
  const [sessionId] = useState(getSessionId);

  const [messages, setMessages] = useState([]);

  const [input, setInput] = useState("");

  const [isLoading, setIsLoading] = useState(false);

  const [error, setError] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();

    const message = input.trim();

    if (!message || isLoading) {
      return;
    }

    setError(null);

    setMessages((current) => [
      ...current,
      {
        role: "user",
        content: message,
      },
    ]);

    setInput("");
    setIsLoading(true);

    try {
      const data = await sendMessage(
        sessionId,
        message,
      );

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: data.response,
        },
      ]);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>AgentForge</h1>
        <span>Direct Agent</span>
      </header>

      <main className="chat">
        {messages.length === 0 && (
          <div className="empty-state">
            <h2>AgentForge Chatbot</h2>
            <p>
              Send a message to start a conversation.
            </p>
          </div>
        )}

        <div className="messages">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${message.role}`}
            >
              <div className="message-role">
                {message.role === "user"
                  ? "You"
                  : "Assistant"}
              </div>

              <div className="message-content">
                {message.content}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message assistant">
              <div className="message-role">
                Assistant
              </div>

              <div className="message-content">
                Thinking...
              </div>
            </div>
          )}
        </div>
      </main>

      <div className="composer-container">
        {error && (
          <div className="error">
            {error}
          </div>
        )}

        <form
          className="composer"
          onSubmit={handleSubmit}
        >
          <input
            type="text"
            value={input}
            onChange={(event) =>
              setInput(event.target.value)
            }
            placeholder="Message AgentForge..."
            disabled={isLoading}
          />

          <button
            type="submit"
            disabled={
              isLoading || !input.trim()
            }
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;

