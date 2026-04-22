const form       = document.getElementById("form");
const chat       = document.getElementById("chat");
const chatEmpty  = document.getElementById("chat-empty");
const questionEl = document.getElementById("question");
const submitBtn  = document.getElementById("submit-btn");

// ── Auto-grow textarea ────────────────────────────────────────────────────────

function autoGrow(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
}

questionEl.addEventListener("input", () => autoGrow(questionEl));

questionEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        form.requestSubmit();
    }
});

// ── Message factory ───────────────────────────────────────────────────────────

function appendMessage(role, text = "") {
    if (chatEmpty) chatEmpty.style.display = "none";

    const wrapper = document.createElement("div");
    wrapper.className = `message message--${role}`;

    const label = document.createElement("div");
    label.className = "message__role";
    label.textContent = role === "user" ? "You" : "Answer";

    const body = document.createElement("div");
    body.className = "message__body";
    body.textContent = text;

    wrapper.appendChild(label);
    wrapper.appendChild(body);
    chat.appendChild(wrapper);
    scrollToBottom();

    return { wrapper, body };
}

function appendChunks(messageWrapper, chunks) {
    const chunksContainer = document.createElement("div");
    chunksContainer.className = "chunks";
    
    const header = document.createElement("div");
    header.className = "chunks__header";
    header.textContent = `${chunks.length} Retrieved Chunks`;
    
    const list = document.createElement("div");
    list.className = "chunks__list";
    
    chunks.forEach(chunk => {
        const chunkEl = document.createElement("div");
        chunkEl.className = "chunk";
        
        const chunkHeader = document.createElement("div");
        chunkHeader.className = "chunk__header";
        
        const chunkId = document.createElement("span");
        chunkId.className = "chunk__id";
        chunkId.textContent = `Chunk ${chunk.chunk_id}`;
        
        const chunkScore = document.createElement("span");
        chunkScore.className = "chunk__score";
        chunkScore.textContent = `${chunk.score.toFixed(3)}`;
        
        const chunkMethod = document.createElement("div");
        chunkMethod.className = "chunk__method";
        
        const methodDot = document.createElement("span");
        methodDot.className = `chunk__method-dot chunk__method-dot--${chunk.method}`;
        
        const methodLabel = document.createElement("span");
        methodLabel.textContent = chunk.method;
        
        chunkMethod.appendChild(methodDot);
        chunkMethod.appendChild(methodLabel);
        
        chunkHeader.appendChild(chunkId);
        chunkHeader.appendChild(chunkScore);
        chunkHeader.appendChild(chunkMethod);
        
        const chunkText = document.createElement("div");
        chunkText.className = "chunk__text";
        chunkText.textContent = chunk.chunk_text.substring(0, 300) + (chunk.chunk_text.length > 300 ? "..." : "");
        
        chunkEl.appendChild(chunkHeader);
        chunkEl.appendChild(chunkText);
        list.appendChild(chunkEl);
    });
    
    chunksContainer.appendChild(header);
    chunksContainer.appendChild(list);
    messageWrapper.appendChild(chunksContainer);
    scrollToBottom();
}

function appendCursor(bodyEl) {
    const cursor = document.createElement("span");
    cursor.className = "cursor";
    bodyEl.appendChild(cursor);
    return cursor;
}

function scrollToBottom() {
    chat.scrollTop = chat.scrollHeight;
}

// ── Streaming submit ──────────────────────────────────────────────────────────

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const question = questionEl.value.trim();
    if (!question) return;

    setInputEnabled(false);
    appendMessage("user", question);
    questionEl.value = "";
    autoGrow(questionEl);

    const { wrapper: answerWrapper, body: answerBody } = appendMessage("answer");
    const cursor = appendCursor(answerBody);

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        if (!response.ok) {
            cursor.remove();
            answerBody.textContent = "Something went wrong. Please try again.";
            answerBody.closest(".message").classList.add("message--error");
            return;
        }

        await streamResponse(response, answerBody, cursor, answerWrapper);

    } catch (err) {
        cursor.remove();
        answerBody.textContent = "Connection error. Is the server running?";
        answerBody.closest(".message").classList.add("message--error");
    } finally {
        setInputEnabled(true);
        questionEl.focus();
    }
});

// ── Server-Sent Events reader ─────────────────────────────────────────────────

async function streamResponse(response, bodyEl, cursor, messageWrapper) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let text = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
            if (line.startsWith("event: chunks")) {
                // Next line should be the data
                continue;
            }
            
            if (line.startsWith("data: ")) {
                const token = line.slice(6);
                if (token === "[DONE]") {
                    cursor.remove();
                    return;
                }
                
                // Check if this is chunks data following an event: chunks line
                const prevLineIndex = lines.indexOf(line) - 1;
                if (prevLineIndex >= 0 && lines[prevLineIndex] === "event: chunks") {
                    try {
                        const chunks = JSON.parse(token);
                        appendChunks(messageWrapper, chunks);
                    } catch (e) {
                        console.error("Failed to parse chunks:", e);
                    }
                    continue;
                }
                
                // Regular text token
                text += token;
                bodyEl.textContent = text;
                bodyEl.appendChild(cursor);
                scrollToBottom();
            }
        }
    }

    cursor.remove();
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function setInputEnabled(enabled) {
    submitBtn.disabled = !enabled;
    questionEl.disabled = !enabled;
}
