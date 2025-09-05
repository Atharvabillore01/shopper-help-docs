async function askQuestion() {
  const input = document.getElementById("chat-input");
  const responseBox = document.getElementById("chat-response");
      const contentBox = document.getElementById("chat-content");
const question = input.value.trim();
 if (!question) return;

 // Show box & reset content
      responseBox.style.display = "block";
      contentBox.innerHTML = "🧠 Thinking...";
  try {
   const response = await fetch("http://localhost:8001/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: question })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullText = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    fullText += chunk;
   contentBox.innerHTML = marked.parse(fullText);
  }

  } catch (err) {
    contentBox.innerText = "Error contacting AI.";
    console.error(err);
  }
}
