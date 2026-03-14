const { createApp, ref, nextTick, onMounted } = Vue;

createApp({
  setup() {
    const input = ref("");
    const messages = ref([]);
    const loading = ref(false);
    const messagesRef = ref(null);

    const API_BASE = window.location.origin;

    const quickQuestions = [
      "最早什么时候可以拍？",
      "明天有时间吗？",
      "如何预约？",
    ];

    function scrollToBottom() {
      nextTick(() => {
        const el = messagesRef.value;
        if (el) el.scrollTop = el.scrollHeight;
      });
    }

    async function send() {
      const question = input.value.trim();
      if (!question || loading.value) return;

      // 添加用户消息
      messages.value.push({ role: "user", content: question });
      input.value = "";
      loading.value = true;
      scrollToBottom();

      try {
        const res = await fetch(`${API_BASE}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, top_k: 3 }),
        });

        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }

        const data = await res.json();
        messages.value.push({
          role: "bot",
          content: data.answer,
          sources: data.sources || [],
        });
      } catch (err) {
        messages.value.push({
          role: "bot",
          content: `抱歉，服务暂时不可用，请稍后再试。\n错误信息：${err.message}`,
          sources: [],
        });
      } finally {
        loading.value = false;
        scrollToBottom();
      }
    }

    function sendQuick(q) {
      input.value = q;
      send();
    }

    return {
      input,
      messages,
      loading,
      messagesRef,
      quickQuestions,
      send,
      sendQuick,
    };
  },
}).mount("#app");
