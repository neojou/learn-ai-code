

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCall {
    pub function: ToolFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: Value, // Ollama 會直接解析好 JSON 物件回傳
}

#[derive(Deserialize, Debug, Default)]
pub struct OllamaResponse {
    #[serde(default)]
    pub model: String,
    pub message: Message,
    #[serde(default)]
    pub done: bool,
}

#[derive(Default, Clone)]
pub struct MessageRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub system: Option<String>,
    pub tools: Option<Vec<Value>>,
    pub max_tokens: Option<u32>,
}

pub struct MessageBuilder<'a> {
    client: &'a Ollama,
    request: MessageRequest,
}

impl<'a> MessageBuilder<'a> {
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.request.system = Some(system.into());
        self
    }

    pub fn tools(mut self, tools: Vec<Value>) -> Self {
        self.request.tools = Some(tools);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.request.max_tokens = Some(max_tokens);
        self
    }

    pub async fn send(self) -> Result<OllamaResponse, Box<dyn std::error::Error>> {
        let url = format!("{}/api/chat", self.client.base_url);

        // 1. 插入 System Prompt
        let mut final_messages = self.request.messages.clone();
        if let Some(sys) = &self.request.system {
            final_messages.insert(
                0,
                Message {
                    role: "system".to_string(),
                    content: sys.clone(),
                    tool_calls: None,
                },
            );
        }

        // 2. 組裝符合 Ollama API 規格的 JSON
        let mut payload = json!({
            "model": self.request.model,
            "messages": final_messages,
            "stream": false,
        });

        if let Some(tools) = self.request.tools {
            payload["tools"] = json!(tools);
        }

        if let Some(max_tokens) = self.request.max_tokens {
            payload["options"] = json!({
                "num_predict": max_tokens
            });
        }

        // 3. 發送請求與解析
        let res = self.client.http.post(&url).json(&payload).send().await?;
        let status = res.status();
        let body_text = res.text().await?;

        if !status.is_success() {
            return Err(format!("Ollama API 錯誤 ({}): {}", status, body_text).into());
        }

        let parsed: OllamaResponse = serde_json::from_str(&body_text)
            .map_err(|e| format!("JSON 解析失敗: {} | 原始回傳: {}", e, body_text))?;

        Ok(parsed)
    }
}

pub struct Messages<'a> {
    client: &'a Ollama,
}

impl<'a> Messages<'a> {
    pub fn create(&self, messages: Vec<Message>) -> MessageBuilder<'a> {
        MessageBuilder {
            client: self.client,
            request: MessageRequest {
                model: self.client.model.clone(),
                messages,
                ..Default::default()
            },
        }
    }
}

pub struct Ollama {
    base_url: String,
    model: String,
    http: Client,
}

impl Ollama {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            http: Client::new(),
        }
    }

    pub fn messages(&self) -> Messages<'_> {
        Messages { client: self }
    }
}
