use reqwest::Client;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};

fn empty_if_null<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

fn empty_tool_calls<'de, D>(deserializer: D) -> Result<Option<Vec<ToolCall>>, D::Error>
where
    D: Deserializer<'de>,
{
    let calls = Option::<Vec<ToolCall>>::deserialize(deserializer)?;
    Ok(calls.filter(|calls| !calls.is_empty()))
}

fn skip_empty_tool_calls(calls: &Option<Vec<ToolCall>>) -> bool {
    calls.as_ref().map(|calls| calls.is_empty()).unwrap_or(true)
}

/// Ollama usually returns `arguments` as an object. Some responses send a JSON string.
fn arguments_as_value<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(text) => match serde_json::from_str::<Value>(&text) {
            Ok(parsed) => Ok(parsed),
            Err(_) => Ok(Value::String(text)),
        },
        other => Ok(other),
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Message {
    pub role: String,
    #[serde(default, deserialize_with = "empty_if_null")]
    pub content: String,
    #[serde(
        default,
        deserialize_with = "empty_tool_calls",
        skip_serializing_if = "skip_empty_tool_calls"
    )]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    pub function: ToolFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolFunction {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    #[serde(default, deserialize_with = "arguments_as_value")]
    pub arguments: Value,
}

#[derive(Deserialize, Debug)]
pub struct OllamaResponse {
    pub message: Message,
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

        // Ollama has no top-level system field. Keep it out of the caller's history.
        let mut final_messages = self.request.messages.clone();
        if let Some(sys) = &self.request.system {
            final_messages.insert(
                0,
                Message {
                    role: "system".to_string(),
                    content: sys.clone(),
                    ..Default::default()
                },
            );
        }

        let mut payload = json!({
            "model": self.request.model,
            "messages": final_messages,
            "stream": false,
        });

        if let Some(tools) = self.request.tools {
            payload["tools"] = json!(tools);
        }

        // Do not send `think: false`. With a system prompt, gemma4 then leaks
        // tool calls into `content` instead of `tool_calls`.
        if let Some(max_tokens) = self.request.max_tokens {
            payload["options"] = json!({
                "num_predict": max_tokens
            });
        }

        let res = self.client.http.post(&url).json(&payload).send().await?;
        let status = res.status();
        let body_text = res.text().await?;

        if !status.is_success() {
            return Err(format!("Ollama API error ({status}): {body_text}").into());
        }

        let parsed: OllamaResponse = serde_json::from_str(&body_text).map_err(|err| {
            format!("JSON parse error: {err} | body: {body_text}")
        })?;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_response(raw: &str) -> OllamaResponse {
        serde_json::from_str(raw).unwrap_or_else(|err| panic!("{err}"))
    }

    #[test]
    fn parses_null_content_and_keeps_tool_call_fields() {
        let parsed = parse_response(
            r#"{
                "model": "gemma4",
                "done": true,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "thinking": "ignore me",
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "index": 0,
                            "name": "bash",
                            "arguments": {"command": "ls"}
                        }
                    }]
                }
            }"#,
        );
        assert_eq!(parsed.message.content, "");
        let call = &parsed.message.tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.id.as_deref(), Some("call_abc"));
        assert_eq!(call.call_type.as_deref(), Some("function"));
        assert_eq!(call.function.index, Some(0));
        assert_eq!(call.function.arguments["command"], "ls");

        let replay = serde_json::to_value(&parsed.message).unwrap();
        assert!(replay.get("thinking").is_none());
        assert_eq!(replay["tool_calls"][0]["id"], "call_abc");
        assert_eq!(replay["tool_calls"][0]["type"], "function");
        assert_eq!(replay["tool_calls"][0]["function"]["index"], 0);
        assert_eq!(replay["tool_calls"][0]["function"]["arguments"]["command"], "ls");

        let tool = Message {
            role: "tool".to_string(),
            content: "file.rs".to_string(),
            tool_name: Some("bash".to_string()),
            ..Default::default()
        };
        let value = serde_json::to_value(&tool).unwrap();
        assert_eq!(value["role"], "tool");
        assert_eq!(value["tool_name"], "bash");
        assert_eq!(value["content"], "file.rs");
        assert!(value.get("tool_calls").is_none());
    }

    #[test]
    fn parses_arguments_given_as_json_string() {
        let parsed = parse_response(
            r#"{
                "model": "gemma4",
                "done": true,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "function": {
                            "name": "bash",
                            "arguments": "{\"command\":\"date\"}"
                        }
                    }]
                }
            }"#,
        );
        let args = &parsed.message.tool_calls.unwrap()[0].function.arguments;
        assert_eq!(args["command"], "date");
    }

    #[test]
    fn omits_empty_tool_calls() {
        let message = Message {
            role: "assistant".to_string(),
            content: "done".to_string(),
            tool_calls: Some(vec![]),
            tool_name: None,
        };
        let value = serde_json::to_value(&message).unwrap();
        assert!(value.get("tool_calls").is_none());
        assert!(value.get("tool_name").is_none());
    }
}
